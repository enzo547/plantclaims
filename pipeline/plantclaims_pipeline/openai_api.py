"""OpenAI API access for extraction jobs.

Use: set OPENAI_API_KEY, `pip install openai`, and give the job an OpenAI model, either in the extraction config
(`model: <OpenAI model id>`, optionally `provider: openai`) or with `job create --model <OpenAI model id>`. Add the model's
prices under `prices.batch` and `prices.sync` of the config (USD per million tokens: input, output, cache_read, cache_write;
cache_write is 0 for OpenAI). `job create` refuses a model without a price, and every job stops at its spend cap. Then run
`job submit` / `job collect` (Batch API) or `job run` (one request per paper) as for any model.

How it works: the job builds every request in one form (plantclaims_pipeline.extract.build_request). This client translates it to the
Responses API and translates each response back, so the store, the validator and the retry round see the same record
whichever provider answered:
  request   system text blocks -> instructions; messages -> input (user / assistant turns; a retry carries the previous
            answer and the validator's error list); output schema -> text.format json_schema; effort -> reasoning.effort;
            max_tokens -> max_output_tokens.
  response  output_text -> the answer text; status `incomplete` with reason max_output_tokens -> stop reason max_tokens;
            a refusal with no text -> stop reason refusal; usage -> input tokens without the cached part, cached input tokens,
            output tokens (reasoning tokens included). The full Responses API body is kept in the stored answer.
Batches go through the Batch API with endpoint /v1/responses and a 24 h completion window. A batch holds at most 50,000
requests and 200 MB, the same limits as batch_max_requests / batch_max_bytes in configs/extract-v0.12.yaml. Output lines
come back in any order and are matched by custom_id; lines of the error file become failed requests, which stay open for
the next `job submit`.

`strict` (config key `openai_strict`, default false) asks the API to enforce the schema exactly. Strict mode accepts only a
subset of JSON Schema; with it off the schema guides the answer and the validator checks every answer as usual.
"""
import io
import json

OPENAI_PREFIXES = ('gpt-', 'o1', 'o3', 'o4')
ENDED = {'completed', 'failed', 'expired', 'cancelled'}


def is_openai(cfg):
    return cfg.get('provider') == 'openai' or str(cfg.get('model') or '').startswith(OPENAI_PREFIXES)


def _text(content):
    if isinstance(content, str):
        return content
    return ''.join(b.get('text', '') for b in content or [] if isinstance(b, dict))


def responses_body(request, strict=False):
    """A request built by plantclaims_pipeline.extract.build_request -> the body of one Responses API call."""
    system = request.get('system') or []
    body = dict(model=request['model'], instructions='\n\n'.join(b['text'] for b in system) if isinstance(system, list) else str(system),
                input=[dict(role=m['role'], content=_text(m['content'])) for m in request['messages']],
                max_output_tokens=request['max_tokens'])
    oc = request.get('output_config') or {}
    if oc.get('effort'):
        body['reasoning'] = dict(effort=oc['effort'])
    if oc.get('format'):
        body['text'] = dict(format=dict(type='json_schema', name='answer', schema=oc['format']['schema'], strict=bool(strict)))
    return body


def message_payload(body, request_id=None):
    """A Responses API body -> the message shape the job stores (content, stop_reason, model, usage), body kept whole."""
    text, refused = '', False
    for item in body.get('output') or []:
        if item.get('type') != 'message':
            continue
        for c in item.get('content') or []:
            if c.get('type') == 'output_text':
                text += c.get('text') or ''
            elif c.get('type') == 'refusal':
                refused = True
    status, reason = body.get('status'), (body.get('incomplete_details') or {}).get('reason')
    if refused and not text:
        stop = 'refusal'
    elif status == 'incomplete':
        stop = 'max_tokens' if reason == 'max_output_tokens' else (reason or 'incomplete')
    elif status == 'completed':
        stop = 'end_turn'
    else:
        stop = status or 'unknown'
    u = body.get('usage') or {}
    cached = (u.get('input_tokens_details') or {}).get('cached_tokens') or 0
    usage = dict(input_tokens=max(0, (u.get('input_tokens') or 0) - cached), output_tokens=u.get('output_tokens') or 0,
                 cache_read_input_tokens=cached, cache_creation_input_tokens=0)
    return dict(id=request_id or body.get('id'), type='message', role='assistant', model=body.get('model'),
                content=[dict(type='text', text=text)], stop_reason=stop, usage=usage, openai_response=body)


def _dump(obj):
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    return dict(obj)


class _Payload(dict):
    """A stored message that also carries the service request id, as the Anthropic SDK objects do."""


class _Batches:
    def __init__(self, client, strict):
        self._c, self._strict = client, strict

    def create(self, requests):
        lines = [json.dumps(dict(custom_id=r['custom_id'], method='POST', url='/v1/responses', body=responses_body(r['params'], self._strict)),
                            ensure_ascii=False) for r in requests]
        f = self._c.files.create(file=('batch.jsonl', io.BytesIO(('\n'.join(lines) + '\n').encode('utf-8'))), purpose='batch')
        b = _dump(self._c.batches.create(input_file_id=f.id, endpoint='/v1/responses', completion_window='24h'))
        return dict(id=b.get('id'), processing_status='ended' if b.get('status') in ENDED else 'in_progress', openai_status=b.get('status'),
                    input_file_id=f.id)

    def retrieve(self, batch_id):
        b = _dump(self._c.batches.retrieve(batch_id))
        return dict(id=batch_id, processing_status='ended' if b.get('status') in ENDED else 'in_progress', openai_status=b.get('status'))

    def results(self, batch_id):
        b = _dump(self._c.batches.retrieve(batch_id))
        for fid in (b.get('output_file_id'), b.get('error_file_id')):
            if not fid:
                continue
            for line in self._c.files.content(fid).text.splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                resp = d.get('response') or {}
                if d.get('error') is None and resp.get('status_code') == 200:
                    yield dict(custom_id=d['custom_id'], result=dict(type='succeeded', message=message_payload(resp.get('body') or {}, resp.get('request_id'))))
                else:
                    code = (d.get('error') or {}).get('code')
                    yield dict(custom_id=d['custom_id'], result=dict(type='expired' if code == 'batch_expired' else 'errored',
                                                                     error=d.get('error') or dict(status_code=resp.get('status_code'))))


class _Messages:
    def __init__(self, client, strict):
        self._c, self._strict = client, strict
        self.batches = _Batches(client, strict)

    def create(self, **request):
        r = self._c.responses.create(**responses_body(request, self._strict))
        rid = getattr(r, '_request_id', None)
        p = _Payload(message_payload(_dump(r), rid))
        p._request_id = rid
        return p


class OpenAIClient:
    """The client interface the job uses (messages.create, messages.batches.create / retrieve / results) over the OpenAI SDK."""

    def __init__(self, client=None, strict=False):
        if client is None:
            try:
                import openai
            except ImportError:
                raise RuntimeError('openai SDK is not installed; `pip install openai` and set OPENAI_API_KEY first')
            client = openai.OpenAI()
        self.messages = _Messages(client, strict)

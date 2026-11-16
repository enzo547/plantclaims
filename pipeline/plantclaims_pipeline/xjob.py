"""Extraction jobs over the store: a job is a paper selection plus one specification. It sends the unanswered questions to the
Anthropic API (Message Batches or one request per paper), collects and validates the answers, and sends one validator-feedback
retry where the configuration allows it. Nothing is sent unless `job submit` or `job run` is called."""
import json
import time
from pathlib import Path

from .derive import Deriver
from .extract import (METHOD_API, ModelCaller, attempt_of, build_registry, build_request, components, output_schema, price, request_settings, sample_papers, select_papers, system_prompt)
from .openai_api import OpenAIClient, is_openai
from .records import digest, now
from .relevance import validate_api_provenance
from .store import STORE_VERSION, Store, input_key, make_spec, request_key

CHANNEL_METHOD = dict(api=METHOD_API)


# Message Batches accepts 100,000 requests or 256 MB per batch, whichever comes first. Every request repeats the prompt,
# the registry and the output schema (about 41 KB with extract-v0.6), so BYTES bind first: roughly 5,000 papers a batch.
BATCH_MAX_BYTES = 200_000_000
BATCH_MAX_REQUESTS = 50_000
COLLECT_CHUNK = 500        # results are stored and validated in chunks, so a large batch never sits in memory as records
MAX_REQUEST_FAILURES = 2   # a question whose request errored/expired this often is no longer resent automatically (e.g. a schema the API rejects)
API_SCHEMA = '''
CREATE TABLE IF NOT EXISTS api_batches (
  job_id TEXT NOT NULL, seq INTEGER NOT NULL, round INTEGER NOT NULL, batch_id TEXT NOT NULL, n_requests INTEGER NOT NULL, bytes INTEGER NOT NULL,
  status TEXT NOT NULL, submitted_at TEXT NOT NULL, ended_at TEXT, succeeded INTEGER, failed_json TEXT, cost_usd REAL, PRIMARY KEY (job_id, seq));
CREATE TABLE IF NOT EXISTS api_requests (
  custom_id TEXT PRIMARY KEY, job_id TEXT NOT NULL, batch_seq INTEGER, input_key TEXT NOT NULL, attempt INTEGER NOT NULL, retry_of TEXT,
  request_sha256 TEXT NOT NULL, feedback_json TEXT, failures INTEGER NOT NULL DEFAULT 0) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS api_requests_batch ON api_requests(job_id, batch_seq);
'''


def custom_id(rkey):
    """Batch custom_id must match [A-Za-z0-9_-]{1,64}; a paper id is a URL. The question's own key (first 32 hex digits) is safe and unique."""
    return rkey[:32]


def input_record(paper, cleaning_version, source):
    """A packet paper (extract.sample_papers shape) as a store input: the same dict plus its key and where it came from."""
    return dict(paper, input_key=input_key(paper['paper_id'], paper['text_sha256'], cleaning_version), cleaning_version=cleaning_version, source=source)


class Job:
    def __init__(self, run, store=None):
        self.run, self.cfg = run, run.config['extract']
        self.store = store or Store(run.root, readonly=run.readonly or run.replay)
        self.deriver = Deriver(self.store)
        if not self.store.readonly:
            self.store.db.executescript(API_SCHEMA)

    # ---- create: spec + selection, inputs built for what is missing ----
    def create(self, selection, channel, limit=None, cells=None, paper_ids=None, pool=None, workers=None):
        """selection = sample:<pilot_sample run> (narrowed by limit / cells / paper_ids), pool:<candidate_pool run>[:N]
        (N = a fixed pseudo-random subset), or ids:<file> together with pool=<candidate_pool run>."""
        if self.run.replay or self.run.readonly:
            raise ValueError('create writes the job')
        if 'spec_key' in self.run.manifest:
            raise ValueError('Job already created')
        if channel != 'api':
            raise ValueError('channel must be api')
        kind, _, ref = selection.partition(':')
        if kind not in ('sample', 'pool', 'ids') or not ref:
            raise ValueError('selection must be sample:<run>, pool:<run>[:N] or ids:<file> (with --pool)')
        request_settings(self.cfg)           # request-v0.8.0: a thinking / effort combination the model rejects fails before a spec is stored
        if self.cfg['model'] not in (self.cfg.get('prices') or {}).get(self.cfg['mode'], {}):
            raise ValueError(f"No {self.cfg['mode']} price for {self.cfg['model']} in the config: unpriced answers would cost 0 and the spend cap would not hold")
        cleaning = self.cfg.get('text_cleaning')
        version = (cleaning or {}).get('version') or 'none'
        reg = build_registry(self.run)
        spec = make_spec(system_prompt(self.run, reg), reg, output_schema(reg), self.cfg, self.cfg['input_fields'],
                         components()['request_builder']['version'])
        skey = self.store.put_spec(spec)
        if kind == 'sample':
            sel, n, built = self._select_sample(ref, skey, channel, cleaning, version, limit, cells, paper_ids)
        else:
            if limit or cells or paper_ids:
                raise ValueError('--limit / --cell / --paper-ids narrow a sample selection only')
            sel, n, built = self._select_from_index(kind, ref, pool, skey, channel, cleaning, version, workers)
        self.run.manifest.update(spec_key=skey, selection=sel, selection_sha256=self.store.job(self.run.id)['selection_sha256'], n_papers=n, channel=channel,
                                 store_version=STORE_VERSION, inputs_built=built, inputs_reused=n - built, components=components(), role=self.cfg['role'],
                                 gate=self.cfg['gate'], model_requested=self.cfg['model'], model_api_requests=0, ingests=[], derivations=[],
                                 formal_corpus_membership=False)
        pending = self.open_counts()['unanswered']
        if pending:
            self.run.finish('awaiting_model_output')
        else:                                   # every question of this job is already answered in the store
            self._finish(self.deriver.stats(self.derivation(), self.run.id))
        return dict(job_id=self.run.id, spec_key=skey, papers=n, inputs_built=built, inputs_reused=n - built, channel=channel,
                    already_answered=n - pending, selection=sel, note='no model request has been sent')

    def _select_sample(self, ref, skey, channel, cleaning, version, limit, cells, paper_ids):
        spath = self.run.root / 'data' / 'runs' / ref / 'manifest.json'
        if not spath.exists():
            raise ValueError('Sample run does not exist')
        smanifest = json.loads(spath.read_text())
        if smanifest.get('kind') != 'pilot_sample' or smanifest.get('status') != 'sample_frozen':
            raise ValueError('Sample run must be a frozen pilot_sample run')
        papers = select_papers(sample_papers(self.run.root, ref, cleaning), limit, cells, paper_ids)
        if not papers:
            raise ValueError('No papers selected')
        source = dict(kind='pilot_sample', run_id=ref, manifest_sha256=digest(spath.read_bytes()))
        inputs = [input_record(p, version, source) for p in papers]
        built = self.store.put_inputs(inputs)
        sel = dict(kind='sample', source_run=ref, limit=limit, cells=sorted(cells) if cells else None,
                   paper_ids_sha256=digest(sorted(str(i).rsplit('/', 1)[-1] for i in paper_ids)) if paper_ids else None, cleaning_version=version)
        self.store.put_job(self.run.id, skey, [(r['paper_id'], r['input_key']) for r in inputs], sel, channel)
        return sel, len(inputs), built

    def _select_from_index(self, kind, ref, pool, skey, channel, cleaning, version, workers):
        """pool: / ids: selections are located through the snapshot index and built from the retained match files
        (xinput). The real-abstract rule, the frame and the strata are those of the pilot sample config, frozen here."""
        import csv
        import gzip
        import os
        import time

        import yaml

        from . import xinput
        scfg_path = self.run.root / 'configs' / 'pilot-sample-v0.yaml'
        scfg = yaml.safe_load(scfg_path.read_text())
        n_limit = None
        if kind == 'pool':
            pool, _, lim = ref.partition(':')
            n_limit = int(lim) if lim else None
        elif not pool:
            raise ValueError('ids:<file> needs --pool <candidate_pool run> (it names the snapshot the ids are looked up in)')
        bound = xinput.bind_pool(self.run.root, self.run.db, pool, self.run.config['scope'])
        where = xinput.frame_sql(bound['rule_sql'], scfg['frame'])
        t0, outside = time.monotonic(), []
        if kind == 'pool':
            groups = xinput.locate_pool(self.run.db, bound['source_run_id'], where, n_limit)
        else:
            ids = [l.strip() for l in Path(ref).read_text().splitlines() if l.strip() and not l.startswith('#')]
            groups, outside = xinput.locate_ids(self.run.db, bound['source_run_id'], where, ids)
        source = dict(kind='candidate_pool', pool_id=pool, source_run_id=bound['source_run_id'])
        sel = dict(kind=kind, pool=bound, frame=scfg['frame'], abstract_rule_sha256=digest(scfg['abstract_rule']), sample_config_sha256=digest(scfg_path.read_bytes()),
                   limit=n_limit, order='source_row_sha256' if n_limit else 'raw_path', cleaning_version=version,
                   **({'ids_file_sha256': digest(Path(ref).read_bytes())} if kind == 'ids' else {}))
        self.store.begin_job(self.run.id, skey, sel, channel)
        stats, excluded = xinput.build(self.run.root, self.store, groups, scfg['abstract_rule'], scfg['strata'], cleaning, source,
                                       workers=workers or min(int(scfg['scan']['workers']), os.cpu_count() or 1), decompressor=scfg['scan']['decompressor'],
                                       on_papers=lambda pairs: self.store.add_job_papers(self.run.id, pairs))
        n = stats['built'] + stats['reused']
        if not n:
            raise ValueError('No papers selected')
        with gzip.open(self.run.directory / 'input-exclusions.csv.gz', 'wt', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(('openalex_id', 'reasons'))
            w.writerows(excluded)
            w.writerows((i, 'outside_pool_frame_or_unknown') for i in outside)
        sel['build'] = dict(stats, outside_frame_or_unknown=len(outside), seconds_total=round(time.monotonic() - t0, 2),
                            seconds_per_1000_papers=round((time.monotonic() - t0) / max(n, 1) * 1000, 2))
        self.store.finish_job(self.run.id, sel)
        return sel, n, stats['built']

    def _finish(self, stats):
        self.run.manifest.update(candidate_stats=stats, processed_at=now())
        self.run.finish('extraction_in_progress' if stats['pending'] else
                        ('candidates_with_quarantine' if stats['quarantined_papers'] or stats['claims_quarantined'] else 'candidates_ready_unreviewed'))

    def set_policy(self, min_core_claims, reason):
        """Explicit, audited job policy; neither the question's spec nor older derivations is rewritten."""
        if self.run.readonly or self.run.replay:
            raise ValueError('Changing job policy writes the ledger')
        if type(min_core_claims) is not int or min_core_claims not in (0, 1):
            raise ValueError('min_core_claims must be 0 or 1')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('A job policy change needs an audit reason')
        policy = dict(self.run.manifest.get('validation_policy') or {})
        old = dict(policy)
        if min_core_claims:
            policy['min_core_claims'] = min_core_claims
        else:
            policy.pop('min_core_claims', None)
        Deriver.current_id(self.spec_key, policy)  # reject unsupported policies before writing
        if policy != old:
            self.run.manifest.setdefault('policy_changes', []).append(dict(before=old, after=policy, reason=reason, at=now()))
            self.run.manifest['validation_policy'] = policy
        did = self.derivation()
        self._finish(self.deriver.stats(did, self.run.id))
        return dict(job_id=self.run.id, validation_policy=policy, derivation_id=did)


    # ---- what is missing ----
    @property
    def spec_key(self):
        if 'spec_key' not in self.run.manifest:
            raise ValueError('Job is not created yet')
        return self.run.manifest['spec_key']

    @property
    def channel(self):
        return self.run.manifest['channel']

    def pending_rows(self, n=None):
        """Job papers whose first-attempt question has no stored answer, in job order."""
        sql = ('SELECT j.paper_id, j.input_key FROM job_papers j LEFT JOIN answers a ON a.input_key = j.input_key AND a.spec_key = ? AND a.channel = ? '
               'AND a.attempt = 1 AND a.replicate = 0 WHERE j.job_id = ? AND a.request_key IS NULL ORDER BY j.seq')
        return self.store.db.execute(sql + (' LIMIT ?' if n else ''), [self.spec_key, self.channel, self.run.id] + ([n] if n else [])).fetchall()


    def pending_pages(self, page=5000):
        """Pending rows in job order without holding millions of rows: keyset pages over job_papers.seq."""
        sql = ('SELECT j.seq, j.paper_id, j.input_key FROM job_papers j LEFT JOIN answers a ON a.input_key = j.input_key AND a.spec_key = ? AND a.channel = ? '
               'AND a.attempt = 1 AND a.replicate = 0 WHERE j.job_id = ? AND j.seq > ? AND a.request_key IS NULL ORDER BY j.seq LIMIT ?')
        last = -1
        while True:
            rows = self.store.db.execute(sql, (self.spec_key, self.channel, self.run.id, last, page)).fetchall()
            if not rows:
                return
            yield from rows
            last = rows[-1]['seq']

    def open_counts(self):
        """Index-only counts of what the job could still ask (no input is read, no request is built)."""
        db, limit = self.store.db, int(self.store.get_spec(self.spec_key)['registry'].get('max_attempts', 1))
        unanswered = db.execute('SELECT count(*) FROM job_papers j LEFT JOIN answers a ON a.input_key = j.input_key AND a.spec_key = ? AND a.channel = ? AND a.attempt = 1 '
                                'AND a.replicate = 0 WHERE j.job_id = ? AND a.request_key IS NULL', (self.spec_key, self.channel, self.run.id)).fetchone()[0]
        did, retryable = self.derivation(write=False), 0
        if self.deriver.exists(did):
            scope, args = self.deriver._scope(did, self.run.id, False, 'derived_papers')
            retryable = db.execute("SELECT count(*) " + scope + " AND (t.status = 'quarantined' OR t.claims_quarantined > 0 OR t.claims_review > 0) AND t.attempt < ?",
                                   args + [limit]).fetchone()[0]
        inflight = given_up = 0
        if db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='api_requests'").fetchone():
            inflight = db.execute("SELECT count(*) FROM api_requests r JOIN api_batches b ON b.job_id = r.job_id AND b.seq = r.batch_seq "
                                  "WHERE r.job_id = ? AND b.status = 'submitted'", (self.run.id,)).fetchone()[0]
            given_up = db.execute('SELECT count(*) FROM api_requests r WHERE r.job_id = ? AND r.failures >= ? AND NOT EXISTS (SELECT 1 FROM answers a WHERE substr(a.request_key, 1, 32) = r.custom_id)',
                                  (self.run.id, MAX_REQUEST_FAILURES)).fetchone()[0]
        return dict(unanswered=unanswered, retryable=retryable, in_flight=inflight, failed_not_resent=given_up, open=max(0, unanswered + retryable - inflight - given_up))

    # ---- derivation of this job under the current validator ----
    def derivation(self, write=True):
        policy = self.run.manifest.get('validation_policy') or {}
        if not write:
            return Deriver.current_id(self.spec_key, policy)
        did = self.deriver.ensure(self.spec_key, policy)
        if did not in [d['derivation_id'] for d in self.run.manifest['derivations']]:
            self.run.manifest['derivations'].append(dict(derivation_id=did, validator=components()['validator'], policy=policy, first_used_at=now()))
        behind = self.deriver.behind(did)
        if behind:       # new derivation (validator changed) or answers that arrived through another job: catch up, no model involved
            self.deriver.derive_missing(did)
        return did

    # ---- ingest: only new records are validated ----
    def _accept(self, records, source, retry=False, sent_feedback=None, ledger=True):
        """`sent_feedback` ({paper_id: error list}) is given by the API path only: a retry answer that was paid for is
        stored with the feedback that was REALLY sent, even if the validator changed between submit and collect."""
        if self.run.replay or self.run.readonly:
            raise ValueError('ingest writes the store')
        if not records:
            raise ValueError('No records to ingest')
        skey, channel, db = self.spec_key, self.channel, self.store.db
        limit = int(self.store.get_spec(skey)['registry'].get('max_attempts', 1))
        did = self.derivation()
        seen, answers = set(), []
        for r in sorted(records, key=attempt_of):
            if not isinstance(r, dict) or set(r) < {'paper_id', 'method', 'model_provenance', 'result'}:
                raise ValueError('Each record needs paper_id, method, model_provenance, result')
            if r['method'] != CHANNEL_METHOD[channel]:
                raise ValueError(f"This job's channel is {channel}; a {r['method']} record does not belong to it")
            validate_api_provenance(r['model_provenance'])
            row = db.execute('SELECT input_key FROM job_papers WHERE job_id=? AND paper_id=?', (self.run.id, r['paper_id'])).fetchone()
            if row is None:
                raise ValueError('Record for a paper outside this job: ' + str(r['paper_id']))
            a = attempt_of(r)
            if (r['paper_id'], a) in seen:
                raise ValueError('Duplicate record for ' + r['paper_id'])
            seen.add((r['paper_id'], a))
            result = r['result']
            body = dict(result=result, **({'payload': r['payload']} if 'payload' in r else {}))
            meta = dict(method=r['method'], source=source, **{k: r[k] for k in ('mode', 'batch_id') if r.get(k) is not None})
            if a > 1:
                if not retry:
                    raise ValueError('A retry record needs --retry: ' + r['paper_id'])
                prev = db.execute('SELECT request_key, input_key, spec_key, channel, attempt, replicate FROM answers WHERE request_key=?', (r.get('retry_of'),)).fetchone()
                if prev is None or (prev['input_key'], prev['spec_key'], prev['channel'], prev['attempt'], prev['replicate']) != (row['input_key'], skey, channel, a - 1, 0):
                    raise ValueError('retry_of must be the request key of the previous attempt of the same question (see `job feedback`): ' + r['paper_id'])
                key = request_key(row['input_key'], skey, channel, a, prev['request_key'], 0)
                cached = db.execute('SELECT body_sha256, meta_json FROM answers WHERE request_key=?', (key,)).fetchone()
                if cached is not None:
                    if cached['body_sha256'] != digest(body):
                        raise ValueError('A different answer is already stored for this request; the first answer is never overwritten: ' + r['paper_id'])
                    # A later policy change cannot revoke an already stored answer's admission.
                    # A repeated import keeps the stored answer's provenance.
                    meta = json.loads(cached['meta_json'])
                    if r.get('validator_feedback', meta.get('validator_feedback', [])) != meta.get('validator_feedback', []):
                        raise ValueError('validator_feedback differs from the stored retry feedback: ' + r['paper_id'])
                else:
                    if a > limit:
                        raise ValueError(f"attempt {a} exceeds max_attempts={limit}: {r['paper_id']}")
                    if sent_feedback is not None:
                        expected = sent_feedback[r['paper_id']]
                    else:
                        expected = self.deriver.feedback_for(did, prev['request_key'])
                        if not expected:
                            raise ValueError('The previous attempt was fully accepted; nothing to retry: ' + r['paper_id'])
                        if r.get('validator_feedback', expected) != expected:
                            raise ValueError('validator_feedback differs from what the validator reports: ' + r['paper_id'])
                    meta['validator_feedback'] = expected
            elif 'retry_of' in r or 'validator_feedback' in r:
                raise ValueError('A first attempt must not carry retry_of / validator_feedback: ' + r['paper_id'])
            answers.append(dict(input_key=row['input_key'], spec_key=skey, paper_id=r['paper_id'], channel=channel, attempt=a, retry_of=r.get('retry_of') if a > 1 else None,
                                replicate=0, status=(result.get('status') if isinstance(result, dict) else None) or 'unparseable',
                                provenance=r['model_provenance'], body=body, meta=meta))
        put = self.store.put_answers(answers, job_id=self.run.id)
        derived = self.deriver.derive(did, put['new'])
        entry = dict(source=source, records=len(records), new=len(put['new']), cached=len(put['cached']), derived=derived, derivation_id=did, at=now(),
                     attempts=sorted({attempt_of(r) for r in records}))
        if not ledger:
            return dict(entry, new_keys=put['new'])
        self.run.manifest['ingests'].append(entry)
        stats = self.deriver.stats(did, self.run.id)
        self._finish(stats)
        return dict(entry, stats=stats, new_papers=self.deriver.paper_rows(did, put['new']))


    # ---- API path (needs the provider's SDK and credentials; nothing is sent unless submit / run_sync is called) ----
    def _client(self, client):
        if client is not None:
            return client
        if is_openai(self.cfg):
            return OpenAIClient(strict=bool(self.cfg.get('openai_strict', False)))
        try:
            import anthropic
        except ImportError:
            raise RuntimeError('anthropic SDK is not installed; `pip install anthropic` and set ANTHROPIC_API_KEY first')
        return anthropic.Anthropic()

    def _api_ready(self):
        if self.run.replay or self.run.readonly:
            raise ValueError('sending writes the ledger')
        now_rb, then_rb = components()['request_builder'], self.run.manifest['components']['request_builder']
        if now_rb['fingerprint'] != then_rb['fingerprint']:
            raise ValueError(f"request_builder changed since this job was created ({then_rb['version']} -> {now_rb['version']}); "
                             'requests of one job are built one way: create a new job (stored answers stay cached under their own spec)')
        spec = self.store.get_spec(self.spec_key)
        return spec, dict(request=dict(spec['request'], cache_ttl=self.cfg['cache_ttl']), system_prompt=spec['system_prompt'], registry=spec['registry'],
                          input_fields=spec['input_fields'], output_schema=spec['output_schema'])

    def spent(self):
        """USD the service reported for answers THIS job paid for (cached answers of other jobs cost this job nothing)."""
        return round(self.store.db.execute("SELECT coalesce(sum(json_extract(provenance_json, '$.cost_usd')), 0) FROM answers WHERE job_id = ? AND channel = 'api'",
                                           (self.run.id,)).fetchone()[0], 6)

    def _cap(self):
        if self.spent() >= self.cfg['spend_cap_usd']:
            raise RuntimeError(f"spend cap {self.cfg['spend_cap_usd']} USD reached (spent {self.spent()}); not sending")

    def _attempt(self, **fields):
        entry = dict(seq=len(self.run.manifest['attempts']) + 1, started_at=now(), **fields)
        self.run.manifest['attempts'].append(entry)
        self.run.flush()
        return entry

    def _provenance(self, payload, request_id, mode):
        usage = payload.get('usage') or {}
        table = self.cfg['prices'][mode]
        cost = price(table, payload.get('model'), usage)
        if cost is None:                # a dated model id in the response is priced as the model the job asked for
            cost = price(table, self.cfg.get('model'), usage)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8')
        return dict(provider='openai' if is_openai(self.cfg) else 'anthropic', model_id=payload.get('model'), request_id=request_id, effort=self.cfg['effort'], stop_reason=payload.get('stop_reason'),
                    input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'), cost_usd=cost, price_checked_on=self.cfg['prices']['checked_on'],
                    accounting_status='api_reported' if cost is not None else 'api_reported_unpriced', human_review='not_performed', raw_sha256=digest(raw))

    def _record(self, paper, payload, request_id, mode, batch_id, attempt, retry_of):
        rec = ModelCaller.result_record(paper, payload, self._provenance(payload, request_id, mode), dict(mode=mode, batch_id=batch_id, attempt=attempt))
        rec['payload'] = payload                                   # the full service response is part of the stored answer body
        if attempt > 1:
            rec['retry_of'] = retry_of
        return rec

    def _questions(self, packet, limit=None):
        """(paper, attempt, retry_of, feedback, request params) for what this job may still ask: unanswered first
        attempts, then validator-feedback retries that are not already in flight."""
        db, did = self.store.db, self.derivation()
        def skip(cid):     # already in a batch that has not ended, or its request failed too often to be resent blindly
            r = db.execute("SELECT r.failures, (SELECT b.status FROM api_batches b WHERE b.job_id = r.job_id AND b.seq = r.batch_seq) FROM api_requests r "
                           "WHERE r.custom_id = ? AND r.job_id = ?", (cid, self.run.id)).fetchone()
            return r is not None and (r[1] == 'submitted' or r[0] >= MAX_REQUEST_FAILURES)
        n = 0
        for row in self.pending_pages():
            rkey = request_key(row['input_key'], self.spec_key, 'api', 1)
            if skip(custom_id(rkey)):
                continue
            paper = self.store.get_input(row['input_key'])
            yield paper, rkey, 1, None, None, build_request(packet, paper)
            n += 1
            if limit and n >= limit:
                return
        for f in self.deriver.feedback(did, self.run.id, int(packet['registry'].get('max_attempts', 1)))['retry']:
            prev = self.store.get_answer(f['retry_of'])
            rkey = request_key(prev['input_key'], self.spec_key, 'api', f['next_attempt'], f['retry_of'])
            if skip(custom_id(rkey)) or 'payload' not in prev['body']:
                continue
            paper = self.store.get_input(prev['input_key'])
            retry = ModelCaller.retry_turn(None, prev['body']['payload'], f['feedback'])
            yield paper, rkey, f['next_attempt'], f['retry_of'], f['feedback'], build_request(packet, paper, retry)
            n += 1
            if limit and n >= limit:
                return

    def submit(self, client=None, max_batches=1, limit=None):
        """Create up to `max_batches` Message Batches from the questions that are still open. A batch is cut by BYTES
        (and by request count); each gets one ledger entry and one api_batches row, each request one api_requests row."""
        spec, packet = self._api_ready()
        client = self._client(client)
        max_bytes, max_requests = int(self.cfg.get('batch_max_bytes', BATCH_MAX_BYTES)), int(self.cfg.get('batch_max_requests', BATCH_MAX_REQUESTS))
        out, chunk, size = [], [], 0

        def send():
            nonlocal chunk, size
            self._cap()
            seq = (self.store.db.execute('SELECT coalesce(max(seq), 0) FROM api_batches WHERE job_id=?', (self.run.id,)).fetchone()[0]) + 1
            rnd = max(q[2] for q in chunk)
            requests = [dict(custom_id=custom_id(q[1]), params=q[5]) for q in chunk]
            try:
                batch = client.messages.batches.create(requests=requests)
            except Exception as exc:                 # ledger the failure, never retry silently
                self._attempt(step='batch_create', request_hash=digest([r['custom_id'] for r in requests]), status='error', error=type(exc).__name__, finished_at=now(),
                              requests=len(requests), bytes=size, mode='batch')
                raise
            payload = batch.to_dict() if hasattr(batch, 'to_dict') else dict(batch)
            with self.store.db:
                self.store.db.execute('INSERT INTO api_batches VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                                      (self.run.id, seq, rnd, payload.get('id'), len(requests), size, 'submitted', now(), None, None, None, None))
                self.store.db.executemany('INSERT INTO api_requests (custom_id, job_id, batch_seq, input_key, attempt, retry_of, request_sha256, feedback_json) '
                                          'VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(custom_id) DO UPDATE SET batch_seq = excluded.batch_seq, request_sha256 = excluded.request_sha256',
                                          [(custom_id(q[1]), self.run.id, seq, q[0]['input_key'], q[2], q[3], digest(q[5]), json.dumps(q[4]) if q[4] else None) for q in chunk])
            self.run.manifest['model_api_requests'] = self.run.manifest.get('model_api_requests', 0) + len(requests)
            self._attempt(step='batch_create', request_hash=digest([r['custom_id'] for r in requests]), status='submitted', finished_at=now(), batch_id=payload.get('id'),
                          processing_status=payload.get('processing_status'), requests=len(requests), bytes=size, mode='batch', round=rnd, batch_seq=seq)
            out.append(dict(seq=seq, batch_id=payload.get('id'), requests=len(requests), bytes=size, round=rnd))
            chunk, size = [], 0

        for q in self._questions(packet, limit):
            nbytes = len(json.dumps(q[5], ensure_ascii=False).encode('utf-8')) + 64
            if chunk and (size + nbytes > max_bytes or len(chunk) >= max_requests):
                send()
                if len(out) >= max_batches:
                    break
            chunk.append(q); size += nbytes
        if chunk and len(out) < max_batches:
            send()
        if out:
            self.run.finish('batch_submitted')
        return dict(job_id=self.run.id, submitted=out, questions=self.open_counts(), spent_usd=self.spent())

    def collect(self, client=None, wait_seconds=0, poll_interval=60):
        """Fetch every submitted batch that has ended: answers go into the store in chunks, only they are validated,
        and rejected papers become open retry questions for the next `submit` (or are sent at once with auto_retry)."""
        spec, packet = self._api_ready()
        client = self._client(client)
        db, results, deadline = self.store.db, [], time.monotonic() + wait_seconds
        for b in db.execute("SELECT * FROM api_batches WHERE job_id = ? AND status = 'submitted' ORDER BY seq", (self.run.id,)).fetchall():
            while True:
                status = client.messages.batches.retrieve(b['batch_id'])
                state = status.processing_status if hasattr(status, 'processing_status') else status['processing_status']
                if state == 'ended' or time.monotonic() >= deadline:
                    break
                time.sleep(poll_interval)
            if state != 'ended':
                results.append(dict(seq=b['seq'], batch_id=b['batch_id'], processing_status=state, collected=0)); continue
            records, feedback, failed, new, cached, cost = [], {}, [], 0, 0, 0.0

            def flush():
                nonlocal records, feedback, new, cached
                if records:
                    res = self._accept(records, source=dict(kind='model_api_batch', batch_id=b['batch_id'], batch_seq=b['seq']), retry=True, sent_feedback=feedback, ledger=False)
                    new += res['new']; cached += res['cached']
                records, feedback = [], {}

            for item in client.messages.batches.results(b['batch_id']):
                cid = item.custom_id if hasattr(item, 'custom_id') else item['custom_id']
                res = item.result if hasattr(item, 'result') else item['result']
                rtype = res.type if hasattr(res, 'type') else res['type']
                req = db.execute('SELECT * FROM api_requests WHERE custom_id = ? AND job_id = ?', (cid, self.run.id)).fetchone()
                if req is None:
                    failed.append(dict(custom_id=cid, error='unknown_custom_id')); continue
                if rtype != 'succeeded':                                          # no answer is stored: the question stays open, its failure is counted
                    with db:
                        db.execute('UPDATE api_requests SET failures = failures + 1 WHERE custom_id = ?', (cid,))
                    failed.append(dict(custom_id=cid, error=rtype)); continue
                msg = res.message if hasattr(res, 'message') else res['message']
                payload = msg.to_dict() if hasattr(msg, 'to_dict') else dict(msg)
                paper = self.store.get_input(req['input_key'])
                rec = self._record(paper, payload, payload.get('id'), 'batch', b['batch_id'], req['attempt'], req['retry_of'])
                cost += rec['model_provenance']['cost_usd'] or 0.0
                if req['attempt'] > 1:
                    feedback[paper['paper_id']] = json.loads(req['feedback_json'])
                records.append(rec)
                if len(records) >= COLLECT_CHUNK:
                    flush()
            flush()
            with db:
                db.execute("UPDATE api_batches SET status = 'collected', ended_at = ?, succeeded = ?, failed_json = ?, cost_usd = ? WHERE job_id = ? AND seq = ?",
                           (now(), new + cached, json.dumps(failed), round(cost, 6), self.run.id, b['seq']))
            self._attempt(step='batch_collect', request_hash=digest(b['batch_id']), status='ok', finished_at=now(), batch_id=b['batch_id'], batch_seq=b['seq'],
                          answers_new=new, answers_cached=cached, failed=len(failed), cost_usd=round(cost, 6), mode='batch')
            results.append(dict(seq=b['seq'], batch_id=b['batch_id'], processing_status='ended', collected=new + cached, new=new, cached=cached, failed=failed, cost_usd=round(cost, 6)))
        did = self.derivation()
        stats = self.deriver.stats(did, self.run.id)
        waiting = db.execute("SELECT count(*) FROM api_batches WHERE job_id = ? AND status = 'submitted'", (self.run.id,)).fetchone()[0]
        questions = self.open_counts()
        open_questions = questions['open']
        if waiting or open_questions:
            self.run.manifest.update(candidate_stats=stats, processed_at=now())
            self.run.finish('batch_submitted' if waiting else 'extraction_in_progress')
        else:
            self._finish(stats)
        return dict(job_id=self.run.id, batches=results, batches_waiting=waiting, questions=questions, stats=stats, spent_usd=self.spent(),
                    note='open questions (unanswered papers, validator-feedback retries) are sent by the next `job submit`' if open_questions else None)

    def run_sync(self, client=None, limit=None):
        """One request per paper, retry in the same conversation (prompt cache hit). For smoke tests and small jobs;
        the whole invocation is ONE ledger entry, per-request accounting is in each answer's provenance."""
        spec, packet = self._api_ready()
        client = self._client(client)
        max_attempts = int(packet['registry'].get('max_attempts', 1))
        sent = new = 0
        cost, error = 0.0, None
        try:
            for row in self.pending_rows(limit):
                paper = self.store.get_input(row['input_key'])
                attempt, retry, retry_of, feedback = 1, None, None, None
                while True:
                    self._cap()
                    request = build_request(packet, paper, retry)
                    rkey = request_key(row['input_key'], self.spec_key, 'api', attempt, retry_of)
                    response = client.messages.create(**request)
                    sent += 1
                    payload = response.to_dict() if hasattr(response, 'to_dict') else dict(response)
                    with self.store.db:
                        self.store.db.execute('INSERT OR REPLACE INTO api_requests (custom_id, job_id, batch_seq, input_key, attempt, retry_of, request_sha256, feedback_json) VALUES (?,?,?,?,?,?,?,?)',
                                              (custom_id(rkey), self.run.id, None, row['input_key'], attempt, retry_of, digest(request), json.dumps(feedback) if feedback else None))
                    rec = self._record(paper, payload, getattr(response, '_request_id', None) or payload.get('id'), 'sync', None, attempt, retry_of)
                    cost += rec['model_provenance']['cost_usd'] or 0.0
                    res = self._accept([rec], source=dict(kind='model_api_sync'), retry=True, sent_feedback={paper['paper_id']: feedback} if feedback else None, ledger=False)
                    new += res['new']
                    if attempt >= max_attempts:
                        break
                    feedback = self.deriver.feedback_for(self.derivation(write=False), rkey)
                    if not feedback:
                        break
                    retry, retry_of, attempt = ModelCaller.retry_turn(None, payload, feedback), rkey, attempt + 1
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.run.manifest['model_api_requests'] = self.run.manifest.get('model_api_requests', 0) + sent
            self._attempt(step='sync_session', request_hash=digest([self.run.id, 'sync', now()]), status='error' if error else 'ok', finished_at=now(), requests=sent,
                          answers_new=new, cost_usd=round(cost, 6), mode='sync', **({'error': error} if error else {}))
        stats = self.deriver.stats(self.derivation(), self.run.id)
        self._finish(stats)
        return dict(job_id=self.run.id, requests=sent, answers_new=new, cost_usd=round(cost, 6), spent_usd=self.spent(), stats=stats)

    # ---- read-only views ----

    def status(self):
        m, did = self.run.manifest, self.derivation(write=False)
        return dict(job_id=self.run.id, status=m['status'], spec_key=self.spec_key, channel=self.channel, papers=m['n_papers'], pending=len(self.pending_rows()),
                    derivation_id=did, derivation_exists=self.deriver.exists(did), stats=self.deriver.stats(did, self.run.id) if self.deriver.exists(did) else None,
                    questions=self.open_counts(), ingests=len(m.get('ingests', [])), model_api_requests=m.get('model_api_requests', 0))

    def report(self, did=None):
        did = did or self.derivation(write=False)
        report = self.deriver.report(did, self.run.id)
        validator = set()
        for row in self.store.db.execute('SELECT a.paper_id, a.meta_json FROM answers a JOIN job_papers j ON j.input_key=a.input_key '
                                         'WHERE j.job_id=? AND a.spec_key=? AND a.channel=? AND a.replicate=0 AND a.attempt>1',
                                         (self.run.id, self.spec_key, self.channel)):
            meta = json.loads(row['meta_json'])
            if meta.get('validator_feedback'):
                validator.add(row['paper_id'])
        report['attempts'].update(validator_retry_papers=len(validator))
        return dict(report, provenance=self.store.provenance_summary(self.spec_key, self.run.id),
                    validation_policy=self.run.manifest.get('validation_policy') or {})

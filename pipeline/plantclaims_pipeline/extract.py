"""Abstract-level extraction: the packet given to the model (title and abstract blocks, cleaned text), the request settings,
the output schema, and the five-gate validator that admits a claim only when its quotes, fields and evidence check out."""
import difflib
import inspect
import json
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

from .records import digest, now
from .relevance import write_once
from . import store as _store
from . import textclean
from .textclean import clean_indexed, compress, expand, raw_span

SCHEMA = '''
CREATE TABLE IF NOT EXISTS extraction_papers (
  run_id TEXT NOT NULL REFERENCES runs(run_id), paper_id TEXT NOT NULL, text_sha256 TEXT NOT NULL,
  status TEXT NOT NULL, label TEXT, reason_code TEXT, plant_subject TEXT, buckets_json TEXT,
  claims_proposed INTEGER NOT NULL, claims_accepted INTEGER NOT NULL, claims_review INTEGER NOT NULL,
  claims_quarantined INTEGER NOT NULL, profile_status TEXT, method TEXT NOT NULL, mode TEXT, model_id TEXT,
  request_id TEXT, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, stop_reason TEXT, raw_sha256 TEXT,
  PRIMARY KEY (run_id, paper_id));
CREATE TABLE IF NOT EXISTS extraction_claims (
  revision_id TEXT PRIMARY KEY, candidate_id TEXT NOT NULL, run_id TEXT NOT NULL REFERENCES runs(run_id),
  paper_id TEXT NOT NULL, kind TEXT NOT NULL, epistemic_status TEXT NOT NULL, origin TEXT NOT NULL,
  polarity TEXT NOT NULL, statistical_conclusion TEXT NOT NULL, buckets_json TEXT NOT NULL,
  quote_match TEXT NOT NULL, data TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS extraction_profiles (
  run_id TEXT NOT NULL REFERENCES runs(run_id), paper_id TEXT NOT NULL, revision_id TEXT NOT NULL,
  study_type TEXT, data TEXT NOT NULL, PRIMARY KEY (run_id, paper_id));
CREATE TABLE IF NOT EXISTS extraction_derivations (
  derivation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), created_at TEXT NOT NULL,
  validator_version TEXT NOT NULL, validator_fingerprint TEXT NOT NULL, policy_json TEXT NOT NULL,
  claim_ids_sha256 TEXT NOT NULL, stats_json TEXT NOT NULL, path TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS extraction_derived_claims (
  derivation_id TEXT NOT NULL REFERENCES extraction_derivations(derivation_id), revision_id TEXT NOT NULL, paper_id TEXT NOT NULL,
  disposition TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (derivation_id, revision_id));
CREATE TABLE IF NOT EXISTS extraction_quarantine (
  run_id TEXT NOT NULL REFERENCES runs(run_id), paper_id TEXT NOT NULL, item TEXT NOT NULL,
  errors_json TEXT NOT NULL, data TEXT NOT NULL, PRIMARY KEY (run_id, paper_id, item));
'''
METHOD_API = 'model_api'

BLOCK_ID = 'abstract'
TITLE_BLOCK_ID = 'title'
ACCEPTED_MATCH = ('exact_raw', 'exact_normalized')
NORMALIZE_MAP = {'\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"', '\u2010': '-', '\u2011': '-', '\u2012': '-',
                 '\u2013': '-', '\u2014': '-', '\u2015': '-', '\u2212': '-', '\u00ad': '', '\u00a0': ' ', '\u2009': ' ', '\u202f': ' '}


# ---- quote location ---------------------------------------------------------------------------------------
def normalize_indexed(text):
    """Return (normalized string, list mapping each normalized char to the index of its source char).
    NFKC, curly quotes/dashes straightened, case-folded letters, all whitespace removed."""
    out, index = [], []
    for i, ch in enumerate(text):
        ch = NORMALIZE_MAP.get(ch, ch)
        if not ch:
            continue  # soft hyphen and similar invisible characters
        for part in unicodedata.normalize('NFKC', ch):
            if part.isspace():
                continue  # whitespace-insensitive: "12 %" matches "12%", line breaks vanish
            for c in (part.casefold() if part.isalpha() else part):
                out.append(c)   # case folding can expand one letter ('İ' -> 'i̇', 'ß' -> 'ss'): one index per output char
                index.append(i)
    return ''.join(out), index


def locate_quote(text, quote):
    """Exact (unique) → normalized (unique) → fuzzy candidate (≥0.9 coverage) → not found; ambiguous when repeated."""
    if not isinstance(quote, str) or not quote.strip():
        return dict(start=None, end=None, status='not_found')
    n = text.count(quote)
    if n == 1:
        start = text.index(quote)
        return dict(start=start, end=start + len(quote), status='exact_raw')
    if n > 1:
        return dict(start=None, end=None, status='ambiguous')
    ntext, index = normalize_indexed(text)
    nquote, _ = normalize_indexed(quote.strip())
    n = ntext.count(nquote) if nquote else 0
    if n == 1:
        s = ntext.index(nquote)
        return dict(start=index[s], end=index[s + len(nquote) - 1] + 1, status='exact_normalized')
    if n > 1:
        return dict(start=None, end=None, status='ambiguous')
    sm = difflib.SequenceMatcher(None, ntext, nquote, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    covered = sum(b.size for b in blocks)
    if nquote and blocks and covered / len(nquote) >= 0.9:
        s, e = blocks[0].a, blocks[-1].a + blocks[-1].size
        if (e - s) <= 1.3 * len(nquote):
            return dict(start=index[s], end=index[e - 1] + 1, status='fuzzy_candidate', coverage=round(covered / len(nquote), 3))
    return dict(start=None, end=None, status='not_found')


# ---- registry, prompt, schema ------------------------------------------------------------------------------
BUCKET_RULE_KEYS = ('structure', 'paper_level', 'paper_level_fallback', 'objective_areas')
BUCKET_RULES_SUPPORTED = dict(structure=('flat_multi_label',), paper_level=('union_of_accepted_claims', 'union_of_accepted_claims_excluding_cited_background'),
                              paper_level_fallback=('relevance_buckets_of_the_answer',))
PAPER_LEVEL_EXCLUDES = {'union_of_accepted_claims_excluding_cited_background': ('cited_background',)}   # paper_level rule value → claim origins that do not count for the paper


def registry_bucket_rules(areas, kinds):
    """The part of the scope's `bucket_rules` that the program executes, frozen into the registry; the validator
    reads nothing else. A rule value this engine does not implement fails here, when the registry is built, never
    silently at validation. A scope without bucket_rules adds nothing to the registry."""
    rules = areas.get('bucket_rules')
    if not rules:
        return {}
    for key, values in BUCKET_RULES_SUPPORTED.items():
        if rules.get(key) not in values:
            raise ValueError(f'Unsupported bucket rule {key}={rules.get(key)!r}')
    for bucket, rule in rules['objective_areas'].items():
        if bucket not in areas['buckets']:
            raise ValueError('Bucket rule for an unknown bucket: ' + bucket)
        if isinstance(rule, dict) and (set(rule) != {'set_by', 'when_claim_kind', 'otherwise'} or rule['set_by'] != 'program'
                                       or rule['when_claim_kind'] not in kinds or rule['otherwise'] != 'removed'):
            raise ValueError('Unsupported program-set bucket rule for ' + bucket)
    return dict(bucket_scope_version=areas['version'], bucket_rules={k: rules[k] for k in BUCKET_RULE_KEYS})


def program_set_buckets(reg):
    """{bucket id: claim kind} of the research areas the program sets from the claim kind (rule 3 of the scope: B8 ⇐
    method_resource_result). Empty for registries without bucket_rules: their bucket lists are stored as answered."""
    areas = (reg.get('bucket_rules') or {}).get('objective_areas', {})
    return {b: r['when_claim_kind'] for b, r in areas.items() if isinstance(r, dict) and r.get('set_by') == 'program'}


def build_registry(run):
    cfg, claims, rel, scope = run.config['extract'], run.config['claims'], run.config['relevance_v0_3'], run.config['scope']
    areas = run.config.get('bucket_scope') or scope     # research areas: extract.bucket_scope, else the scope file
    return dict(schema_version=claims['schema_version'], relevance_schema_version=rel['schema_version'],
                relevance_prompt_version=rel['prompt_version'], reason_codes=rel['reason_codes'], next_actions=rel['next_actions'],
                bucket_definitions=areas['buckets'], **registry_bucket_rules(areas, claims['kinds']), kinds=claims['kinds'], epistemic_status=claims['epistemic_status'],
                origin=claims['origin'], polarity=claims['polarity'], statistical_conclusion=claims['statistical_conclusion'],
                fields=claims['fields'], enum_fields=claims.get('enum_fields', {}), missing_codes=claims['missing_codes'], required_by_kind=claims['required_by_kind'],
                max_claims_per_abstract=claims['max_claims_per_abstract'], profile=cfg['profile'], profile_enabled=bool(cfg.get('profile_enabled', True)),
                gate=cfg['gate'], claim_keys=claims.get('claim_keys', ['id', 's', 'k', 'es', 'o', 'pol', 'stat', 'bk', 'ev', 'f', 'miss', 'rev']),
                direction_inferred_vocabulary=claims.get('direction_inferred_vocabulary', []),
                # v0.6 additions; absent keys keep the v0.5 behaviour (exact literal match, separate `miss`, no basis / input_issues)
                **{k: claims[k] for k in ('basis', 'basis_definitions', 'input_issues', 'field_literal_match', 'evidence_ids') if k in claims},
                max_attempts=int(cfg.get('max_attempts', 1)), schema_profile=cfg.get('schema_profile', 'per_kind'))


def system_prompt(run, registry):
    text = run.config['extract_prompt']
    text = text.replace('{relevance_prompt_v0_3}', run.config['relevance_prompt_v0_3'].rstrip('\n'))
    text = text.replace('{max_claims}', str(registry['max_claims_per_abstract']))
    if '{' + 'relevance_prompt_v0_3' + '}' in text or '{max_claims}' in text:
        raise ValueError('Prompt placeholders were not substituted')
    return text


def _nullable(schema):
    return dict(anyOf=[schema, dict(type='null')])      # type arrays (["string", "null"]) are not accepted by structured outputs


def _evidence_schema(reg=None):
    ids = dict(enum=list(reg['evidence_ids'])) if reg and reg.get('evidence_ids') else dict(type='string')
    return dict(type='array', items=dict(type='object', additionalProperties=False, required=['id', 'block_id', 'quote'],
                                         properties=dict(id=ids, block_id=dict(enum=[TITLE_BLOCK_ID, BLOCK_ID]) if reg and reg.get('evidence_ids') else dict(type='string'),
                                                         quote=dict(type='string'))))


def merged_fields(reg):
    """claims-v0.6 wire format: no separate `miss`; a field in f is {v, ev} or {miss: code}."""
    return 'miss' not in reg.get('claim_keys', ['miss'])     # registries before claims-v0.4 carry no claim_keys: legacy form


def relevance_keys(reg):
    keys = ['label', 'reason_code', 'rationale', 'plant_subject', 'research_question', 'buckets', 'missing_reason', 'next_action']
    return keys + (['input_issues'] if reg.get('input_issues') else [])


def output_schema(reg):
    """Structured-output JSON schema for one paper. Only constructs the API accepts are used (anyOf, enum, const, $ref,
    required, additionalProperties=false; no oneOf, no type arrays, no minLength/maxItems). What JSON Schema cannot
    express stays with the validator: a field may cite only evidence ids of its own claim, `any`-of required groups,
    quote location, literal support."""
    program = program_set_buckets(reg)
    buckets = [b for b in sorted(reg['bucket_definitions']) if b not in program]     # the model cannot output a program-set area (B8)
    enum_fields = reg.get('enum_fields', {})
    ev_ref = dict(type='array', items=(dict(enum=list(reg['evidence_ids'])) if reg.get('evidence_ids') else dict(type='string')))
    merged = merged_fields(reg)

    def present(f):
        return dict(type='object', additionalProperties=False, required=['v', 'ev'],
                    properties=dict(v=(dict(enum=enum_fields[f]) if f in enum_fields else dict(type='string')), ev=ev_ref))
    missing = dict(type='object', additionalProperties=False, required=['miss'], properties=dict(miss=dict(enum=sorted(reg['missing_codes']))))
    defs = {}
    if merged:
        defs = dict(present=present(None), missing=missing, **{'present_' + f: present(f) for f in enum_fields})
        field_props = {f: dict(anyOf=[{'$ref': '#/$defs/' + ('present_' + f if f in enum_fields else 'present')}, {'$ref': '#/$defs/missing'}]) for f in reg['fields']}
    else:
        field_props = {f: present(f) for f in reg['fields']}
    base = dict(id=dict(type='string'), s=dict(type='string'), k=dict(enum=reg['kinds']), es=dict(enum=reg['epistemic_status']),
                o=dict(enum=reg['origin']), pol=dict(enum=reg['polarity']), stat=dict(enum=reg['statistical_conclusion']),
                basis=dict(enum=reg.get('basis', [])), bk=dict(type='array', items=dict(enum=buckets)), ev=_evidence_schema(reg),
                f=dict(type='object', additionalProperties=False, properties=field_props),
                miss=dict(type='object', additionalProperties=False, properties={f: dict(enum=sorted(reg['missing_codes'])) for f in reg['fields']}),
                rev=_nullable(dict(type='string')), dir_inf=dict(type='boolean'))

    def claim_schema(kind=None):
        props = {k: base[k] for k in reg['claim_keys']}
        if kind:
            props = dict(props, k=dict(const=kind), f=dict(base['f'], required=list(reg['required_by_kind'].get(kind, {}).get('all', []))))
        return dict(type='object', additionalProperties=False, required=list(reg['claim_keys']), properties=props)
    if merged and reg.get('schema_profile', 'per_kind') == 'per_kind':
        claim = dict(anyOf=[claim_schema(k) for k in reg['kinds']])     # per kind: its `all` fields are required keys of f
    else:
        claim = claim_schema()
    item = dict(type='object', additionalProperties=False, required=['text', 'evidence_ids'],
                properties=dict(text=dict(type='string'), evidence_ids=dict(type='array', items=dict(type='string'))))
    prof = reg['profile']
    profile_props = dict(study_type=dict(enum=prof['study_type']), evidence=_evidence_schema())
    for name in prof['items']:
        profile_props[name] = dict(type='array', items=item) if name in prof['lists'] else item
    profile = dict(type='object', additionalProperties=False, required=sorted(profile_props), properties=profile_props)
    # User decision 2026-09-18 (B): the scope label is a gate and buckets are optional annotations; neither carries quotes.
    # One variant per label, so a reason code can only be chosen from the codes registered for that label.
    keys = relevance_keys(reg)

    def relevance(label):
        props = dict(label=dict(const=label), reason_code=dict(enum=list(reg['reason_codes'][label])), rationale=dict(type='string'),
                     plant_subject=(dict(type='string') if label == 'core' else _nullable(dict(type='string'))),
                     research_question=(dict(type='string') if label == 'core' else _nullable(dict(type='string'))),
                     buckets=dict(type='array', items=dict(enum=buckets)),
                     missing_reason=(dict(type='string') if label == 'insufficient_evidence' else dict(type='null')),
                     next_action=dict(enum=reg['next_actions']), input_issues=dict(type='array', items=dict(enum=list(reg.get('input_issues') or []))))
        return dict(type='object', additionalProperties=False, required=keys, properties={k: props[k] for k in keys})
    profile_slot = dict(anyOf=[profile, dict(type='null')]) if reg.get('profile_enabled', True) else dict(type='null')
    # The model returns an assessed record only; execution_failed records are written by the program (refusal,
    # truncation, unparseable output), so the root stays a plain object (no root-level union).
    schema = dict(type='object', additionalProperties=False, required=['paper_id', 'text_sha256', 'status', 'relevance', 'claims', 'profile'],
                  properties=dict(paper_id=dict(type='string'), text_sha256=dict(type='string'), status=dict(const='assessed'),
                                  relevance=dict(anyOf=[relevance(l) for l in reg['reason_codes']]), claims=dict(type='array', items=claim), profile=profile_slot))
    if defs:
        schema['$defs'] = defs
    return schema


EFFORT_LEVELS = ('low', 'medium', 'high', 'xhigh', 'max')


def request_settings(cfg):
    """(thinking, effort) exactly as sent, or ValueError for a combination the model rejects (request-v0.8.0, 2026-09-30).
    Rules from the claude-api skill (model and thinking tables, cached 2026-09-25):
      - Claude Haiku 4.5 (`claude-haiku-4-5*`) has no adaptive thinking and rejects `output_config.effort`: thinking is either off
        (config `thinking: none`) or a fixed budget (`thinking: {type: enabled, budget_tokens: N}`, 1024 <= N < max_tokens);
      - every other current Claude model (Opus 5.5 / 5, Sonnet 5.5 / 5, Fable 5.1) takes adaptive thinking only (a budget is a 400)
        and an effort level. A config without `thinking` means adaptive, so every earlier config builds the same request as before."""
    model = cfg.get('model') or ''
    thinking, effort = cfg.get('thinking') or 'adaptive', cfg.get('effort')
    if model.startswith('claude-haiku-4-5'):
        if effort is not None:
            raise ValueError(f'{model} rejects output_config.effort: set effort to none')
        if thinking == 'none':
            return None, None
        if isinstance(thinking, dict) and thinking.get('type') == 'enabled' and set(thinking) == {'type', 'budget_tokens'} \
                and isinstance(thinking['budget_tokens'], int) and 1024 <= thinking['budget_tokens'] < cfg['max_tokens']:
            return dict(type='enabled', budget_tokens=thinking['budget_tokens']), None
        raise ValueError(f'{model} takes thinking none or {{type: enabled, budget_tokens: N}} with 1024 <= N < max_tokens ({cfg["max_tokens"]}); got {thinking!r}')
    if thinking != 'adaptive':
        raise ValueError(f'{model} takes adaptive thinking only (a thinking budget is rejected); got {thinking!r}')
    if effort not in EFFORT_LEVELS:
        raise ValueError(f'{model} needs an effort level {EFFORT_LEVELS}; got {effort!r}')
    return dict(type='adaptive'), effort


def build_request(packet, paper, retry=None):
    cfg = packet['request']
    # The breakpoint sits on the LAST system block so that prompt + registry are cached together (the cache covers the
    # rendered prefix up to the marker; a marker on the first block would leave the registry uncached on every request).
    system = [dict(type='text', text=packet['system_prompt']),
              dict(type='text', text=json.dumps(packet['registry'], ensure_ascii=False, sort_keys=True), cache_control=dict(type='ephemeral', ttl=cfg['cache_ttl']))]
    visible = {k: paper[k] for k in packet['input_fields']}
    if 'blocks' in visible:   # the model sees block_id + text only, never raw_text or the offset map
        visible['blocks'] = [dict(block_id=b['block_id'], text=b['text']) for b in paper['blocks']]
    user = json.dumps(visible, ensure_ascii=False)
    messages = [dict(role='user', content=user)]
    if retry:   # validator-feedback retry: the previous answer goes back unchanged, then the error list; system and schema stay identical (cache hit)
        messages += [dict(role='assistant', content=retry['previous_content']), dict(role='user', content=retry['feedback_text'])]
    thinking, effort = request_settings(cfg)
    out = dict(model=cfg['model'], max_tokens=cfg['max_tokens'], system=system)
    if thinking is not None:                    # Haiku 4.5 with thinking off sends no thinking parameter at all
        out['thinking'] = thinking
    out['output_config'] = dict(**({'effort': effort} if effort is not None else {}), format=dict(type='json_schema', schema=packet['output_schema']))
    out['messages'] = messages
    return out


def price(prices, model, usage):
    p = prices.get((model or '').split('@')[0])
    if p is None:
        return None
    cost = (usage.get('input_tokens', 0) * p['input'] + usage.get('output_tokens', 0) * p['output']
            + (usage.get('cache_read_input_tokens') or 0) * p['cache_read'] + (usage.get('cache_creation_input_tokens') or 0) * p['cache_write'])
    return round(cost / 1e6, 6)


# ---- sample selection ---------------------------------------------------------------------------------------
def sample_papers(root, sample_run, cleaning=None):
    """Packet papers from the frozen sample. With `cleaning` (extract-v0.6+) every block carries the cleaned text the
    model will see, the untouched raw text and the run-length map between them; text_sha256 covers the cleaned text
    (that is what the model reads and echoes) and raw_text_sha256 the source."""
    path = Path(root) / 'data' / 'runs' / sample_run / 'sample.jsonl'
    if not path.exists():
        raise ValueError('Sample run has no sample.jsonl')
    out = []
    for line in path.read_text(encoding='utf-8').splitlines():
        w = json.loads(line)
        p = w['pilot']
        text = p['abstract_text']
        title = w.get('title') or ''
        # Decision 2026-09-18 (issue 1): the title is a quotable block, because the organism often appears only there.
        blocks = [dict(block_id=TITLE_BLOCK_ID, text=title), dict(block_id=BLOCK_ID, text=text)]
        extra = {}
        if cleaning:
            report = {}
            for b in blocks:
                clean, starts, ends, stats = clean_indexed(b['text'], tags=cleaning.get('tags') or (), passes=cleaning.get('passes', 2))
                b.update(raw_text=b['text'], text=clean, clean_to_raw=compress(starts, ends))
                report[b['block_id']] = stats
            extra = dict(raw_text_sha256=digest('\n'.join(b['raw_text'] for b in blocks).encode('utf-8')), cleaning=dict(version=cleaning.get('version'), blocks=report))
        out.append(dict(paper_id=w['id'], index_sha256=p['index_sha256'], text_sha256=digest('\n'.join(b['text'] for b in blocks).encode('utf-8')),
                        cell_id=p['cell_id'], primary_group=p['primary_group'], era=p['era'], rank_digest=p['rank_digest'],
                        title=title, publication_year=w.get('publication_year'), blocks=blocks, **extra))
    return out


def select_papers(papers, limit=None, cells=None, paper_ids=None):
    """Deterministic: within-cell order by rank_digest, round-robin across cells until the limit.
    `paper_ids` restricts the packet to an explicit id list (full OpenAlex ids or bare W-ids); unknown ids are an error."""
    if paper_ids:
        wanted = {str(i).rsplit('/', 1)[-1] for i in paper_ids}
        have = {p['paper_id'].rsplit('/', 1)[-1] for p in papers}
        if wanted - have:
            raise ValueError('paper ids not in the sample: ' + ', '.join(sorted(wanted - have)[:5]))
        papers = [p for p in papers if p['paper_id'].rsplit('/', 1)[-1] in wanted]
    if cells:
        papers = [p for p in papers if p['cell_id'] in set(cells)]
    if not limit or limit >= len(papers):
        return sorted(papers, key=lambda p: (p['cell_id'], p['rank_digest']))
    by_cell = defaultdict(list)
    for p in sorted(papers, key=lambda p: (p['cell_id'], p['rank_digest'])):
        by_cell[p['cell_id']].append(p)
    chosen, keys = [], sorted(by_cell)
    while len(chosen) < limit:
        progressed = False
        for k in keys:
            if by_cell[k] and len(chosen) < limit:
                chosen.append(by_cell[k].pop(0)); progressed = True
        if not progressed:
            break
    return sorted(chosen, key=lambda p: (p['cell_id'], p['rank_digest']))


# ---- validation ----------------------------------------------------------------------------------------------
def _locate_all(items, blocks, text_sha256, maps=None, allowed_ids=None):
    """Locate an evidence list; returns (located list, errors). Ids must be unique and blocks must exist.
    `maps` (block_id → (starts, ends)) adds raw_start/raw_end for packets whose block text was cleaned."""
    located, errors, seen = [], [], set()
    for e in items:
        if not isinstance(e, dict) or set(e) != {'id', 'block_id', 'quote'} or not all(isinstance(e[k], str) and e[k] for k in e):
            errors.append('invalid_evidence_schema'); continue
        if e['id'] in seen:
            errors.append('duplicate_evidence_id:' + e['id']); continue
        seen.add(e['id'])
        if allowed_ids and e['id'] not in allowed_ids:
            errors.append('evidence_id_not_in_vocabulary:' + e['id']); continue
        block = blocks.get(e['block_id'])
        if block is None:
            errors.append('block_outside_document:' + e['id']); continue
        loc = locate_quote(block, e['quote'])
        raw = {}
        if maps and maps.get(e['block_id']):
            rs, re_ = raw_span(*maps[e['block_id']], loc['start'], loc['end'])
            raw = dict(raw_start=rs, raw_end=re_)
        located.append(dict(e, start=loc['start'], end=loc['end'], offset_unit='unicode_codepoints', quote_match=loc['status'],
                            text_sha256=text_sha256, **raw, **({'coverage': loc['coverage']} if 'coverage' in loc else {})))
    return located, errors


def literal_supported(value, quote, mode):
    """'exact' | 'normalized' | None. Normalized = the quote-location normalization on both sides (NFKC, case-folded,
    quotes/dashes straightened, whitespace removed); used from claims-v0.6 on (the v0.5 run lost four claims to
    "peach" vs "Peach")."""
    if value in quote:
        return 'exact'
    if mode == 'normalized':
        nv = normalize_indexed(value)[0]
        if nv and nv in normalize_indexed(quote)[0]:
            return 'normalized'
    return None


def validate_claim(claim, blocks, text_sha256, reg, maps=None):
    errors = []
    required = list(reg.get('claim_keys', ['id', 's', 'k', 'es', 'o', 'pol', 'stat', 'bk', 'ev', 'f', 'miss', 'rev']))
    if not isinstance(claim, dict) or set(claim) != set(required):
        return None, ['claim_schema_keys'], 'quarantined'
    dir_inf = bool(claim.get('dir_inf', False)) if 'dir_inf' in required else False
    if 'dir_inf' in required and not isinstance(claim.get('dir_inf'), bool):
        errors.append('invalid_dir_inf')
    for key, enum in (('k', 'kinds'), ('es', 'epistemic_status'), ('o', 'origin'), ('pol', 'polarity'), ('stat', 'statistical_conclusion')):
        if claim[key] not in reg[enum]:
            errors.append('invalid_' + key)
    if not isinstance(claim['s'], str) or not claim['s'].strip():
        errors.append('empty_statement')
    if not isinstance(claim['bk'], list) or any(b not in reg['bucket_definitions'] for b in claim['bk']) or len(set(claim['bk'])) != len(claim['bk']):
        errors.append('invalid_buckets')
    if 'basis' in required and claim['basis'] not in reg.get('basis', []):
        errors.append('invalid_basis')
    located, ev_errors = _locate_all(claim['ev'] if isinstance(claim['ev'], list) else [], blocks, text_sha256, maps, reg.get('evidence_ids'))
    errors += ev_errors
    if not located:
        errors.append('no_evidence')
    quotes = {e['id']: e['quote'] for e in located}
    if merged_fields(reg):   # claims-v0.6 wire format: a field is {v, ev} or {miss: code}; split it back into the two stored maps
        if not isinstance(claim['f'], dict):
            return None, errors + ['invalid_fields_schema'], 'quarantined'
        f = {n: s for n, s in claim['f'].items() if not (isinstance(s, dict) and set(s) == {'miss'})}
        miss = {n: s['miss'] for n, s in claim['f'].items() if isinstance(s, dict) and set(s) == {'miss'}}
    else:
        f, miss = claim['f'], claim['miss']
    if not isinstance(f, dict) or not isinstance(miss, dict):
        return None, errors + ['invalid_fields_schema'], 'quarantined'
    mode, literal_match = reg.get('field_literal_match', 'exact'), {}
    for name, spec in f.items():
        if name not in reg['fields']:
            errors.append('unknown_field:' + name); continue
        if not isinstance(spec, dict) or set(spec) != {'v', 'ev'} or not isinstance(spec['v'], str) or not spec['v'].strip() or not isinstance(spec['ev'], list) or not spec['ev']:
            errors.append('invalid_present_field:' + name); continue
        if any(ref not in quotes for ref in spec['ev']):
            errors.append('field_evidence_unknown:' + name); continue
        if name == 'direction' and dir_inf:
            if spec['v'] not in reg.get('direction_inferred_vocabulary', []):
                errors.append('inferred_direction_not_in_vocabulary')
            continue  # inferred direction is exempt from the literal-substring rule (user decision 2026-09-18)
        if name in reg.get('enum_fields', {}):
            if spec['v'] not in reg['enum_fields'][name]:
                errors.append('enum_field_value_not_in_vocabulary:' + name)
            continue  # controlled label (e.g. interactor_role): enum check instead of the substring rule
        support = [m for m in (literal_supported(spec['v'], quotes[ref], mode) for ref in spec['ev']) if m]
        if not support:
            errors.append('field_literal_not_in_cited_evidence:' + name)
        else:
            literal_match[name] = 'exact' if 'exact' in support else 'normalized'
    for name, code in miss.items():
        if name not in reg['fields']:
            errors.append('unknown_missing_field:' + name)
        elif code not in reg['missing_codes']:
            errors.append('invalid_missing_code:' + name)
        elif name in f:
            errors.append('field_both_present_and_missing:' + name)
    rule = reg['required_by_kind'].get(claim['k'], {}) if claim['k'] in reg['kinds'] else {}
    stated = set(f) | set(miss)
    for name in rule.get('all', []):
        if name not in stated:
            errors.append(f"required_field_missing:{claim['k']}:{name}")
    if rule.get('any') and not any(name in stated for name in rule['any']):
        errors.append(f"required_any_missing:{claim['k']}:{'|'.join(rule['any'])}")
    if claim['rev'] is not None and not isinstance(claim['rev'], str):
        errors.append('invalid_rev')
    if dir_inf and 'direction' not in f:
        errors.append('dir_inf_without_direction')
    statuses = {e['quote_match'] for e in located}
    disposition = 'accepted'
    if errors:
        disposition = 'quarantined'
    elif statuses - set(ACCEPTED_MATCH):
        if statuses <= set(ACCEPTED_MATCH) | {'fuzzy_candidate'}:
            disposition = 'review'
        else:
            errors.append('quote_not_located'); disposition = 'quarantined'
    program = program_set_buckets(reg)
    answered = list(claim['bk']) if isinstance(claim['bk'], list) else []
    # Rule 3 of the scope: a program-set area (B8) never comes from the model — it is dropped from the answer and added when the
    # claim kind matches. Areas are flat (no primary area), so the stored list is in registry order and the order the model
    # wrote survives only in buckets_model. Registries without bucket_rules keep the answered list as it is.
    earned = {b for b in answered if b not in program} | {b for b, kind in program.items() if kind == claim['k']}
    areas = [b for b in sorted(reg['bucket_definitions']) if b in earned] if program else answered
    out = dict(local_id=claim['id'], statement=claim['s'], kind=claim['k'], epistemic_status=claim['es'], origin=claim['o'],
               polarity=claim['pol'], statistical_conclusion=claim['stat'], buckets=areas, **({'buckets_model': answered} if program else {}), evidence=located,
               fields={n: dict(value=s['v'], evidence_ids=list(s['ev']), **({'literal_match': literal_match[n]} if mode == 'normalized' and n in literal_match else {}))
                       for n, s in f.items() if n in reg['fields'] and isinstance(s, dict) and set(s) == {'v', 'ev'}},
               missing={n: reg['missing_codes'][c] for n, c in miss.items() if c in reg['missing_codes']},
               **({'basis': claim['basis']} if 'basis' in required else {}),
               needs_schema_review=claim['rev'], direction_inferred=dir_inf,
               quote_match=('exact' if statuses <= set(ACCEPTED_MATCH) else 'fuzzy_candidate'))
    return out, errors, disposition


def validate_profile(profile, blocks, text_sha256, reg):
    prof = reg['profile']
    errors = []
    expected = set(prof['items']) | {'study_type', 'evidence'}
    if not isinstance(profile, dict) or set(profile) != expected:
        return None, ['profile_schema_keys']
    if profile['study_type'] not in prof['study_type']:
        errors.append('invalid_study_type')
    located, ev_errors = _locate_all(profile['evidence'] if isinstance(profile['evidence'], list) else [], blocks, text_sha256)
    errors += ev_errors
    ok_ids = {e['id'] for e in located if e['quote_match'] in ACCEPTED_MATCH}
    out = dict(study_type=profile['study_type'], evidence=located)

    def check(name, item):
        if not isinstance(item, dict) or set(item) != {'text', 'evidence_ids'} or not isinstance(item['text'], str) or not item['text'].strip():
            errors.append('invalid_profile_item:' + name); return None
        if not isinstance(item['evidence_ids'], list) or not item['evidence_ids'] or any(i not in ok_ids for i in item['evidence_ids']):
            errors.append('profile_item_without_located_evidence:' + name); return None
        return dict(text=item['text'], evidence_ids=list(item['evidence_ids']))
    for name in prof['items']:
        value = profile[name]
        if name in prof['lists']:
            if not isinstance(value, list):
                errors.append('invalid_profile_item:' + name); continue
            out[name] = [c for c in (check(name, v) for v in value) if c]
        else:
            out[name] = check(name, value)
    return (out if not errors else None), errors


def contributing_claims(accepted, reg):
    """The accepted claims whose areas count for the paper under the registry's paper_level rule: every accepted claim
    (union_of_accepted_claims) or every one whose origin is not cited_background (union_of_accepted_claims_excluding_cited_background). A claim
    left out here keeps its own areas — only the paper level ignores it."""
    excluded = PAPER_LEVEL_EXCLUDES.get((reg.get('bucket_rules') or {}).get('paper_level'), ())
    return [c for c in accepted if c['origin'] not in excluded]


def paper_buckets(answered, accepted, reg):
    """Rule 2: a paper's areas are never judged separately — they are the union of the areas of its contributing accepted
    claims (contributing_claims). The answer's own list is the fallback for a paper without a contributing claim, minus
    program-set areas (only a claim kind earns those). Returns (areas in registry order, source); the source names the
    paper_level rule value that was applied, or the fallback."""
    contributing = contributing_claims(accepted, reg)
    if contributing:
        union = {b for c in contributing for b in c['buckets']}
        return [b for b in sorted(reg['bucket_definitions']) if b in union], reg['bucket_rules']['paper_level']
    program = program_set_buckets(reg)
    return [b for b in answered if b not in program], 'relevance_buckets_of_the_answer'


def validate_paper(record, paper, reg, run_id):
    """One model record for one packet paper → dict(status, relevance, claims{accepted,review,quarantined}, profile, errors)."""
    res = dict(paper_id=paper['paper_id'], status=None, errors=[], relevance=None, claims_accepted=[], claims_review=[],
               claims_quarantined=[], profile=None, profile_errors=[], claims_proposed=0)
    blocks = {b['block_id']: b['text'] for b in paper['blocks']}
    maps = {b['block_id']: expand(b['clean_to_raw'], len(b['text'])) for b in paper['blocks'] if 'clean_to_raw' in b}
    if not isinstance(record, dict) or record.get('paper_id') != paper['paper_id'] or record.get('text_sha256') != paper['text_sha256']:
        res['errors'].append('identity_mismatch'); res['status'] = 'quarantined'; return res
    if record.get('status') == 'execution_failed':
        if set(record) != {'paper_id', 'text_sha256', 'status', 'error'} or not isinstance(record['error'], str) or not record['error']:
            res['errors'].append('invalid_failed_record'); res['status'] = 'quarantined'
        else:
            res['status'] = 'execution_failed'; res['error'] = record['error']
        return res
    if record.get('status') != 'assessed' or set(record) != {'paper_id', 'text_sha256', 'status', 'relevance', 'claims', 'profile'}:
        res['errors'].append('record_schema_keys'); res['status'] = 'quarantined'; return res
    rel = record['relevance']
    keys = set(relevance_keys(reg))
    if not isinstance(rel, dict) or set(rel) != keys:
        res['errors'].append('relevance_schema_keys'); res['status'] = 'quarantined'; return res
    label = rel['label']
    if label not in reg['reason_codes'] or rel['reason_code'] not in reg['reason_codes'][label]:
        res['errors'].append('reason_code_not_in_label_registry')
    if rel['next_action'] not in reg['next_actions']:
        res['errors'].append('invalid_next_action')
    if not isinstance(rel['rationale'], str) or not rel['rationale'].strip():
        res['errors'].append('empty_rationale')
    if label == 'core' and not (isinstance(rel['plant_subject'], str) and rel['plant_subject'].strip() and isinstance(rel['research_question'], str) and rel['research_question'].strip()):
        res['errors'].append('core_requires_subject_and_question')
    if label == 'insufficient_evidence' and not (isinstance(rel['missing_reason'], str) and rel['missing_reason'].strip()):
        res['errors'].append('insufficient_requires_missing_reason')
    if label != 'insufficient_evidence' and rel['missing_reason'] not in (None, ''):
        res['errors'].append('missing_reason_only_for_insufficient')
    buckets = rel['buckets'] if isinstance(rel['buckets'], list) else None
    if buckets is None or any(b not in reg['bucket_definitions'] for b in buckets) or len(set(buckets)) != len(buckets):
        res['errors'].append('invalid_buckets')  # optional multi-label annotation: enum and uniqueness only, no evidence (decision B)
    issues = rel.get('input_issues') if 'input_issues' in keys else None
    if 'input_issues' in keys and (not isinstance(issues, list) or any(i not in reg['input_issues'] for i in issues) or len(set(issues)) != len(issues)):
        res['errors'].append('invalid_input_issues')
    claims, profile = record['claims'], record['profile']
    if not isinstance(claims, list):
        res['errors'].append('invalid_claims_list'); claims = []
    res['claims_proposed'] = len(claims)
    if label != 'core' and reg['gate'] == 'core_only':
        if claims:
            res['errors'].append('claims_on_non_core')
        if profile is not None:
            res['errors'].append('profile_on_non_core')
    if len(claims) > reg['max_claims_per_abstract']:
        res['errors'].append('claims_over_budget')
    if res['errors']:
        res['status'] = 'quarantined'; return res
    res['status'] = 'assessed'
    res['relevance'] = dict(label=label, reason_code=rel['reason_code'], rationale=rel['rationale'], plant_subject=rel['plant_subject'],
                            research_question=rel['research_question'], buckets=list(buckets),
                            missing_reason=rel['missing_reason'], next_action=rel['next_action'],
                            **({'input_issues': list(issues)} if 'input_issues' in keys else {}))

    def settled():
        if reg.get('bucket_rules'):     # registries without bucket_rules keep the answered list
            final, source = paper_buckets(list(buckets), res['claims_accepted'], reg)
            res['relevance'].update(buckets=final, buckets_model=list(buckets), buckets_source=source)
        # Opt-in validation policy: keep the original label and all claim diagnostics.
        # Older registries still allow a core abstract with no extractable result.
        if label == 'core' and len(res['claims_accepted']) < reg.get('min_core_claims', 0):
            res['status'] = 'quarantined'
            res['errors'].append('core_requires_accepted_claim')
        return res
    if label != 'core':
        return settled()
    seen_ids = set()
    for pos, claim in enumerate(claims):
        out, errors, disposition = validate_claim(claim, blocks, paper['text_sha256'], reg, maps)
        lid = claim.get('id') if isinstance(claim, dict) else None
        if lid in seen_ids:
            errors.append('duplicate_local_id'); disposition = 'quarantined'
        seen_ids.add(lid)
        if disposition == 'quarantined':
            res['claims_quarantined'].append(dict(response_position=pos, local_id=lid, errors=errors, raw=claim,
                                                  evidence_status={e['id']: e['quote_match'] for e in (out or {}).get('evidence', [])}))
            continue
        fp = digest([run_id, paper['paper_id'], paper['text_sha256'], claim])
        out.update(id='claim-' + fp[:16], revision_id='rev-' + fp, paper_id=paper['paper_id'], run_id=run_id,
                   text_sha256=paper['text_sha256'], schema_version=reg['schema_version'], publication_status='candidate_only',
                   validation=dict(schema='passed', quote_match=out['quote_match'], field_literal_support='passed',
                                   field_semantic_support='not_assessed', semantic_review='not_assessed'))
        (res['claims_accepted'] if disposition == 'accepted' else res['claims_review']).append(out)
    if not reg.get('profile_enabled', True):
        if profile is not None:
            res['profile_errors'].append('profile_disabled_at_this_level')
    elif profile is None:
        res['profile_errors'].append('profile_missing_for_core')
    else:
        prof, errors = validate_profile(profile, blocks, paper['text_sha256'], reg)
        if errors:
            res['profile_errors'] = errors
        else:
            fp = digest([run_id, paper['paper_id'], paper['text_sha256'], profile])
            prof.update(revision_id='profile-' + fp, paper_id=paper['paper_id'], run_id=run_id)
            res['profile'] = prof
    return settled()


# ---- validator feedback and attempts --------------------------------------------------------------------------------
RETRY_HINTS = {
    'quote_not_located': 'copy every quote character for character from the named block (title or abstract); do not add markup, entities or words',
    'quote_fuzzy_match': 'the quote nearly matches the block but not exactly; copy it again character for character',
    'ambiguous': 'the quote occurs more than once in its block; lengthen it until it is unique',
    'field_literal_not_in_cited_evidence': 'a field value must be a substring of a quote that the field cites; cite the quote that contains it or extend that quote',
    'field_evidence_unknown': "a field may cite only evidence ids that are listed in this same claim's ev",
    'evidence_id_not_in_vocabulary': 'use the evidence ids of the registry (e0..e7 for the abstract, t0/t1 for the title)',
    'required_field_missing': 'give the required field a value {v, ev} or an explicit {miss: nr|na|un}',
    'required_any_missing': 'state at least one field of the listed group (value or explicit miss code)',
    'reason_code_not_in_label_registry': 'choose a reason_code that the registry lists for the chosen label',
    'invalid_basis': 'basis must be one of the registry values',
    'invalid_input_issues': 'input_issues must be a list of registry values without repeats',
    'identity_mismatch': 'echo paper_id and text_sha256 exactly as supplied',
    'claims_on_non_core': 'claims must be an empty list unless label is core',
    'claims_over_budget': 'return at most the permitted number of claims',
}


def attempt_of(record):
    return int(record.get('attempt') or 1)


def record_sha(record):
    return digest(record)


def paper_feedback(v):
    """Validator result → the error list that is sent back for a retry ([] = nothing to fix). Codes, claim ids, field
    names and evidence statuses only; never the located source text."""
    if v['status'] == 'execution_failed':
        return []
    items = []
    if v['status'] == 'quarantined':
        items.append(dict(item='paper', errors=list(v['errors'])))
    for q in v['claims_quarantined']:
        bad = {i: s for i, s in (q.get('evidence_status') or {}).items() if s not in ACCEPTED_MATCH}
        items.append(dict(item='claim', id=q['local_id'], position=q['response_position'], errors=list(q['errors']), **({'evidence_status': bad} if bad else {})))
    for c in v['claims_review']:
        bad = {e['id']: e['quote_match'] for e in c['evidence'] if e['quote_match'] not in ACCEPTED_MATCH}
        items.append(dict(item='claim', id=c['local_id'], errors=['quote_fuzzy_match'], evidence_status=bad))
    return items


def feedback_text(items):
    codes = sorted({e.split(':')[0] for it in items for e in it['errors']} | {s for it in items for s in (it.get('evidence_status') or {}).values() if s == 'ambiguous'})
    hints = {c: RETRY_HINTS[c] for c in codes if c in RETRY_HINTS}
    return ('The program validated your previous answer and could not accept the items listed below. Return the COMPLETE corrected '
            'JSON object for this paper again (same paper_id and text_sha256, all claims, not only the corrected ones). Change only what the '
            'errors require; do not add claims to compensate.\n'
            + json.dumps(dict(validator_errors=items, hints=hints), ensure_ascii=False, indent=1))


# ---- model calls ------------------------------------------------------------------------------------------------
class ModelCaller:
    """Sends requests through a provided client (sync or Message Batches); never constructs credentials itself.
    Every raw response is written once; every operation is logged as a manifest attempt; the spend cap is
    checked before sending; refusals and truncation become execution_failed records, never fabricated output."""

    def __init__(self, run, packet, client):
        self.run, self.packet, self.client = run, packet, client
        self.cfg = run.config['extract']
        self.mode = self.cfg['mode']
        self.prices = self.cfg['prices'][self.mode]
        self.raw = run.raw / 'responses'
        self.raw.mkdir(parents=True, exist_ok=True)
        self.max_attempts = int(packet['registry'].get('max_attempts', 1))

    def spent(self):
        return round(sum((a.get('cost_usd') or 0.0) for a in self.run.manifest['attempts']), 6)

    def _attempt(self, **fields):
        entry = dict(seq=len(self.run.manifest['attempts']) + 1, started_at=now(), **fields)
        self.run.manifest['attempts'].append(entry)
        self.run.flush()
        return entry

    def _provenance(self, payload, entry, request_id):
        usage = payload.get('usage') or {}
        cost = price(self.prices, payload.get('model'), usage)
        return dict(provider='anthropic', model_id=payload.get('model'), request_id=request_id, effort=self.cfg['effort'],
                    stop_reason=payload.get('stop_reason'), input_tokens=usage.get('input_tokens'), output_tokens=usage.get('output_tokens'),
                    cost_usd=cost, price_checked_on=self.cfg['prices']['checked_on'],
                    accounting_status='api_reported' if cost is not None else 'api_reported_unpriced',
                    human_review='not_performed', raw_sha256=entry['raw_sha256'])

    def raw_path(self, paper, attempt=1):
        # attempt 1 keeps the historical name; a retry gets its own write-once file
        return self.raw / (digest(paper['paper_id'].encode())[:16] + ('' if attempt == 1 else f'.a{attempt}') + '.json')

    def _store(self, paper, payload, request_id, extra, attempt=1, request_hash=None, previous=None, feedback=None):
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        target = self.raw_path(paper, attempt)
        write_once(target, raw)
        entry = self._attempt(step=paper['paper_id'], request_hash=request_hash or digest(build_request(self.packet, paper)), status='ok', finished_at=now(),
                              model_id=payload.get('model'), request_id=request_id, stop_reason=payload.get('stop_reason'),
                              usage=payload.get('usage') or {}, raw_path=str(target.relative_to(self.run.root)), raw_sha256=digest(raw),
                              cost_usd=price(self.prices, payload.get('model'), payload.get('usage') or {}), attempt=attempt, **extra)
        chain = dict(attempt=attempt) if (attempt > 1 or self.max_attempts > 1) else {}
        if attempt > 1:
            chain.update(retry_of=record_sha(previous), validator_feedback=feedback)
        return self.result_record(paper, payload, self._provenance(payload, entry, request_id), dict(extra, **chain))

    def retry_turn(self, payload, feedback):
        content = payload.get('content') or [dict(type='text', text='')]
        return dict(previous_content=content, feedback_text=feedback_text(feedback))

    @staticmethod
    def result_record(paper, payload, provenance, extra):
        """Turn a raw message payload into the per-paper record the validator consumes."""
        stop = payload.get('stop_reason')
        if stop == 'refusal':
            result = dict(paper_id=paper['paper_id'], text_sha256=paper['text_sha256'], status='execution_failed', error='model_refusal')
        elif stop == 'max_tokens':
            result = dict(paper_id=paper['paper_id'], text_sha256=paper['text_sha256'], status='execution_failed', error='truncated_max_tokens')
        else:
            text = ''.join(b.get('text', '') for b in payload.get('content', []) if b.get('type') == 'text')
            try:
                result = json.loads(text)
            except ValueError:
                result = dict(paper_id=paper['paper_id'], text_sha256=paper['text_sha256'], status='execution_failed', error='invalid_json')
        return dict(paper_id=paper['paper_id'], method=METHOD_API, model_provenance=provenance, result=result, **extra)

    def run_sync(self, papers):
        """One request per paper; when the validator rejects part of the answer and attempts remain, the error list is
        sent back once in the same conversation (system prompt and schema unchanged, so the prompt cache still hits).
        Every attempt is a ledgered request with its own write-once raw file and its own record."""
        out = []
        for paper in papers:
            attempt, retry, previous, feedback = 1, None, None, None
            while True:
                if self.spent() >= self.cfg['spend_cap_usd']:
                    raise RuntimeError(f"spend cap {self.cfg['spend_cap_usd']} USD reached; not sending")
                request = build_request(self.packet, paper, retry)
                try:
                    response = self.client.messages.create(**request)
                except Exception as exc:  # ledger the failure, never retry silently
                    self._attempt(step=paper['paper_id'], request_hash=digest(request), status='error', error=type(exc).__name__, finished_at=now(), mode=self.mode, attempt=attempt)
                    raise
                payload = response.to_dict() if hasattr(response, 'to_dict') else dict(response)
                record = self._store(paper, payload, getattr(response, '_request_id', None) or payload.get('id'), dict(mode='sync', batch_id=None),
                                     attempt=attempt, request_hash=digest(request), previous=previous, feedback=feedback)
                out.append(record)
                if attempt >= self.max_attempts:
                    break
                feedback = paper_feedback(validate_paper(record['result'], paper, self.packet['registry'], self.run.id))
                if not feedback:
                    break
                retry, previous, attempt = self.retry_turn(payload, feedback), record, attempt + 1
        return out

    def submit_batch(self, papers, retries=None):
        """Round 1: every paper. Round 2 (`retries` = {paper_id: dict(previous=record, feedback=[...])}): only the papers
        whose first answer the validator rejected, each with its previous answer and the error list appended."""
        if self.spent() >= self.cfg['spend_cap_usd']:
            raise RuntimeError(f"spend cap {self.cfg['spend_cap_usd']} USD reached; not submitting")
        rnd = 2 if retries else 1
        requests = []
        for p in papers:
            retry = None
            if retries:
                prev_payload = json.loads(self.raw_path(p, 1).read_text(encoding='utf-8'))
                retry = self.retry_turn(prev_payload, retries[p['paper_id']]['feedback'])
            requests.append(dict(custom_id=p['paper_id'], params=build_request(self.packet, p, retry)))
        batch = self.client.messages.batches.create(requests=requests)
        payload = batch.to_dict() if hasattr(batch, 'to_dict') else dict(batch)
        self._attempt(step='batch_create', request_hash=digest([r['custom_id'] for r in requests]), status='submitted', finished_at=now(),
                      batch_id=payload.get('id'), processing_status=payload.get('processing_status'), requests=len(requests), mode='batch', round=rnd)
        if retries:
            self.run.manifest.setdefault('batch_rounds', []).append(self.run.manifest['batch'])
        self.run.manifest['batch'] = dict(batch_id=payload.get('id'), submitted_at=now(), requests=len(requests), paper_ids=[p['paper_id'] for p in papers], round=rnd,
                                          request_hashes={r['custom_id']: digest(r['params']) for r in requests})
        self.run.flush()
        return payload.get('id')

    def collect_batch(self, papers, wait_seconds=0, poll_interval=60, retries=None):
        batch_id = self.run.manifest['batch']['batch_id']
        rnd = self.run.manifest['batch'].get('round', 1)
        hashes = self.run.manifest['batch'].get('request_hashes', {})
        deadline = time.monotonic() + wait_seconds
        while True:
            status = self.client.messages.batches.retrieve(batch_id)
            state = status.processing_status if hasattr(status, 'processing_status') else status['processing_status']
            if state == 'ended':
                break
            if time.monotonic() >= deadline:
                return dict(batch_id=batch_id, processing_status=state, collected=0)
            time.sleep(poll_interval)
        by_id = {p['paper_id']: p for p in papers}
        out, errors = [], []
        for item in self.client.messages.batches.results(batch_id):
            custom_id = item.custom_id if hasattr(item, 'custom_id') else item['custom_id']
            res = item.result if hasattr(item, 'result') else item['result']
            rtype = res.type if hasattr(res, 'type') else res['type']
            paper = by_id.get(custom_id)
            if paper is None:
                errors.append(dict(custom_id=custom_id, error='unknown_custom_id')); continue
            if rtype != 'succeeded':
                self._attempt(step=custom_id, request_hash=digest(build_request(self.packet, paper)), status='error', error='batch_' + rtype, finished_at=now(), mode='batch', batch_id=batch_id)
                errors.append(dict(custom_id=custom_id, error=rtype)); continue
            msg = res.message if hasattr(res, 'message') else res['message']
            payload = msg.to_dict() if hasattr(msg, 'to_dict') else dict(msg)
            link = (retries or {}).get(custom_id, {})
            out.append(self._store(paper, payload, payload.get('id'), dict(mode='batch', batch_id=batch_id), attempt=rnd,
                                   request_hash=hashes.get(custom_id), previous=link.get('previous'), feedback=link.get('feedback')))
        self.run.manifest['batch'].update(ended_at=now(), succeeded=len(out), failed=errors)
        self.run.flush()
        return dict(batch_id=batch_id, processing_status='ended', collected=len(out), failed=errors, records=out)


COMPONENT_VERSIONS = dict(packet_builder='packet-v0.7.1', request_builder='request-v0.8.0', validator='validator-v0.9.1', store='store-v0.1.1')
VALIDATOR_POLICY_KEYS = ('field_literal_match', 'required_by_kind', 'bucket_rules', 'min_core_claims')   # registry keys that are validator policy, not wire format
LOCK_FILE = Path(__file__).with_name('components.lock.json')


def policy_from_yaml(loaded, kinds):
    """Validator-policy keys of a YAML file. A claims-vX.yaml gives field_literal_match / required_by_kind; a scope file
    gives bucket_rules, reduced to the executed part exactly as a registry would carry it."""
    policy = {k: loaded[k] for k in VALIDATOR_POLICY_KEYS if k in loaded and k != 'bucket_rules'}
    if loaded.get('bucket_rules'):
        policy['bucket_rules'] = registry_bucket_rules(loaded, kinds)['bucket_rules']
    return policy


def _fingerprint(*parts):
    return digest([inspect.getsource(x) if callable(x) else json.dumps(x, sort_keys=True, ensure_ascii=False, default=str) for x in parts])


def components():
    fp = dict(
        packet_builder=_fingerprint(sample_papers, select_papers, build_registry, registry_bucket_rules, list(BUCKET_RULE_KEYS), BUCKET_RULES_SUPPORTED,
                                    program_set_buckets, system_prompt, output_schema, _evidence_schema, _nullable,
                                    merged_fields, relevance_keys, textclean.clean_indexed, textclean._one_pass, textclean._allowed, textclean._is_block,
                                    textclean.compress, list(textclean.DEFAULT_TAGS), list(textclean.NAMESPACES), sorted(textclean.BLOCK_TAGS),
                                    textclean.ENTITY.pattern, textclean.TAG.pattern, textclean.MATH.pattern, textclean.LATEX.pattern),
        request_builder=_fingerprint(request_settings, list(EFFORT_LEVELS), build_request, paper_feedback, feedback_text, RETRY_HINTS, ModelCaller.retry_turn, ModelCaller.result_record),
        validator=_fingerprint(NORMALIZE_MAP, list(ACCEPTED_MATCH), normalize_indexed, locate_quote, _locate_all, literal_supported, merged_fields,
                               relevance_keys, program_set_buckets, PAPER_LEVEL_EXCLUDES, contributing_claims, paper_buckets, validate_claim, validate_profile,
                               validate_paper, textclean.expand, textclean.raw_span),
        store=_fingerprint(_store.STORE_VERSION, _store.input_key, _store.spec_key, _store.request_key, _store.make_spec, _store.Store._append,
                           _store.Store._read, _store.Store.put_inputs, _store.Store.put_answers, _store.Store.put_spec, digest))
    return {name: dict(version=COMPONENT_VERSIONS[name], fingerprint=fp[name]) for name in COMPONENT_VERSIONS}


def components_lock_status():
    current = components()
    locked = json.loads(LOCK_FILE.read_text()) if LOCK_FILE.exists() else {}
    return dict(current=current, locked=locked, in_sync=(locked == current))


# ---- the run -------------------------------------------------------------------------------------------------------



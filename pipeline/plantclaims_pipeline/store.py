"""Content-addressed extraction store: inputs, specifications, requests and model answers, each addressed by its content."""
import gzip
import hashlib
import json
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

from .records import digest, now

STORE_VERSION = 'store-v0.1'
SHARD_LIMIT = 256 * 1024 * 1024
MEMBER_RECORDS = 2000  # records per gzip member at most: reading one record decompresses one member
MEMBER_CACHE = 8       # decoded members kept per kind: validation alternates between an answers member and an inputs member
KINDS = ('inputs', 'answers')

SCHEMA = '''
CREATE TABLE IF NOT EXISTS members (
  kind TEXT NOT NULL, shard TEXT NOT NULL, offset INTEGER NOT NULL, length INTEGER NOT NULL, sha256 TEXT NOT NULL,
  n_lines INTEGER NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (kind, shard, offset));
CREATE TABLE IF NOT EXISTS inputs (
  input_key TEXT PRIMARY KEY, paper_id TEXT NOT NULL, text_sha256 TEXT NOT NULL, raw_text_sha256 TEXT,
  cleaning_version TEXT NOT NULL, publication_year INTEGER, source_json TEXT NOT NULL,
  shard TEXT NOT NULL, offset INTEGER NOT NULL, length INTEGER NOT NULL, line_no INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS inputs_paper ON inputs(paper_id);
CREATE TABLE IF NOT EXISTS specs (
  spec_key TEXT PRIMARY KEY, schema_version TEXT, model TEXT, effort TEXT, path TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS answers (
  request_key TEXT PRIMARY KEY, input_key TEXT NOT NULL REFERENCES inputs(input_key), spec_key TEXT NOT NULL REFERENCES specs(spec_key),
  paper_id TEXT NOT NULL, channel TEXT NOT NULL, attempt INTEGER NOT NULL, retry_of TEXT, replicate INTEGER NOT NULL,
  status TEXT NOT NULL, body_sha256 TEXT NOT NULL, provenance_json TEXT NOT NULL, meta_json TEXT NOT NULL, job_id TEXT,
  shard TEXT NOT NULL, offset INTEGER NOT NULL, length INTEGER NOT NULL, line_no INTEGER NOT NULL, ingested_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS answers_spec_paper ON answers(spec_key, paper_id, replicate, attempt);
CREATE INDEX IF NOT EXISTS answers_input ON answers(input_key);
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY, spec_key TEXT NOT NULL REFERENCES specs(spec_key), channel TEXT, selection_json TEXT NOT NULL,
  selection_sha256 TEXT NOT NULL, n_papers INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS job_papers (
  job_id TEXT NOT NULL REFERENCES jobs(job_id), seq INTEGER NOT NULL, paper_id TEXT NOT NULL, input_key TEXT NOT NULL REFERENCES inputs(input_key),
  PRIMARY KEY (job_id, paper_id));
CREATE INDEX IF NOT EXISTS job_papers_seq ON job_papers(job_id, seq);
'''


# ---- keys: every one is the hash of what it stands for -------------------------------------------------------
def input_key(paper_id, text_sha256, cleaning_version):
    return digest(['input', paper_id, text_sha256, cleaning_version])


def spec_key(spec):
    return digest(['spec', spec])


def request_key(ikey, skey, channel, attempt=1, retry_of=None, replicate=0):
    if channel not in ('api', 'interactive'):
        raise ValueError('channel must be api or interactive')
    return digest(['request', ikey, skey, channel, int(attempt), retry_of, int(replicate)])


def _ids_sha(ids):
    """Hash of an ordered id sequence without holding it: sha256 over `id\n`."""
    h = hashlib.sha256()
    for i in ids:
        h.update(i.encode('utf-8') + b'\n')
    return h.hexdigest()


def make_spec(system_prompt, registry, output_schema, request, input_fields, request_builder=None):
    """What is asked and how. `request` keeps only what changes the model's behaviour (not prices, caps, cache TTL);
    `input_fields` is what the model sees of a paper; `request_builder` is the component VERSION (a deliberate bump
    means requests are built differently, so earlier answers no longer answer the same question)."""
    return dict(system_prompt=system_prompt, registry=registry, output_schema=output_schema, input_fields=list(input_fields),
                request={**{k: request.get(k) for k in ('model', 'effort', 'max_tokens')},
                         **({'thinking': request['thinking']} if request.get('thinking') not in (None, 'adaptive') else {})},   # store-v0.1.1: only when set
                request_builder=request_builder)


class Store:
    def __init__(self, root, readonly=False):
        self.root = Path(root).resolve()
        self.dir = self.root / 'data' / 'extraction'
        self.readonly = readonly
        if readonly:
            if not (self.dir / 'extraction.sqlite').exists():
                raise ValueError('Extraction store does not exist yet')
            self.db = sqlite3.connect(f"file:{self.dir / 'extraction.sqlite'}?mode=ro", uri=True, timeout=60)
        else:
            for sub in KINDS + ('specs',):
                (self.dir / sub).mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.dir / 'extraction.sqlite', timeout=60)
            try:
                self.db.execute('PRAGMA journal_mode=WAL')
            except sqlite3.OperationalError:
                pass
            self.db.executescript(SCHEMA)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self._cache = {k: OrderedDict() for k in KINDS}   # kind -> {(shard, offset): decoded lines}, least recently used first

    def close(self):
        self.db.close()

    # ---- shard members ----
    def _shard_for(self, kind, incoming):
        shards = sorted((self.dir / kind).glob('shard-*.jsonl.gz'))
        if shards and shards[-1].stat().st_size + incoming <= SHARD_LIMIT:
            return shards[-1]
        n = int(shards[-1].name[6:12]) + 1 if shards else 1
        return self.dir / kind / f'shard-{n:06d}.jsonl.gz'

    def _append(self, kind, records):
        """Append one gzip member holding `records` (one JSON line each). Returns the locator; the caller writes the
        members row and its own index rows in ONE transaction afterwards."""
        if self.readonly:
            raise ValueError('Store is open read-only')
        lines = [json.dumps(r, ensure_ascii=False, sort_keys=True).encode('utf-8') + b'\n' for r in records]
        data = gzip.compress(b''.join(lines), mtime=0)
        shard = self._shard_for(kind, len(data))
        with open(shard, 'ab') as fh:
            fh.seek(0, os.SEEK_END)
            offset = fh.tell()
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        return dict(kind=kind, shard=shard.name, offset=offset, length=len(data), sha256=digest(data), n_lines=len(lines))

    def _member_row(self, loc):
        self.db.execute('INSERT INTO members VALUES (?,?,?,?,?,?,?)',
                        (loc['kind'], loc['shard'], loc['offset'], loc['length'], loc['sha256'], loc['n_lines'], now()))

    def _read(self, kind, shard, offset, length, line_no):
        cache, locator = self._cache[kind], (shard, offset)
        if locator in cache:
            cache.move_to_end(locator)
        else:
            with open(self.dir / kind / shard, 'rb') as fh:
                fh.seek(offset)
                data = fh.read(length)
            cache[locator] = gzip.decompress(data).split(b'\n')
            while len(cache) > MEMBER_CACHE:
                cache.popitem(last=False)
        return json.loads(cache[locator][line_no])

    # ---- inputs ----
    def put_inputs(self, records):
        """records: dicts with input_key, paper_id, text_sha256, cleaning_version, blocks (+ optional raw_text_sha256,
        publication_year, source, meta). Existing keys are skipped, so building inputs is idempotent."""
        fresh, seen = [], set()
        for r in records:
            if r['input_key'] != input_key(r['paper_id'], r['text_sha256'], r['cleaning_version']):
                raise ValueError('input_key does not match its content: ' + r['paper_id'])
            if r['input_key'] in seen or self.db.execute('SELECT 1 FROM inputs WHERE input_key=?', (r['input_key'],)).fetchone():
                continue
            seen.add(r['input_key']); fresh.append(r)
        for start in range(0, len(fresh), MEMBER_RECORDS):
            part = fresh[start:start + MEMBER_RECORDS]
            loc = self._append('inputs', part)
            with self.db:
                self._member_row(loc)
                for n, r in enumerate(part):
                    self.db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                                    (r['input_key'], r['paper_id'], r['text_sha256'], r.get('raw_text_sha256'), r['cleaning_version'], r.get('publication_year'),
                                     json.dumps(r.get('source') or {}, sort_keys=True), loc['shard'], loc['offset'], loc['length'], n, now()))
        return len(fresh)

    def get_input(self, ikey):
        row = self.db.execute('SELECT shard, offset, length, line_no FROM inputs WHERE input_key=?', (ikey,)).fetchone()
        if row is None:
            raise KeyError('No input ' + ikey)
        return self._read('inputs', *row)

    # ---- specs ----
    def put_spec(self, spec):
        key = spec_key(spec)
        path = self.dir / 'specs' / (key + '.json')
        data = (json.dumps(spec, ensure_ascii=False, sort_keys=True, indent=1) + '\n').encode('utf-8')
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError('Spec file differs from its key')
        else:
            with open(path, 'xb') as fh:
                fh.write(data)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO specs VALUES (?,?,?,?,?,?)',
                            (key, spec['registry'].get('schema_version'), spec['request'].get('model'), spec['request'].get('effort'),
                             str(path.relative_to(self.root)), now()))
        return key

    def get_spec(self, skey):
        row = self.db.execute('SELECT path FROM specs WHERE spec_key=?', (skey,)).fetchone()
        if row is None:
            raise KeyError('No spec ' + skey)
        spec = json.loads((self.root / row[0]).read_text(encoding='utf-8'))
        if spec_key(spec) != skey:
            raise ValueError('Spec file does not match its key')
        return spec

    # ---- answers ----
    def put_answers(self, answers, job_id=None):
        """answers: dicts with input_key, spec_key, paper_id, channel, attempt, retry_of, replicate, status, provenance,
        body (what the model returned: result + raw payload), meta. The request_key is computed here. Returns
        dict(new=[keys], cached=[keys]); a different body under an existing key refuses the whole call."""
        fresh, cached, seen = [], [], set()
        for a in answers:
            key = request_key(a['input_key'], a['spec_key'], a['channel'], a['attempt'], a.get('retry_of'), a.get('replicate', 0))
            body_sha = digest(a['body'])
            row = self.db.execute('SELECT body_sha256 FROM answers WHERE request_key=?', (key,)).fetchone()
            if row is not None or key in seen:
                if row is not None and row[0] != body_sha:
                    raise ValueError(f"A different answer is already stored for this request ({a['paper_id']}, attempt {a['attempt']}); the first answer is never overwritten")
                cached.append(key); continue
            seen.add(key)
            fresh.append(dict(a, request_key=key, body_sha256=body_sha))
        for start in range(0, len(fresh), MEMBER_RECORDS):
            part = fresh[start:start + MEMBER_RECORDS]
            loc = self._append('answers', [dict(request_key=a['request_key'], body=a['body']) for a in part])
            with self.db:
                self._member_row(loc)
                for n, a in enumerate(part):
                    self.db.execute('INSERT INTO answers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                    (a['request_key'], a['input_key'], a['spec_key'], a['paper_id'], a['channel'], int(a['attempt']), a.get('retry_of'),
                                     int(a.get('replicate', 0)), a['status'], a['body_sha256'], json.dumps(a['provenance'], ensure_ascii=False, sort_keys=True),
                                     json.dumps(a.get('meta') or {}, ensure_ascii=False, sort_keys=True), job_id,
                                     loc['shard'], loc['offset'], loc['length'], n, now()))
        return dict(new=[a['request_key'] for a in fresh], cached=cached)

    def get_answer(self, rkey):
        row = self.db.execute('SELECT rowid AS answer_seq, * FROM answers WHERE request_key=?', (rkey,)).fetchone()   # rowid: insertion order, never reused (answers are never deleted)
        if row is None:
            raise KeyError('No answer ' + rkey)
        stored = self._read('answers', row['shard'], row['offset'], row['length'], row['line_no'])
        if stored['request_key'] != rkey or digest(stored['body']) != row['body_sha256']:
            raise ValueError('Stored answer does not match its index row')
        out = {k: row[k] for k in row.keys() if k not in ('provenance_json', 'meta_json', 'shard', 'offset', 'length', 'line_no')}
        return dict(out, provenance=json.loads(row['provenance_json']), meta=json.loads(row['meta_json']), body=stored['body'])

    def latest_answer_rows(self, skey, job_id=None):
        """The answer that counts for each paper of a spec: highest attempt, replicate 0. Rows only (no bodies)."""
        join = 'JOIN job_papers j ON j.paper_id = a.paper_id AND j.job_id = ? ' if job_id else ''
        args = ([job_id] if job_id else []) + [skey]
        return self.db.execute(
            'SELECT a.* FROM answers a ' + join +
            'WHERE a.spec_key = ? AND a.replicate = 0 AND a.attempt = (SELECT max(b.attempt) FROM answers b WHERE b.spec_key = a.spec_key '
            'AND b.paper_id = a.paper_id AND b.replicate = 0) ORDER BY a.paper_id', args).fetchall()

    def provenance_summary(self, skey, job_id=None):
        """Who answered, as recorded: the model id the service reported;
        tokens and cost are sums of what the service reported and stay null where nothing was reported."""
        join = 'JOIN job_papers j ON j.job_id = ? AND j.input_key = a.input_key ' if job_id else ''
        rows = self.db.execute(
            "SELECT a.channel, json_extract(a.provenance_json, '$.model_id') AS model_id, "
            "json_extract(a.provenance_json, '$.model_id_as_stated_by_session') AS stated, a.attempt, count(*) AS answers, "
            "sum(json_extract(a.provenance_json, '$.input_tokens')) AS input_tokens, sum(json_extract(a.provenance_json, '$.output_tokens')) AS output_tokens, "
            "sum(json_extract(a.provenance_json, '$.cost_usd')) AS cost_usd FROM answers a " + join +
            'WHERE a.spec_key = ? GROUP BY 1, 2, 3, 4 ORDER BY 1, 2, 3, 4', ([job_id] if job_id else []) + [skey]).fetchall()
        return [dict(r) for r in rows]

    # ---- jobs (a selection of papers bound to a spec) ----
    def put_job(self, job_id, skey, papers, selection, channel=None):
        """papers: ordered list of (paper_id, input_key). Write-once per job id."""
        row = self.db.execute('SELECT spec_key, selection_sha256, n_papers FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is not None:
            sha = _ids_sha(p for p, _ in papers)
            if (row[0], row[1], row[2]) != (skey, sha, len(papers)):
                raise ValueError('Job exists with a different spec or selection')
            return sha
        self.begin_job(job_id, skey, selection, channel)
        self.add_job_papers(job_id, papers)
        return self.finish_job(job_id)

    # A large selection arrives in pieces (xinput.build), so a job can also be written in three steps.
    def begin_job(self, job_id, skey, selection, channel=None):
        with self.db:
            self.db.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?,?)', (job_id, skey, channel, json.dumps(selection, sort_keys=True), '', 0, now()))

    def add_job_papers(self, job_id, papers):
        if self.db.execute('SELECT selection_sha256 FROM jobs WHERE job_id=?', (job_id,)).fetchone()[0]:
            raise ValueError('Job selection is already closed')
        seq = self.db.execute('SELECT coalesce(max(seq), -1) + 1 FROM job_papers WHERE job_id=?', (job_id,)).fetchone()[0]
        with self.db:
            self.db.executemany('INSERT INTO job_papers VALUES (?,?,?,?)', [(job_id, seq + n, p, k) for n, (p, k) in enumerate(papers)])

    def finish_job(self, job_id, selection=None):
        sha = _ids_sha(r[0] for r in self.db.execute('SELECT paper_id FROM job_papers WHERE job_id=? ORDER BY seq', (job_id,)))
        n = self.db.execute('SELECT count(*) FROM job_papers WHERE job_id=?', (job_id,)).fetchone()[0]
        with self.db:
            self.db.execute('UPDATE jobs SET selection_sha256 = ?, n_papers = ? WHERE job_id = ?', (sha, n, job_id))
            if selection is not None:
                self.db.execute('UPDATE jobs SET selection_json = ? WHERE job_id = ?', (json.dumps(selection, sort_keys=True), job_id))
        return sha

    def job(self, job_id):
        row = self.db.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        if row is None:
            raise KeyError('No job ' + job_id)
        return dict(row)

    def resolve_spec(self, prefix):
        rows = self.db.execute('SELECT spec_key FROM specs WHERE spec_key LIKE ?', (prefix + '%',)).fetchall()
        if len(rows) != 1:
            raise ValueError(f'{len(rows)} specs match {prefix!r}')
        return rows[0][0]

    def stats(self):
        """What the store holds, from the index only (no shard is read)."""
        specs = [dict(r) for r in self.db.execute(
            'SELECT s.spec_key, s.schema_version, s.model, s.effort, (SELECT count(*) FROM answers a WHERE a.spec_key = s.spec_key) AS answers, '
            '(SELECT count(DISTINCT a.paper_id) FROM answers a WHERE a.spec_key = s.spec_key) AS papers_answered FROM specs s ORDER BY s.created_at')]
        jobs = [dict(r) for r in self.db.execute('SELECT job_id, spec_key, channel, n_papers, created_at FROM jobs ORDER BY created_at')]
        answers = [dict(r) for r in self.db.execute('SELECT channel, attempt, status, count(*) AS n FROM answers GROUP BY 1, 2, 3 ORDER BY 1, 2, 3')]
        files = {k: dict(shards=len(list((self.dir / k).glob('shard-*.jsonl.gz'))), bytes=sum(p.stat().st_size for p in (self.dir / k).glob('shard-*.jsonl.gz'))) for k in KINDS}
        db = self.dir / 'extraction.sqlite'
        return dict(store_version=STORE_VERSION, inputs=self.db.execute('SELECT count(*) FROM inputs').fetchone()[0],
                    papers=self.db.execute('SELECT count(DISTINCT paper_id) FROM inputs').fetchone()[0], specs=specs, jobs=jobs, answers=answers, files=files,
                    index_bytes=db.stat().st_size if db.exists() else 0)

    # ---- integrity ----
    def verify(self):
        """Re-hash every member, check that members tile their shard without overlap, and report bytes that no member
        covers (left by an interrupted append)."""
        problems, orphan, checked = [], 0, 0
        for kind in KINDS:
            by_shard = {}
            for m in self.db.execute('SELECT shard, offset, length, sha256, n_lines FROM members WHERE kind=? ORDER BY shard, offset', (kind,)):
                by_shard.setdefault(m['shard'], []).append(m)
            for shard in sorted({p.name for p in (self.dir / kind).glob('shard-*.jsonl.gz')} | set(by_shard)):
                path, pos = self.dir / kind / shard, 0
                if not path.exists():
                    problems.append(f'{kind}/{shard}: indexed but missing'); continue
                with open(path, 'rb') as fh:
                    for m in by_shard.get(shard, []):
                        if m['offset'] < pos:
                            problems.append(f'{kind}/{shard}@{m["offset"]}: members overlap')
                        orphan += max(0, m['offset'] - pos)
                        fh.seek(m['offset'])
                        data = fh.read(m['length'])
                        if digest(data) != m['sha256']:
                            problems.append(f'{kind}/{shard}@{m["offset"]}: hash mismatch')
                        elif gzip.decompress(data).count(b'\n') != m['n_lines']:
                            problems.append(f'{kind}/{shard}@{m["offset"]}: line count mismatch')
                        pos = max(pos, m['offset'] + m['length']); checked += 1
                orphan += max(0, path.stat().st_size - pos)
        counts = {t: self.db.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in ('members', 'inputs', 'specs', 'answers', 'jobs', 'job_papers')}
        indexed = {k: self.db.execute(f'SELECT coalesce(sum(n_lines), 0) FROM members WHERE kind=?', (k,)).fetchone()[0] for k in KINDS}
        for k in KINDS:
            if indexed[k] != counts[k]:
                problems.append(f'{k}: {counts[k]} index rows but {indexed[k]} stored lines')
        return dict(store_version=STORE_VERSION, ok=not problems, problems=problems, members_checked=checked, unindexed_bytes=orphan, counts=counts,
                    bytes={k: sum(p.stat().st_size for p in (self.dir / k).glob('shard-*.jsonl.gz')) for k in KINDS})

"""Candidate pool from the public OpenAlex snapshot (selection rules in configs/scope-v0.6.yaml), filtered offline after download."""

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .enumerate import SCHEMA, index_row, subfield_id
from .records import digest, now, save_json

EXTRA_COLUMNS = (('is_xpac', 'INTEGER'), ('has_content_pdf', 'INTEGER'), ('has_content_grobid', 'INTEGER'))
COLUMNS = ('run_id', 'openalex_id', 'doi', 'pmid', 'pmcid', 'publication_year', 'type', 'language', 'has_abstract',
           'abstract_status', 'oa_status', 'is_oa', 'primary_subfield_id', 'core_subfield_ids', 'is_retracted',
           'updated_date', 'raw_path', 'json_pointer', 'source_row_sha256') + tuple(c for c, _ in EXTRA_COLUMNS)


def ensure_schema(db):
    db.executescript(SCHEMA)
    present = {r[1] for r in db.execute('PRAGMA table_info(enumeration_records)')}
    for column, ctype in EXTRA_COLUMNS:
        if column not in present:
            db.execute(f'ALTER TABLE enumeration_records ADD COLUMN {column} {ctype}')


def relative_key(url):
    """s3://openalex/data/jsonl/works/updated_date=X/part_N.gz -> works/updated_date=X/part_N.gz"""
    m = re.search(r'/works/(updated_date=[0-9-]+/part_\d+\.gz)$', url)
    if not m:
        raise ValueError('Unexpected snapshot file url: ' + url)
    return 'works/' + m.group(1)


def http_fetch(url, target, chunk_bytes, timeout):
    """Stream one object to `target`, returning (bytes, sha256). Raises on HTTP/network errors."""
    sha = hashlib.sha256()
    size = 0
    temp = target.with_name(target.name + '.part')
    with requests.get(url, stream=True, timeout=timeout, headers={'User-Agent': 'PlantClaims/0.1 snapshot'}) as r:
        r.raise_for_status()
        with temp.open('wb') as out:
            for chunk in r.iter_content(chunk_bytes):
                out.write(chunk)
                sha.update(chunk)
                size += len(chunk)
    os.replace(temp, target)
    return size, sha.hexdigest()


class Snapshot:
    def __init__(self, run, fetch=None):
        self.run = run
        self.cfg = run.config['snapshot']
        self.fetch = fetch or (lambda url, target: http_fetch(url, target, self.cfg['download']['chunk_bytes'],
                                                              self.cfg['download']['timeout_seconds']))
        self.files_path = run.directory / 'files.json'
        self.files = json.loads(self.files_path.read_text()) if self.files_path.exists() else {}
        self.lock = threading.Lock()

    # ---------- stage 1: freeze the snapshot manifest ----------
    def fetch_manifest(self):
        if 'snapshot_input' in self.run.manifest:
            return self.run.manifest['snapshot_input']
        if self.run.replay:
            raise ValueError('No frozen snapshot input to replay')
        src = self.cfg['source']
        raw_manifest = self.run.raw / 'manifest-source.json'
        raw_deleted = self.run.raw / 'deleted_ids.csv'
        started = now()
        m_bytes, m_sha = self.fetch(f"{src['bucket_https']}/{src['manifest_key']}", raw_manifest)
        deleted_status = 'fetched'
        try:
            d_bytes, d_sha = self.fetch(f"{src['bucket_https']}/{src['deleted_ids_key']}", raw_deleted)
        except requests.HTTPError as exc:  # the deletion log is documented but absent in some releases
            if getattr(exc.response, 'status_code', None) != 404:
                raise
            d_bytes, d_sha, deleted_status = None, None, 'not_present_http_404'
            raw_deleted = None
        manifest = json.loads(raw_manifest.read_text())
        registry = json.loads((self.run.root / 'data' / 'runs' / self.cfg['p0_run'] / 'taxonomy-registry.json').read_text())
        core = {name: subfield_id(registry[name]) for name in self.cfg['predicate']['core_subfields']}
        if any(v is None for v in core.values()):
            raise ValueError('Unresolved core subfield in the P0 registry')
        files = [dict(key=relative_key(f['url']), url=f['url'],
                      https_url=f['url'].replace('s3://openalex/', src['bucket_https'].rstrip('/') + '/', 1),
                      content_length=f['meta']['content_length'], record_count=f['meta']['record_count']) for f in manifest['files']]
        if any(not f['https_url'].startswith('https://') for f in files):
            raise ValueError('Snapshot file url is not under the expected bucket')
        if len({f['key'] for f in files}) != len(files):
            raise ValueError('Duplicate file keys in the snapshot manifest')
        frozen = dict(snapshot_date=manifest['date'], format=manifest.get('format'), entity=manifest.get('entity'),
                      record_count=manifest['record_count'], content_length=manifest['content_length'],
                      file_count=len(files), files=files, manifest_sha256=m_sha, manifest_bytes=m_bytes,
                      deleted_ids_status=deleted_status, deleted_ids_sha256=d_sha, deleted_ids_bytes=d_bytes,
                      deleted_ids_rows=(max(0, sum(1 for _ in raw_deleted.open()) - 1) if raw_deleted else None),
                      scope_version=self.run.config['scope']['version'], registry_sha256=digest(registry),
                      core_subfield_ids=core, fetched_at=started,
                      note='snapshot_date is this corpus version as_of; records are candidates, not memberships')
        self.run.manifest['snapshot_input'] = frozen
        self.run.manifest['attempts'].append(dict(seq=len(self.run.manifest['attempts']) + 1, step='fetch-manifest',
                                                   url=f"{src['bucket_https']}/{src['manifest_key']}", request_hash=digest(src['manifest_key']),
                                                   started_at=started, finished_at=now(), status=200, bytes=m_bytes, sha256=m_sha,
                                                   raw_path=str(raw_manifest.relative_to(self.run.root))))
        self.run.flush()
        return frozen

    @property
    def input(self):
        return self.run.manifest['snapshot_input']

    @property
    def core_ids(self):
        return set(self.input['core_subfield_ids'].values())

    def _save_files(self):
        save_json(self.files_path, self.files)

    # ---------- stage 2: download and verify ----------
    def _download_one(self, f):
        target = self.run.raw / f['key']
        target.parent.mkdir(parents=True, exist_ok=True)
        entry = self.files.get(f['key'], {})
        if entry.get('download') == 'ok' and target.exists() and target.stat().st_size == f['content_length']:
            return f['key'], entry, False
        dl = self.cfg['download']
        last_error = None
        for attempt in range(1, dl['max_retries_per_file'] + 1):
            started = now()
            t0 = time.monotonic()
            try:
                size, sha = self.fetch(f['https_url'], target)
                if size != f['content_length']:
                    raise ValueError(f'content_length mismatch: got {size}, manifest {f["content_length"]}')
                entry.update(download='ok', bytes=size, sha256=sha, seconds=round(time.monotonic() - t0, 2),
                             attempts=attempt, started_at=started, finished_at=now(), error=None)
                return f['key'], entry, True
            except Exception as exc:  # network/HTTP/size errors are recorded, never hidden
                last_error = f'{type(exc).__name__}: {exc}'[:300]
                if target.exists() and not target.name.endswith('.part'):
                    target.unlink()
                time.sleep(dl['retry_pause_seconds'])
        entry.update(download='failed', error=last_error, attempts=dl['max_retries_per_file'], finished_at=now())
        return f['key'], entry, True

    def download(self, parallel=None, limit_keys=None):
        if self.run.replay:
            raise ValueError('download makes network requests; not allowed in replay')
        files = [f for f in self.input['files'] if not limit_keys or f['key'] in limit_keys]
        parallel = parallel or self.cfg['download']['parallel']
        done = 0
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = [pool.submit(self._download_one, f) for f in files]
            for fut in as_completed(futures):
                key, entry, changed = fut.result()
                with self.lock:
                    self.files[key] = entry
                    done += 1
                    if changed and done % self.cfg['download']['flush_every_files'] == 0:
                        self._save_files()
        self._save_files()
        self._sync_attempts()
        ok = [k for k in self.files if self.files[k].get('download') == 'ok' and (not limit_keys or k in limit_keys)]
        failed = [k for k in self.files if self.files[k].get('download') == 'failed' and (not limit_keys or k in limit_keys)]
        return dict(requested=len(files), ok=len(ok), failed=len(failed), failed_keys=failed[:20])

    def _sync_attempts(self):
        """One ledger row per downloaded file (idempotent by step name)."""
        existing = {a['step'] for a in self.run.manifest['attempts']}
        for key, e in sorted(self.files.items()):
            if 'download' in e and key not in existing:
                f = next(x for x in self.input['files'] if x['key'] == key)
                self.run.manifest['attempts'].append(dict(seq=len(self.run.manifest['attempts']) + 1, step=key, url=f['url'],
                                                           request_hash=digest(f['url']), started_at=e.get('started_at'),
                                                           finished_at=e.get('finished_at'), status=200 if e['download'] == 'ok' else None,
                                                           error=e.get('error'), bytes=e.get('bytes'), sha256=e.get('sha256'),
                                                           raw_path=str((self.run.raw / key).relative_to(self.run.root))))
        self.run.flush()

    # ---------- stage 3: filter retained raw parts ----------
    def filter(self, workers=None, keys=None, force=False, out_root=None):
        files = self.input['files']
        todo = []
        for f in files:
            e = self.files.get(f['key'], {})
            if e.get('download') != 'ok' or (keys and f['key'] not in keys):
                continue
            if not force and e.get('filter') == 'ok' and out_root is None:
                continue
            todo.append(f)
        out_root = out_root or (self.run.raw / 'matches')
        results = {}
        jobs = [(str(self.run.raw / f['key']), str(out_root / f['key'].replace('.gz', '.jsonl.gz')),
                 sorted(self.core_ids), self.cfg['predicate'], f['record_count'], self.cfg['filter']['decompressor']) for f in todo]
        with ProcessPoolExecutor(max_workers=workers or self.cfg['filter']['workers']) as pool:
            for f, res in zip(todo, pool.map(filter_part, jobs, chunksize=1)):
                results[f['key']] = res
                if out_root == self.run.raw / 'matches':
                    e = self.files.setdefault(f['key'], {})
                    e.update(filter='ok' if res['lines_read'] == f['record_count'] else 'count_mismatch', **res)
        if out_root == self.run.raw / 'matches':
            self._save_files()
        return results

    # ---------- stage 4: index matches into SQLite ----------
    def index(self, batch_rows=None):
        if self.run.replay:
            raise ValueError('index writes SQLite; not allowed in replay')
        ensure_schema(self.run.db)
        batch, total, dup, abstract, xpac = [], 0, 0, {}, 0
        seen = set(r[0] for r in self.run.db.execute('SELECT openalex_id FROM enumeration_records WHERE run_id=?', (self.run.id,)))
        placeholders = ','.join('?' * len(COLUMNS))
        sql = f'INSERT OR IGNORE INTO enumeration_records ({",".join(COLUMNS)}) VALUES ({placeholders})'
        for f in self.input['files']:
            e = self.files.get(f['key'], {})
            if e.get('filter') not in ('ok', 'count_mismatch') or e.get('indexed'):
                continue
            path = self.run.raw / 'matches' / f['key'].replace('.gz', '.jsonl.gz')
            rel = str(path.relative_to(self.run.root))
            with gzip.open(path, 'rb') as fh:
                for n, line in enumerate(fh):
                    work = json.loads(line)
                    if work['id'] in seen:
                        dup += 1
                        continue
                    seen.add(work['id'])
                    row = index_row(self.run.id, work, rel, f'/line/{n}', self.core_ids)
                    hc = work.get('has_content') or {}
                    row = row + (None if work.get('is_xpac') is None else int(bool(work.get('is_xpac'))),
                                 None if hc.get('pdf') is None else int(bool(hc.get('pdf'))),
                                 None if hc.get('grobid_xml') is None else int(bool(hc.get('grobid_xml'))))
                    abstract[row[9]] = abstract.get(row[9], 0) + 1
                    xpac += bool(work.get('is_xpac'))
                    batch.append(row)
                    total += 1
                    if len(batch) >= (batch_rows or self.cfg['index']['batch_rows']):
                        with self.run.db:
                            self.run.db.executemany(sql, batch)
                        batch = []
            e['indexed'] = True
        if batch:
            with self.run.db:
                self.run.db.executemany(sql, batch)
        self._save_files()
        return dict(indexed_new=total, duplicates_skipped=dup, abstract_status=abstract, xpac=xpac)

    # ---------- summaries ----------
    def summary(self):
        files = self.input['files']
        fs = self.files
        dl_ok = [k for k in fs if fs[k].get('download') == 'ok']
        flt = [k for k in fs if fs[k].get('filter') in ('ok', 'count_mismatch')]
        rows = self.run.db.execute('SELECT COUNT(*), COUNT(DISTINCT openalex_id) FROM enumeration_records WHERE run_id=?', (self.run.id,)).fetchone() \
            if not self.run.replay else (None, None)
        return dict(run_id=self.run.id, snapshot_date=self.input['snapshot_date'], scope_version=self.input['scope_version'],
                    files_total=len(files), files_downloaded=len(dl_ok),
                    files_failed=sum(fs[k].get('download') == 'failed' for k in fs),
                    bytes_downloaded=sum(fs[k].get('bytes') or 0 for k in dl_ok), bytes_expected=self.input['content_length'],
                    files_filtered=len(flt), files_count_mismatch=sum(fs[k].get('filter') == 'count_mismatch' for k in fs),
                    lines_read=sum(fs[k].get('lines_read') or 0 for k in flt), records_expected=self.input['record_count'],
                    matched=sum(fs[k].get('matched') or 0 for k in flt),
                    prefilter_hits=sum(fs[k].get('prefilter_hits') or 0 for k in flt),
                    indexed_rows=rows[0], indexed_distinct=rows[1], deleted_ids_rows=self.input['deleted_ids_rows'],
                    corpus_memberships=0, relevance_labels=0, confirmed_work_families=0)


TOKEN_TEMPLATE = b'/subfields/%s"'


def filter_part(job):
    """Worker: stream one gz part, keep lines whose work has type==article and a core-subfield topic.
    Returns counts and a content hash of the kept lines (order-preserving), for replay comparison."""
    src, dst, core_ids, predicate, expected, decompressor = job
    tokens = [TOKEN_TEMPLATE % c.encode() for c in core_ids]
    core = set(core_ids)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    lines = matched = pre = 0
    h = hashlib.sha256()
    proc = None
    if decompressor == 'pigz' and shutil.which('pigz'):
        proc = subprocess.Popen(['pigz', '-dc', src], stdout=subprocess.PIPE, bufsize=1 << 20)
        stream = proc.stdout
    else:
        stream = gzip.open(src, 'rb')
    temp = dst + '.part'
    with gzip.open(temp, 'wb', compresslevel=6) as out:
        for line in stream:
            lines += 1
            if not any(t in line for t in tokens):
                continue
            pre += 1
            work = json.loads(line)
            if work.get('type') != predicate['type']:
                continue
            if not any(subfield_id(t.get('subfield')) in core for t in (work.get('topics') or [])):
                continue
            if predicate.get('exclude_xpac') and work.get('is_xpac'):
                continue
            matched += 1
            out.write(line if line.endswith(b'\n') else line + b'\n')
            h.update(hashlib.sha256(line.rstrip(b'\n')).digest())
    if proc:
        proc.stdout.close()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f'pigz failed on {src} with code {rc}')
    else:
        stream.close()
    os.replace(temp, dst)
    return dict(lines_read=lines, prefilter_hits=pre, matched=matched, matches_sha256=h.hexdigest(),
                expected_records=expected, seconds=round(time.monotonic() - t0, 2))

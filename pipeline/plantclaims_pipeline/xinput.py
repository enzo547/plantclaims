"""Store inputs for large selections: from a candidate pool or an id list straight to cleaned inputs."""
import json
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .adapters import oa_record
from .candidates import TABLE, rule_sql
from .enumerate import subfield_id
from .extract import BLOCK_ID, TITLE_BLOCK_ID
from .records import digest
from .sample import abstract_quality, cell_id, cell_of, compile_rule, open_lines
from .store import MEMBER_RECORDS, input_key
from .textclean import clean_indexed, compress

LOCATE_COLUMNS = ('openalex_id', 'raw_path', 'json_pointer', 'source_row_sha256')


def bind_pool(root, db, pool_id, scope):
    """The pool's frozen rule and the snapshot run it selects from (same checks as the pilot sample's bind_pool)."""
    path = Path(root) / 'data' / 'runs' / pool_id / 'manifest.json'
    if not path.exists():
        raise ValueError('Pool run does not exist')
    manifest = json.loads(path.read_text())
    if manifest.get('kind') != 'candidate_pool' or manifest.get('status') != 'pool_built':
        raise ValueError('Pool must be a built candidate_pool run')
    row = db.execute('SELECT source_run_id, scope_version, rule_sql FROM candidate_pools WHERE pool_id=?', (pool_id,)).fetchone()
    if row is None:
        raise ValueError('Pool has no candidate_pools ledger row')
    source, scope_version, rule = row
    if rule != rule_sql(scope) or scope_version != scope['version']:
        raise ValueError('Pool rule or scope version differs from the current scope configuration')
    return dict(pool_id=pool_id, pool_manifest_sha256=digest(path.read_bytes()), source_run_id=source, scope_version=scope_version, rule_sql=rule)


def frame_sql(rule, frame):
    parts = [rule, f"{TABLE}.abstract_status IN ({','.join(repr(s) for s in frame['abstract_status'])})"]
    if frame.get('language'):
        parts.append(f"{TABLE}.language IN ({','.join(repr(s) for s in frame['language'])})")
    if frame.get('exclude_retracted'):
        parts.append(f"COALESCE({TABLE}.is_retracted, 0) = 0")
    return ' AND '.join(parts)


def locate_pool(db, source, where, limit=None):
    """Frame members as (raw_path, [(line, id, row hash), ...]) groups. Without a limit: every member, streamed in file
    order (memory = one file's rows). With a limit: the N smallest row hashes — a fixed pseudo-random subset that is
    spread over the match files the way a full build is."""
    cols = ','.join(LOCATE_COLUMNS)
    if limit:
        rows = db.execute(f'SELECT {cols} FROM {TABLE} WHERE {where} ORDER BY source_row_sha256 LIMIT ?', (source, limit)).fetchall()
        by_file = defaultdict(list)
        for oid, path, pointer, sha in rows:
            by_file[path].append((int(pointer.rsplit('/', 1)[-1]), oid, sha))
        yield from sorted(by_file.items())
        return
    current, group = None, []
    for oid, path, pointer, sha in db.execute(f'SELECT {cols} FROM {TABLE} WHERE {where} ORDER BY raw_path', (source,)):
        if path != current and group:
            yield current, group
            group = []
        current = path
        group.append((int(pointer.rsplit('/', 1)[-1]), oid, sha))
    if group:
        yield current, group


def locate_ids(db, source, where, ids):
    """An explicit id list: rows in 400-id chunks; ids outside the frame (or unknown) are returned, never silently dropped."""
    ids = [i if str(i).startswith('http') else 'https://openalex.org/' + str(i) for i in ids]
    cols = ','.join(LOCATE_COLUMNS)
    by_file, found = defaultdict(list), set()
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        for oid, path, pointer, sha in db.execute(f"SELECT {cols} FROM {TABLE} WHERE {where} AND openalex_id IN ({','.join('?' * len(chunk))})", (source, *chunk)):
            by_file[path].append((int(pointer.rsplit('/', 1)[-1]), oid, sha)); found.add(oid)
    return sorted(by_file.items()), [i for i in ids if i not in found]


def build_part(job):
    """Worker: one match file → input records for the wanted lines (+ exclusions with reason codes)."""
    root, rel, wanted, rule, strata, cleaning, source, decompressor = job
    compiled = compile_rule(rule)
    want = {line: (oid, sha) for line, oid, sha in wanted}
    last, records, excluded, read = max(want), [], [], 0
    for n, line in open_lines(Path(root) / rel, decompressor):
        read += 1
        if n in want:
            oid, sha = want[n]
            work = json.loads(line)
            if work.get('id') != oid or digest(work) != sha:
                raise ValueError(f'Retained record for {oid} does not match the index hash')
            rec = oa_record(work)
            chars, words, reasons = abstract_quality(rec['abstract'], compiled) if rec['abstract'] else (0, 0, ['abstract_not_reconstructable'])
            if reasons:
                excluded.append((oid, ','.join(reasons)))
            else:
                records.append(paper_input(work, rec['abstract'], sha, strata, cleaning, source))
        if n >= last:
            break
    missing = len(want) - len(records) - len(excluded)
    if missing:
        raise ValueError(f'{missing} wanted lines were not found in {rel}')
    return dict(rel=rel, lines_read=read, records=records, excluded=excluded)


def paper_input(work, abstract, index_sha, strata, cleaning, source):
    """Same shape as extract.sample_papers gives a packet paper, so the same paper + text + cleaning has the same input key."""
    title = work.get('title') or ''
    blocks = [dict(block_id=TITLE_BLOCK_ID, text=title), dict(block_id=BLOCK_ID, text=abstract)]
    extra = {}
    if cleaning:
        report = {}
        for b in blocks:
            clean, starts, ends, stats = clean_indexed(b['text'], tags=cleaning.get('tags') or (), passes=cleaning.get('passes', 2))
            b.update(raw_text=b['text'], text=clean, clean_to_raw=compress(starts, ends))
            report[b['block_id']] = stats
        extra = dict(raw_text_sha256=digest('\n'.join(b['raw_text'] for b in blocks).encode('utf-8')), cleaning=dict(version=cleaning.get('version'), blocks=report))
    version = (cleaning or {}).get('version') or 'none'
    text_sha = digest('\n'.join(b['text'] for b in blocks).encode('utf-8'))
    group, era = cell_of(strata, subfield_id((work.get('primary_topic') or {}).get('subfield')), work.get('publication_year'))
    return dict(paper_id=work['id'], index_sha256=index_sha, text_sha256=text_sha, cell_id=cell_id(group, era), primary_group=group, era=era, title=title,
                publication_year=work.get('publication_year'), blocks=blocks, input_key=input_key(work['id'], text_sha, version), cleaning_version=version,
                source=source, **extra)


def build(root, store, groups, rule, strata, cleaning, source, workers=16, decompressor='pigz', on_papers=None):
    """groups: iterable of (raw_path, [(line, id, row hash)]). Files are read by `workers` processes, at most 2×workers
    in flight, results taken in submission order (deterministic job order, bounded memory). Inputs go to the store in
    members of MEMBER_RECORDS; `on_papers([(paper_id, input_key)])` receives the job's papers in the same order."""
    t0 = time.monotonic()
    stats = dict(files=0, lines_read=0, selected=0, built=0, reused=0, excluded=0, exclusion_reasons=Counter())
    excluded, buffer = [], []

    def flush():
        nonlocal buffer
        if buffer:
            built = store.put_inputs(buffer)
            stats['built'] += built; stats['reused'] += len(buffer) - built
            if on_papers:
                on_papers([(r['paper_id'], r['input_key']) for r in buffer])
            buffer = []

    def take(res):
        stats['files'] += 1; stats['lines_read'] += res['lines_read']
        for oid, reasons in res['excluded']:
            excluded.append((oid, reasons)); stats['exclusion_reasons'].update(reasons.split(','))
        buffer.extend(res['records'])
        if len(buffer) >= MEMBER_RECORDS:
            flush()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        window = deque()
        for rel, wanted in groups:
            stats['selected'] += len(wanted)
            window.append(pool.submit(build_part, (str(root), rel, wanted, rule, strata, cleaning, source, decompressor)))
            if len(window) >= 2 * workers:
                take(window.popleft().result())
        while window:
            take(window.popleft().result())
    flush()
    stats['excluded'] = len(excluded)
    stats['exclusion_reasons'] = dict(stats['exclusion_reasons'])
    stats['seconds'] = round(time.monotonic() - t0, 2)
    return stats, excluded

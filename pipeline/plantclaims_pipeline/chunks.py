"""Chunk plan of the full run: one seeded random order of the extraction frame cut into chunks, so that each chunk can be
answered independently."""
import csv
import gzip
import hashlib
import io
import json
import resource
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from . import xinput
from .records import digest

CHUNK_RULES = dict(version='chunk-rules-v0.1', order='sha256(salt NUL group key), ascending; members of a group by openalex id',
                   cut='consecutive groups; chunk 0 until >= calibration_size papers, then chunks until >= chunk_size papers; groups never split',
                   group_key='doi:<normalised doi> for a DOI shared by 2..max_group_size frame records outside exclude_doi_prefixes, else id:<openalex id>')


def norm_doi(doi):
    d = (doi or '').strip().lower()
    for p in ('https://doi.org/', 'http://doi.org/', 'https://dx.doi.org/', 'http://dx.doi.org/', 'doi:'):
        if d.startswith(p):
            d = d[len(p):]
    return d or None


def order_key(salt, group):
    return hashlib.sha256((salt + '\0' + group).encode('utf-8')).hexdigest()


def frame_rows(run, cfg):
    """(openalex_id, doi) of the frame and a description of how it was obtained."""
    root = run.root
    sample_dir = root / 'data' / 'runs' / cfg['frame']['sample_run']
    summary = json.loads((sample_dir / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((sample_dir / 'manifest.json').read_text(encoding='utf-8'))
    if summary.get('pool_id') != cfg['frame']['pool']:
        raise ValueError(f"Sample run {cfg['frame']['sample_run']} was drawn from {summary.get('pool_id')}, not {cfg['frame']['pool']}")
    scfg = yaml.safe_load((root / 'configs' / 'pilot-sample-v0.yaml').read_text(encoding='utf-8'))
    used = (manifest.get('config_snapshot') or {}).get('pilot_sample') or {}
    if digest(used.get('abstract_rule')) != digest(scfg['abstract_rule']) or digest(used.get('frame')) != digest(scfg['frame']):
        raise ValueError('configs/pilot-sample-v0.yaml (frame or abstract rule) differs from what the sample run used; job create would select differently')
    exclusions = sample_dir / 'exclusions.csv'
    with exclusions.open(newline='', encoding='utf-8') as fh:
        excluded = {row['openalex_id'] for row in csv.DictReader(fh)}
    if len(excluded) != summary['excluded_count']:
        raise ValueError('exclusions.csv does not match the sample summary')
    bound = xinput.bind_pool(root, run.db, cfg['frame']['pool'], run.config['scope'])
    where = xinput.frame_sql(bound['rule_sql'], scfg['frame'])
    rows = [(oid, doi) for oid, doi in run.db.execute(f'SELECT openalex_id, doi FROM {xinput.TABLE} WHERE {where}', (bound['source_run_id'],))
            if oid not in excluded]
    if len(rows) != summary['frame_count']:
        raise ValueError(f"Frame has {len(rows)} records, the sample run counted {summary['frame_count']}")
    return rows, dict(pool=cfg['frame']['pool'], source_run_id=bound['source_run_id'], sample_run=cfg['frame']['sample_run'],
                      frame_where=where, excluded=len(excluded), exclusions_sha256=digest(exclusions.read_bytes()),
                      abstract_rule_sha256=digest(scfg['abstract_rule']), frame_count=len(rows))


def groups_of(rows, grouping):
    """{group key: [openalex ids]} and statistics. Only the DOI rule is implemented (title grouping is a follow-up)."""
    if grouping.get('key') != 'doi':
        raise ValueError('Only DOI grouping is implemented')
    by_doi = defaultdict(list)
    for oid, doi in rows:
        d = norm_doi(doi)
        if d:
            by_doi[d].append(oid)
    prefixes = tuple(grouping.get('exclude_doi_prefixes') or ())
    cap = int(grouping['max_group_size'])
    groups, stats = {}, Counter()
    oversize, excluded_prefix = [], []
    for oid, doi in rows:
        d = norm_doi(doi)
        members = by_doi.get(d) if d else None
        if members and len(members) > 1:
            if d.startswith(prefixes):
                if d not in excluded_prefix:
                    excluded_prefix.append(d)
            elif len(members) > cap:
                if d not in oversize:
                    oversize.append(d)
            else:
                groups.setdefault('doi:' + d, sorted(members))
                continue
        groups['id:' + oid] = [oid]
    for key, members in groups.items():
        if key.startswith('doi:'):
            stats['doi_groups'] += 1
            stats['papers_in_doi_groups'] += len(members)
    stats.update(doi_prefix_excluded_groups=len(excluded_prefix), doi_prefix_excluded_papers=sum(len(by_doi[d]) for d in excluded_prefix),
                 doi_oversize_groups=len(oversize), doi_oversize_papers=sum(len(by_doi[d]) for d in oversize))
    return groups, dict(stats, oversize=sorted((len(by_doi[d]), d) for d in oversize)[::-1][:20], excluded_prefix_examples=sorted(excluded_prefix)[:20])


def flag_sets(run, cfg, frame_ids):
    """{flag: set of frame openalex ids}."""
    root, out = run.root, {}
    for name, spec in (cfg.get('flags') or {}).items():
        src = spec['source']
        if src == 'sample_run':
            path = root / 'data' / 'runs' / cfg['frame']['sample_run'] / 'sample.csv'
            with path.open(newline='', encoding='utf-8') as fh:
                ids = {row['openalex_id'] for row in csv.DictReader(fh)}
        else:
            raise ValueError('Unknown flag source: ' + src)
        out[name] = ids & frame_ids['ids']
    return out


def plan(groups, salt, calibration_size, chunk_size):
    """[(chunk, group key, members)] in order."""
    ordered = sorted(groups.items(), key=lambda kv: order_key(salt, kv[0]))
    out, chunk, filled = [], 0, 0
    target = calibration_size
    for key, members in ordered:
        if filled >= target:
            chunk, filled, target = chunk + 1, 0, chunk_size
        out.append((chunk, key, members))
        filled += len(members)
    return out


def build(run):
    """The chunk_plan run: frame → groups → order → chunks, written once."""
    if run.replay or run.readonly:
        raise ValueError('build needs a writable, non-replay run')
    if (run.directory / 'summary.json').exists():
        raise ValueError('Run already built')
    cfg = run.config['chunking']
    rows, frame = frame_rows(run, cfg)
    groups, gstats = groups_of(rows, cfg['grouping'])
    by_doi = defaultdict(set)
    for oid, doi in rows:
        d = norm_doi(doi)
        if d:
            by_doi[d].add(oid)
    ids = {oid for oid, _ in rows}
    flags = flag_sets(run, cfg, dict(ids=ids, by_doi=by_doi))
    ordered = plan(groups, cfg['salt'], int(cfg['calibration_size']), int(cfg['chunk_size']))
    flag_of = defaultdict(list)
    for name in sorted(flags):
        for oid in flags[name]:
            flag_of[oid].append(name)
    chunk_dir = run.directory / 'chunks'
    chunk_dir.mkdir(parents=True, exist_ok=False)
    per_chunk, rank = defaultdict(lambda: dict(papers=0, groups=0, doi_groups=0, flags=Counter(), ids=[])), 0
    order_buf = io.StringIO()
    w = csv.writer(order_buf, lineterminator='\n')
    w.writerow(('rank', 'openalex_id', 'group', 'chunk', 'flags'))
    for chunk, key, members in ordered:
        c = per_chunk[chunk]
        c['groups'] += 1
        c['doi_groups'] += key.startswith('doi:')
        for oid in members:
            rank += 1
            w.writerow((rank, oid, key, chunk, ';'.join(flag_of.get(oid, ()))))
            c['papers'] += 1
            c['ids'].append(oid)
            for f in flag_of.get(oid, ()):
                c['flags'][f] += 1
    if rank != len(rows):
        raise ValueError('Order lost or duplicated papers')
    order_bytes = gzip.compress(order_buf.getvalue().encode('utf-8'), mtime=0)
    (run.directory / 'order.csv.gz').write_bytes(order_bytes)
    chunks, first = [], 1
    for n in sorted(per_chunk):
        c = per_chunk[n]
        data = ('\n'.join(c['ids']) + '\n').encode('utf-8')
        name = f'chunk-{n:03d}.txt'
        (chunk_dir / name).write_bytes(data)
        chunks.append(dict(chunk=n, role='calibration' if n == 0 else 'regular', file=f'chunks/{name}', papers=c['papers'], groups=c['groups'],
                           doi_groups=c['doi_groups'], first_rank=first, last_rank=first + c['papers'] - 1, ids_sha256=digest(data), flags=dict(sorted(c['flags'].items()))))
        first += c['papers']
    run.artifact('chunks.json', chunks)
    summary = dict(rules=CHUNK_RULES, config_version=cfg['version'], salt=cfg['salt'], frame=frame, grouping=gstats,
                   chunks=len(chunks), calibration_papers=chunks[0]['papers'], regular_chunk_papers=dict(min=min(c['papers'] for c in chunks[1:]) if len(chunks) > 1 else None,
                                                                                                         max=max(c['papers'] for c in chunks[1:]) if len(chunks) > 1 else None),
                   flags={k: len(v) for k, v in sorted(flags.items())}, order_sha256=digest(order_bytes),
                   peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    run.artifact('summary.json', summary)
    run.manifest['chunk_plan'] = dict(order_sha256=summary['order_sha256'], chunks=[{k: c[k] for k in ('chunk', 'papers', 'ids_sha256')} for c in chunks],
                                      summary_sha256=digest((run.directory / 'summary.json').read_bytes()),
                                      chunks_sha256=digest((run.directory / 'chunks.json').read_bytes()))
    run.finish('planned')
    return dict(run_id=run.id, papers=rank, chunks=len(chunks), calibration=chunks[0]['papers'], grouping={k: v for k, v in gstats.items() if not isinstance(v, list)},
                flags=summary['flags'], peak_rss_mb=summary['peak_rss_mb'], directory=str(run.directory))


def verify(root, run_id):
    """Offline check of a built plan: files against the manifest, every frame paper exactly once, chunk files = order rows."""
    d = Path(root) / 'data' / 'runs' / run_id
    m = json.loads((d / 'manifest.json').read_text(encoding='utf-8'))
    if m.get('kind') != 'chunk_plan' or m.get('status') != 'planned':
        raise ValueError('Not a built chunk_plan run: ' + run_id)
    cp, problems = m['chunk_plan'], []
    order = (d / 'order.csv.gz').read_bytes()
    if digest(order) != cp['order_sha256']:
        problems.append('order.csv.gz hash')
    for name, key in (('summary.json', 'summary_sha256'), ('chunks.json', 'chunks_sha256')):
        if digest((d / name).read_bytes()) != cp[key]:
            problems.append(name + ' hash')
    rows = list(csv.DictReader(io.StringIO(gzip.decompress(order).decode('utf-8'))))
    ids = [r['openalex_id'] for r in rows]
    if len(ids) != len(set(ids)):
        problems.append('duplicate ids in order')
    if [int(r['rank']) for r in rows] != list(range(1, len(rows) + 1)):
        problems.append('ranks not consecutive')
    by_chunk = defaultdict(list)
    for r in rows:
        by_chunk[int(r['chunk'])].append(r['openalex_id'])
    for c in cp['chunks']:
        data = (d / 'chunks' / f"chunk-{c['chunk']:03d}.txt").read_bytes()
        if digest(data) != c['ids_sha256']:
            problems.append(f"chunk {c['chunk']} hash")
        if data.decode('utf-8').split() != by_chunk[c['chunk']]:
            problems.append(f"chunk {c['chunk']} differs from the order")
    groups = defaultdict(set)
    for r in rows:
        groups[r['group']].add(int(r['chunk']))
    split = [g for g, cs in groups.items() if len(cs) > 1]
    if split:
        problems.append(f'{len(split)} groups split across chunks')
    return dict(run_id=run_id, ok=not problems, problems=problems, papers=len(rows), chunks=len(cp['chunks']))

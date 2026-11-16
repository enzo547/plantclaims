"""Stratified, seeded draw from a candidate pool; its frame is the paper set the full run extracts."""
import csv
import gzip
import hashlib
import heapq
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

from .adapters import oa_record, reconstruct_abstract
from .candidates import TABLE, rule_sql
from .enumerate import subfield_id
from .records import digest, now

SCHEMA = '''
CREATE TABLE IF NOT EXISTS pilot_samples (
  sample_id TEXT PRIMARY KEY REFERENCES runs(run_id), pool_id TEXT NOT NULL REFERENCES runs(run_id),
  seed INTEGER NOT NULL, target INTEGER NOT NULL, frame_sql TEXT NOT NULL, frame_count INTEGER NOT NULL,
  excluded_count INTEGER NOT NULL, sample_count INTEGER NOT NULL, sample_ids_sha256 TEXT NOT NULL,
  abstract_rule_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pilot_sample_members (
  sample_id TEXT NOT NULL REFERENCES pilot_samples(sample_id), openalex_id TEXT NOT NULL,
  cell_id TEXT NOT NULL, primary_group TEXT NOT NULL, era TEXT NOT NULL, rank_digest TEXT NOT NULL,
  PRIMARY KEY (sample_id, openalex_id));
CREATE TABLE IF NOT EXISTS pilot_sample_exclusions (
  sample_id TEXT NOT NULL REFERENCES pilot_samples(sample_id), openalex_id TEXT NOT NULL,
  cell_id TEXT NOT NULL, reasons TEXT NOT NULL, abstract_chars INTEGER NOT NULL, abstract_words INTEGER NOT NULL,
  PRIMARY KEY (sample_id, openalex_id));
'''
ROW_COLUMNS = ('openalex_id', 'doi', 'pmid', 'pmcid', 'publication_year', 'language', 'primary_subfield_id',
               'core_subfield_ids', 'oa_status', 'is_oa', 'is_retracted', 'raw_path', 'json_pointer',
               'source_row_sha256', 'has_content_pdf', 'has_content_grobid', 'updated_date')
CSV_COLUMNS = ('openalex_id', 'doi', 'pmid', 'pmcid', 'title', 'publication_year', 'language', 'primary_subfield_id',
               'primary_group', 'era', 'cell_id', 'core_subfield_ids', 'oa_status', 'is_oa', 'year_suspect',
               'has_content_pdf', 'has_content_grobid', 'abstract_chars', 'abstract_words', 'rank_digest')
WORD = re.compile(r'\S*[^\W_]\S*')


# ---- pure helpers ----------------------------------------------------------------------------------------
def cell_of(strata, primary_subfield_id, year):
    group = None
    for name, ids in strata['primary_groups'].items():
        if ids == 'other':
            continue
        if primary_subfield_id is not None and str(primary_subfield_id) in [str(i) for i in ids]:
            group = name
            break
    if group is None:
        group = next(name for name, ids in strata['primary_groups'].items() if ids == 'other')
    if year is None:
        era = strata['year_missing_label']
    else:
        era = None
        for label, lo, hi in strata['year_eras']:
            if (lo is None or year >= lo) and (hi is None or year <= hi):
                era = label
                break
        if era is None:
            raise ValueError(f'Year {year} falls outside every configured era')
    return group, era


def cell_id(group, era):
    return f'{group}/{era}'


def compile_rule(rule):
    return dict(min_chars=rule['min_chars'], min_words=rule['min_words'],
                patterns=[(code, re.compile(rx, re.I)) for code, rx in rule.get('placeholder_patterns', [])],
                citation=(re.compile(rule['citation_only']['pattern'], re.I), rule['citation_only']['max_chars']) if rule.get('citation_only') else None)


def abstract_quality(text, compiled):
    """Return (chars, words, reasons); an empty reason list means the text passes the real-abstract rule."""
    text = (text or '').strip()
    chars, words = len(text), len(WORD.findall(text))
    reasons = []
    if chars < compiled['min_chars']:
        reasons.append('too_short_chars')
    if words < compiled['min_words']:
        reasons.append('too_short_words')
    for code, rx in compiled['patterns']:
        if rx.search(text):
            reasons.append(code)
    if compiled['citation'] and chars < compiled['citation'][1] and compiled['citation'][0].search(text):
        reasons.append('citation_only')
    return chars, words, reasons


def allocate(populations, target, floor):
    """Proportional allocation with a per-cell floor and largest-remainder rounding.

    Cells whose (non-zero) population is below the floor get their whole population and a recorded shortfall;
    the deficit is redistributed proportionally over the remaining cells (recorded as an override). Empty cells
    are reported as empty and take no part. Cells whose proportional share is below the floor are pinned at the
    floor and the rest is re-proportioned; a cell can never receive more than its population. Deterministic.
    """
    cells = sorted(populations)
    fixed, shortfall, pinned = {}, {}, []
    for c in cells:
        if populations[c] == 0:
            fixed[c] = 0
        elif populations[c] < floor:
            fixed[c] = populations[c]
            shortfall[c] = floor - populations[c]
    while True:
        free = [c for c in cells if c not in fixed]
        remaining = target - sum(fixed.values())
        free_pop = sum(populations[c] for c in free)
        if remaining < 0:
            raise ValueError('Floors exceed the target')
        if not free:
            if remaining:
                raise ValueError('Target exceeds the frame population')
            break
        raw = {c: remaining * populations[c] / free_pop for c in free}
        low = [c for c in free if raw[c] < floor]
        over = [c for c in free if raw[c] > populations[c]]
        if low or over:
            for c in low:
                fixed[c] = min(floor, populations[c]); pinned.append(c)
            for c in over:
                fixed[c] = populations[c]; pinned.append(c)
            continue
        rounded = {c: int(raw[c]) for c in free}
        extra = remaining - sum(rounded.values())
        for c in sorted(free, key=lambda c: (-(raw[c] - rounded[c]), c))[:extra]:
            rounded[c] += 1
        fixed.update(rounded)
        break
    total_pop = sum(populations.values())
    out = {c: dict(population=populations[c], proportional_share=round(target * populations[c] / total_pop, 3) if total_pop else 0,
                   allocated=fixed[c], shortfall=shortfall.get(c, 0), empty=populations[c] == 0, pinned_at_floor=c in pinned,
                   inclusion_probability=(fixed[c] / populations[c]) if populations[c] else 0) for c in cells}
    deficit = sum(shortfall.values())
    override = dict(deficit=deficit, redistributed_to='proportional_cells', rule='largest_remainder_over_free_cells') if deficit else None
    if sum(fixed.values()) != target:
        raise ValueError('Allocation does not sum to the target')
    return dict(cells=out, target=target, floor=floor, override=override)


def open_lines(path, decompressor):
    """Yield (line_no, line_bytes) from a gzip file, via pigz when available (much faster on large parts)."""
    if decompressor and shutil.which(decompressor):
        proc = subprocess.Popen([decompressor, '-dc', str(path)], stdout=subprocess.PIPE, bufsize=1 << 20)
        try:
            for n, line in enumerate(proc.stdout):
                yield n, line
        finally:
            proc.stdout.close()
            proc.terminate()
            proc.wait()
    else:
        with gzip.open(path, 'rb') as fh:
            for n, line in enumerate(fh):
                yield n, line


def in_frame(work, core_ids, frame):
    """Mirror of the index predicate + frame SQL, evaluated on the raw work object."""
    if work.get('type') != 'article' or work.get('is_xpac'):
        return False
    if frame.get('exclude_retracted') and work.get('is_retracted'):
        return False
    if frame.get('language') and work.get('language') not in frame['language']:
        return False
    if not any(subfield_id(t.get('subfield')) in core_ids for t in (work.get('topics') or [])):
        return False
    return True


def scan_part(job):
    """Worker: one match file → in-frame count and the rows failing the abstract rule."""
    src, core_ids, frame, rule, decompressor = job
    compiled = compile_rule(rule)
    core = set(core_ids)
    lines = passed = 0
    excluded = []
    for _, line in open_lines(src, decompressor):
        lines += 1
        work = json.loads(line)
        if not in_frame(work, core, frame):
            continue
        try:
            text = reconstruct_abstract(work.get('abstract_inverted_index'))
        except ValueError:
            continue
        if not text:
            continue
        chars, words, reasons = abstract_quality(text, compiled)
        if reasons:
            excluded.append((work['id'], subfield_id((work.get('primary_topic') or {}).get('subfield')),
                             work.get('publication_year'), chars, words, ','.join(reasons)))
        else:
            passed += 1
    return dict(src=src, lines=lines, passed=passed, excluded=excluded)


def ids_sha256(ids):
    h = hashlib.sha256()
    for oid in sorted(ids):
        h.update(oid.encode()); h.update(b'\n')
    return h.hexdigest()


# ---- the run ---------------------------------------------------------------------------------------------
class PilotSample:
    def __init__(self, run):
        self.run = run
        self.cfg = run.config['pilot_sample']
        self.scope = run.config['scope']
        self.pool = self.cfg.get('pool')
        if not self.pool:
            raise ValueError('pilot_sample.pool is required')
        self.db = run.db
        self.strata = self.cfg['strata']
        self.source = None  # snapshot run id bound to the pool (the `?` in rule_sql)
        self.frame_sql = None
        self.frame_sql_no_retraction = None
        self.core_ids = None

    # ---- binding ----
    def bind_pool(self):
        path = self.run.root / 'data' / 'runs' / self.pool / 'manifest.json'
        if not path.exists():
            raise ValueError('Pool run does not exist')
        manifest = json.loads(path.read_text())
        if manifest.get('kind') != 'candidate_pool':
            raise ValueError('Pool must be a candidate_pool run')
        if manifest.get('status') != 'pool_built':
            raise ValueError(f"Pool is not built (status {manifest.get('status')})")
        ledger = self.db.execute('SELECT source_run_id, scope_version, rule_sql, member_count, members_sha256 FROM candidate_pools WHERE pool_id=?',
                                 (self.pool,)).fetchone() if self.db.execute("SELECT name FROM sqlite_master WHERE name='candidate_pools'").fetchone() else None
        if ledger is None:
            raise ValueError('Pool has no candidate_pools ledger row')
        source, scope_version, rule, member_count, members_sha = ledger
        if rule != rule_sql(self.scope) or scope_version != self.scope['version']:
            raise ValueError('Pool rule or scope version differs from the current scope configuration')
        src_path = self.run.root / 'data' / 'runs' / source / 'manifest.json'
        src_manifest = json.loads(src_path.read_text()) if src_path.exists() else {}
        snap = src_manifest.get('snapshot_input', {})
        core_ids = snap.get('core_subfield_ids')
        if not core_ids:
            raise ValueError('Source snapshot run has no frozen core subfield IDs')
        frame = self.cfg['frame']
        parts = [rule, f"{TABLE}.abstract_status IN ({','.join(repr(s) for s in frame['abstract_status'])})"]
        if frame.get('language'):
            parts.append(f"{TABLE}.language IN ({','.join(repr(s) for s in frame['language'])})")
        no_retraction = ' AND '.join(parts)
        if frame.get('exclude_retracted'):
            parts.append(f"COALESCE({TABLE}.is_retracted, 0) = 0")
        frame_sql = ' AND '.join(parts)
        bound = dict(pool_id=self.pool, pool_manifest_sha256=digest(path.read_bytes()), source_run_id=source,
                     snapshot_date=snap.get('snapshot_date') or snap.get('date'), core_subfield_ids=core_ids,
                     scope_version=scope_version, rule_sql=rule, member_count=member_count, members_sha256=members_sha,
                     frame_sql=frame_sql, frame=frame, abstract_rule=self.cfg['abstract_rule'], strata=self.strata,
                     seed=self.cfg['seed'], target=self.cfg['target'], min_per_cell=self.cfg['min_per_cell'],
                     role=self.cfg['role'], bound_at=now())
        stored = self.run.manifest.get('sample_input')
        strip = lambda d: {k: v for k, v in d.items() if k != 'bound_at'}  # noqa: E731
        if stored is not None and strip(stored) != strip(bound):
            raise ValueError('Sample input changed since this run was created; start a new sample run')
        if stored is None and not self.run.replay:
            self.run.manifest['sample_input'] = bound
            self.run.flush()
        self.source, self.frame_sql, self.frame_sql_no_retraction = source, frame_sql, no_retraction
        self.core_ids = sorted(core_ids.values())
        return bound

    # ---- stage: abstract-rule scan over the retained match files ----
    def scan_abstracts(self, workers=None):
        matches = self.run.root / 'data' / 'raw' / self.source / 'matches'
        files = sorted(str(p) for p in matches.rglob('*.jsonl.gz'))
        if not files:
            raise ValueError('No retained match files for the source snapshot run')
        jobs = [(f, self.core_ids, self.cfg['frame'], self.cfg['abstract_rule'], self.cfg['scan']['decompressor']) for f in files]
        passed, excluded, lines = 0, {}, 0
        with ProcessPoolExecutor(max_workers=workers or self.cfg['scan']['workers']) as pool:
            for res in pool.map(scan_part, jobs, chunksize=1):
                passed += res['passed']; lines += res['lines']
                for oid, sub, year, chars, words, reasons in res['excluded']:
                    if oid in excluded:
                        raise ValueError(f'Duplicate id across match files: {oid}')
                    excluded[oid] = dict(cell_id=cell_id(*cell_of(self.strata, sub, year)), chars=chars, words=words, reasons=reasons)
        frame_rows = self.db.execute(f"SELECT count(*) FROM {TABLE} WHERE {self.frame_sql}", (self.source,)).fetchone()[0]
        if passed + len(excluded) != frame_rows:
            raise ValueError(f'Scan saw {passed + len(excluded)} frame rows in the match files but the index has {frame_rows}')
        return dict(files=len(files), lines_read=lines, frame_rows=frame_rows, passed=passed, excluded=excluded)

    # ---- frame ----
    def frame_counts(self, excluded):
        suspect_below = self.strata.get('year_suspect_below') or 0
        index_pop, suspect, retracted = Counter(), Counter(), Counter()
        for sub, year, n, s in self.db.execute(
                f"SELECT primary_subfield_id, publication_year, count(*), sum(CASE WHEN publication_year < ? THEN 1 ELSE 0 END) "
                f"FROM {TABLE} WHERE {self.frame_sql} GROUP BY 1, 2", (suspect_below, self.source)):
            c = cell_id(*cell_of(self.strata, sub, year))
            index_pop[c] += n; suspect[c] += s or 0
        if self.cfg['frame'].get('exclude_retracted'):
            for sub, year, n in self.db.execute(
                    f"SELECT primary_subfield_id, publication_year, count(*) FROM {TABLE} WHERE {self.frame_sql_no_retraction} "
                    f"AND COALESCE({TABLE}.is_retracted, 0) = 1 GROUP BY 1, 2", (self.source,)):
                retracted[cell_id(*cell_of(self.strata, sub, year))] += n
        for group in self.strata['primary_groups']:
            for era in [e[0] for e in self.strata['year_eras']] + [self.strata['year_missing_label']]:
                index_pop.setdefault(cell_id(group, era), 0)
        excl_by_cell, reasons_by_cell = Counter(), defaultdict(Counter)
        for e in excluded.values():
            excl_by_cell[e['cell_id']] += 1
            for r in e['reasons'].split(','):
                reasons_by_cell[e['cell_id']][r] += 1
        populations = {c: index_pop[c] - excl_by_cell.get(c, 0) for c in index_pop}
        if any(v < 0 for v in populations.values()):
            raise ValueError('More exclusions than index rows in a cell')
        return dict(index_populations=dict(index_pop), excluded_by_cell=dict(excl_by_cell),
                    exclusion_reasons_by_cell={c: dict(v) for c, v in reasons_by_cell.items()}, populations=populations,
                    year_suspect=dict(suspect), retracted_excluded=dict(retracted), frame_count=sum(populations.values()))

    # ---- selection ----
    def draw(self, allocation, excluded):
        seed = self.cfg['seed']
        want = {c: v['allocated'] for c, v in allocation['cells'].items()}
        best = defaultdict(list)
        for oid, sub, year in self.db.execute(
                f"SELECT openalex_id, primary_subfield_id, publication_year FROM {TABLE} WHERE {self.frame_sql}", (self.source,)):
            if oid in excluded:
                continue
            c = cell_id(*cell_of(self.strata, sub, year))
            k = want.get(c, 0)
            if not k:
                continue
            lst = best[c]
            lst.append((digest({'seed': seed, 'openalex_id': oid}), oid))
            if len(lst) > 4 * k + 64:
                best[c] = heapq.nsmallest(k, lst)
        chosen = {}
        for c, k in want.items():
            for rank, oid in heapq.nsmallest(k, best.get(c, [])):
                chosen[oid] = dict(cell_id=c, rank_digest=rank)
        for c, k in want.items():
            got = sum(1 for v in chosen.values() if v['cell_id'] == c)
            if got != k:
                raise ValueError(f'Cell {c}: allocated {k}, selected {got}')
        return chosen

    def rows(self, ids):
        out = {}
        ids = list(ids)
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            sql = f"SELECT {','.join(ROW_COLUMNS)} FROM {TABLE} WHERE run_id=? AND openalex_id IN ({','.join('?' * len(chunk))})"
            for r in self.db.execute(sql, (self.source, *chunk)):
                out[r[0]] = dict(zip(ROW_COLUMNS, r))
        missing = [i for i in ids if i not in out]
        if missing:
            raise ValueError(f'{len(missing)} selected IDs are not in the index')
        return out

    # ---- materialization ----
    def materialize(self, chosen, rows):
        by_file = defaultdict(dict)
        for oid in chosen:
            r = rows[oid]
            by_file[r['raw_path']][int(r['json_pointer'].rsplit('/', 1)[-1])] = oid
        records, decomp = {}, self.cfg['scan']['decompressor']
        compiled = compile_rule(self.cfg['abstract_rule'])
        suspect_below = self.strata.get('year_suspect_below') or 0
        for rel, wanted in sorted(by_file.items()):
            last = max(wanted)
            for n, line in open_lines(self.run.root / rel, decomp):
                if n in wanted:
                    work = json.loads(line)
                    oid = wanted[n]
                    if work.get('id') != oid or digest(work) != rows[oid]['source_row_sha256']:
                        raise ValueError(f'Retained record for {oid} does not match the index hash')
                    rec = oa_record(work)
                    if rec['abstract_status'] != 'reconstructed' or not rec['abstract']:
                        raise ValueError(f'{oid}: abstract not reconstructable although the index says it is')
                    chars, words, reasons = abstract_quality(rec['abstract'], compiled)
                    if reasons:
                        raise ValueError(f'{oid}: selected record fails the abstract rule ({reasons}); scan and draw disagree')
                    if not in_frame(work, set(self.core_ids), self.cfg['frame']):
                        raise ValueError(f'{oid}: selected record is outside the frame predicate on the raw object')
                    group, era = chosen[oid]['cell_id'].split('/')
                    year = work.get('publication_year')
                    work['pilot'] = dict(sample_id=self.run.id, cell_id=chosen[oid]['cell_id'], primary_group=group, era=era,
                                         rank_digest=chosen[oid]['rank_digest'], abstract_text=rec['abstract'],
                                         abstract_chars=chars, abstract_words=words,
                                         year_suspect=bool(year is not None and year < suspect_below),
                                         index_sha256=rows[oid]['source_row_sha256'], source_run_id=self.source, pool_id=self.pool)
                    records[oid] = work
                if n >= last:
                    break
        missing = [oid for oid in chosen if oid not in records]
        if missing:
            raise ValueError(f'{len(missing)} selected records were not found in the retained match files')
        return records

    def write_outputs(self, records, rows, excluded):
        order = sorted(records, key=lambda oid: (records[oid]['pilot']['cell_id'], records[oid]['pilot']['rank_digest']))
        with open(self.run.directory / 'sample.jsonl', 'w', encoding='utf-8') as fh:
            for oid in order:
                fh.write(json.dumps(records[oid], ensure_ascii=False) + '\n')
        with open(self.run.directory / 'sample.csv', 'w', encoding='utf-8', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
            w.writeheader()
            for oid in order:
                work, p, r = records[oid], records[oid]['pilot'], rows[oid]
                w.writerow(dict(openalex_id=oid, doi=r['doi'], pmid=r['pmid'], pmcid=r['pmcid'], title=work.get('title'),
                                publication_year=r['publication_year'], language=r['language'], primary_subfield_id=r['primary_subfield_id'],
                                primary_group=p['primary_group'], era=p['era'], cell_id=p['cell_id'], core_subfield_ids=r['core_subfield_ids'],
                                oa_status=r['oa_status'], is_oa=r['is_oa'], year_suspect=int(p['year_suspect']),
                                has_content_pdf=r['has_content_pdf'], has_content_grobid=r['has_content_grobid'],
                                abstract_chars=p['abstract_chars'], abstract_words=p['abstract_words'], rank_digest=p['rank_digest']))
        with open(self.run.directory / 'exclusions.csv', 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(('openalex_id', 'cell_id', 'reasons', 'abstract_chars', 'abstract_words'))
            for oid in sorted(excluded):
                e = excluded[oid]
                w.writerow((oid, e['cell_id'], e['reasons'], e['chars'], e['words']))
        return order

    def distributions(self, records, rows):
        p = lambda oid: records[oid]['pilot']  # noqa: E731
        lengths = sorted(p(o)['abstract_chars'] for o in records)
        q = lambda f: lengths[min(len(lengths) - 1, int(f * len(lengths)))] if lengths else None  # noqa: E731
        return dict(by_group=dict(Counter(p(o)['primary_group'] for o in records)), by_era=dict(Counter(p(o)['era'] for o in records)),
                    by_oa_status=dict(Counter(str(rows[o]['oa_status']) for o in records)),
                    year_suspect=sum(1 for o in records if p(o)['year_suspect']),
                    with_pmid=sum(1 for o in records if rows[o]['pmid']), with_doi=sum(1 for o in records if rows[o]['doi']),
                    has_content_pdf=sum(1 for o in records if rows[o]['has_content_pdf']),
                    abstract_chars=dict(min=lengths[0] if lengths else None, p25=q(0.25), median=q(0.5), p75=q(0.75), max=lengths[-1] if lengths else None),
                    by_decade=dict(sorted(Counter(str((rows[o]['publication_year'] // 10) * 10) if rows[o]['publication_year'] is not None else 'null' for o in records).items())))

    # ---- actions ----
    def build(self, workers=None):
        if self.run.replay:
            raise ValueError('build writes the ledger; not allowed in replay')
        self.bind_pool()
        self.db.executescript(SCHEMA)
        if self.db.execute('SELECT 1 FROM pilot_samples WHERE sample_id=?', (self.run.id,)).fetchone():
            raise ValueError('Sample already built for this run')
        scan = self.scan_abstracts(workers)
        excluded = scan['excluded']
        frame = self.frame_counts(excluded)
        allocation = allocate(frame['populations'], self.cfg['target'], self.cfg['min_per_cell'])
        chosen = self.draw(allocation, excluded)
        rows = self.rows(chosen)
        records = self.materialize(chosen, rows)
        order = self.write_outputs(records, rows, excluded)
        sha = ids_sha256(chosen)
        reasons = Counter()
        for e in excluded.values():
            for r in e['reasons'].split(','):
                reasons[r] += 1
        cells = {c: dict(v, index_population=frame['index_populations'][c], excluded=frame['excluded_by_cell'].get(c, 0),
                         exclusion_reasons=frame['exclusion_reasons_by_cell'].get(c, {}),
                         retracted_excluded=frame['retracted_excluded'].get(c, 0), year_suspect=frame['year_suspect'].get(c, 0),
                         selected=sum(1 for o in chosen if chosen[o]['cell_id'] == c)) for c, v in allocation['cells'].items()}
        summary = dict(sample_id=self.run.id, pool_id=self.pool, source_run_id=self.source, scope_version=self.scope['version'],
                       seed=self.cfg['seed'], target=self.cfg['target'], min_per_cell=self.cfg['min_per_cell'], frame_sql=self.frame_sql,
                       index_frame_rows=scan['frame_rows'], scan=dict(files=scan['files'], lines_read=scan['lines_read'], passed=scan['passed']),
                       excluded_count=len(excluded), excluded_ids_sha256=ids_sha256(excluded), exclusion_reasons=dict(reasons),
                       retracted_excluded=sum(frame['retracted_excluded'].values()),
                       frame_count=frame['frame_count'], sample_count=len(chosen), sample_ids_sha256=sha,
                       allocation=dict(method=self.cfg['allocation'], override=allocation['override'],
                                       shortfall_total=sum(v['shortfall'] for v in allocation['cells'].values())),
                       distributions=self.distributions(records, rows), role=self.cfg['role'],
                       relevance_labels=0, corpus_memberships=0, gold=False,
                       files=dict(jsonl='sample.jsonl', csv='sample.csv', cells='cells.json', exclusions='exclusions.csv'))
        with self.db:
            self.db.execute('INSERT INTO pilot_samples VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                            (self.run.id, self.pool, self.cfg['seed'], self.cfg['target'], self.frame_sql, frame['frame_count'], len(excluded),
                             len(chosen), sha, json.dumps(self.cfg['abstract_rule'], sort_keys=True), now()))
            self.db.executemany('INSERT INTO pilot_sample_members VALUES (?,?,?,?,?,?)',
                                [(self.run.id, oid, m['cell_id'], *m['cell_id'].split('/'), m['rank_digest']) for oid, m in chosen.items()])
            self.db.executemany('INSERT INTO pilot_sample_exclusions VALUES (?,?,?,?,?,?)',
                                [(self.run.id, oid, e['cell_id'], e['reasons'], e['chars'], e['words']) for oid, e in excluded.items()])
        self.run.artifact('cells.json', dict(cells=cells, target=allocation['target'], floor=allocation['floor'], override=allocation['override']))
        self.run.artifact('summary.json', summary)
        self.run.manifest['checks'].append(dict(check='scan_frame_rows_match_index', passed=True, rows=scan['frame_rows'], at=now()))
        self.run.manifest['checks'].append(dict(check='sample_records_match_index_hashes_and_abstract_rule', passed=True, count=len(order), at=now()))
        self.run.manifest['checks'].append(dict(check='cells_sum_to_target', passed=len(chosen) == self.cfg['target'], at=now()))
        return summary

    def verify(self, workers=None):
        """Replay: rescan the match files, recompute frame, allocation and selection; re-hash the frozen sample.jsonl."""
        stored = json.loads((self.run.directory / 'summary.json').read_text())
        stored_cells = json.loads((self.run.directory / 'cells.json').read_text())['cells']
        self.bind_pool()
        scan = self.scan_abstracts(workers)
        excluded = scan['excluded']
        frame = self.frame_counts(excluded)
        allocation = allocate(frame['populations'], self.cfg['target'], self.cfg['min_per_cell'])
        chosen = self.draw(allocation, excluded)
        current = dict(frame_count=frame['frame_count'], excluded_count=len(excluded), excluded_ids_sha256=ids_sha256(excluded),
                       sample_count=len(chosen), sample_ids_sha256=ids_sha256(chosen), frame_sql=self.frame_sql,
                       allocated={c: v['allocated'] for c, v in allocation['cells'].items()})
        stored_cmp = dict(frame_count=stored['frame_count'], excluded_count=stored['excluded_count'], excluded_ids_sha256=stored['excluded_ids_sha256'],
                          sample_count=stored['sample_count'], sample_ids_sha256=stored['sample_ids_sha256'], frame_sql=stored['frame_sql'],
                          allocated={c: v['allocated'] for c, v in stored_cells.items()})
        diffs = {k: dict(stored=stored_cmp[k], current=current[k]) for k in stored_cmp if stored_cmp[k] != current[k]}
        ledger = {r[0] for r in self.db.execute('SELECT openalex_id FROM pilot_sample_members WHERE sample_id=?', (self.run.id,))}
        rows = self.rows(ledger) if ledger else {}
        compiled = compile_rule(self.cfg['abstract_rule'])
        file_ids, bad = [], []
        with open(self.run.directory / 'sample.jsonl', encoding='utf-8') as fh:
            for line in fh:
                work = json.loads(line)
                p = work.pop('pilot')
                file_ids.append(work['id'])
                ok = work['id'] in rows and digest(work) == rows[work['id']]['source_row_sha256'] == p['index_sha256']
                ok = ok and not abstract_quality(p['abstract_text'], compiled)[2] and oa_record(work)['abstract'] == p['abstract_text']
                if not ok:
                    bad.append(work['id'])
        if bad or set(file_ids) != ledger or ids_sha256(file_ids) != stored['sample_ids_sha256']:
            diffs['frozen_file'] = dict(bad_records=len(bad), file_records=len(file_ids), ledger_records=len(ledger))
        out = dict(sample_id=self.run.id, identical=not diffs, differing=diffs, current=current)
        self.run.artifact('verify.json', out)
        return out

    def status(self):
        cols = ('pool_id', 'seed', 'target', 'frame_count', 'excluded_count', 'sample_count', 'sample_ids_sha256', 'created_at')
        row = self.db.execute(f'SELECT {",".join(cols)} FROM pilot_samples WHERE sample_id=?', (self.run.id,)).fetchone() \
            if self.db.execute("SELECT name FROM sqlite_master WHERE name='pilot_samples'").fetchone() else None
        summary_path = self.run.directory / 'summary.json'
        return dict(sample_id=self.run.id, pool_id=self.pool, status=self.run.manifest.get('status'),
                    ledger=dict(zip(cols, row)) if row else None,
                    summary=json.loads(summary_path.read_text()) if summary_path.exists() else None)

"""Candidate-pool selection over a completed OpenAlex snapshot run (selection rules in configs/scope-v0.6.yaml)."""
import hashlib
import json
from collections import Counter

from .records import digest, now

TABLE = 'enumeration_records'
POOLS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS candidate_pools (
  pool_id TEXT PRIMARY KEY REFERENCES runs(run_id), source_run_id TEXT NOT NULL REFERENCES runs(run_id),
  scope_version TEXT NOT NULL, rule_sql TEXT NOT NULL, member_count INTEGER NOT NULL,
  members_sha256 TEXT NOT NULL, created_at TEXT NOT NULL)
'''
STRATA = (('S1_pre2005', None, 2004), ('S2_2005_2017', 2005, 2017), ('S3_2018_onwards', 2018, None), ('year_missing', None, None))


def rule_sql(scope):
    pool = scope['candidate_pool']
    parts = [f"{TABLE}.run_id = ?", f"{TABLE}.type = 'article'", f"{TABLE}.core_subfield_ids <> ''"]
    if pool.get('exclude_xpac'):
        parts.append(f"COALESCE({TABLE}.is_xpac, 0) = 0")
    return ' AND '.join(parts)


class CandidatePool:
    def __init__(self, run):
        self.run = run
        self.scope = run.config['scope']
        self.source = run.config['candidate_pool'].get('source_run')
        if not self.source:
            raise ValueError('candidate_pool.source_run is required')
        self.where = rule_sql(self.scope)
        self.db = run.db

    # ---- input binding -------------------------------------------------------------------------
    def bind_source(self):
        path = self.run.root / 'data' / 'runs' / self.source / 'manifest.json'
        if not path.exists():
            raise ValueError('Source run does not exist')
        manifest = json.loads(path.read_text())
        if manifest.get('kind') != 'snapshot_filter':
            raise ValueError('Source run must be a snapshot_filter run')
        if manifest.get('status') != 'indexed':
            raise ValueError(f"Source run is not indexed (status {manifest.get('status')})")
        summary_path = path.parent / 'summary.json'
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        bound = dict(source_run_id=self.source, source_manifest_sha256=digest(path.read_bytes()),
                     snapshot_date=manifest.get('snapshot_input', {}).get('date'),
                     source_indexed_rows=summary.get('indexed_rows'), source_matched=summary.get('matched'),
                     scope_version=self.scope['version'], rule_sql=self.where, bound_at=now())
        stored = self.run.manifest.get('pool_input')
        if stored is not None and {k: v for k, v in stored.items() if k != 'bound_at'} != {k: v for k, v in bound.items() if k != 'bound_at'}:
            raise ValueError('Pool input changed since this run was created; start a new pool run')
        if stored is None and not self.run.replay:
            self.run.manifest['pool_input'] = bound
            self.run.flush()
        return bound

    # ---- counting ------------------------------------------------------------------------------
    def counts(self):
        q = lambda sql, *a: self.db.execute(sql, (self.source, *a)).fetchall()  # noqa: E731
        total = q(f"SELECT count(*) FROM {TABLE} WHERE {self.where}")[0][0]
        excluded_xpac = q(f"SELECT count(*) FROM {TABLE} WHERE {TABLE}.run_id = ? AND COALESCE({TABLE}.is_xpac, 0) = 1")[0][0]
        source_rows = q(f"SELECT count(*) FROM {TABLE} WHERE {TABLE}.run_id = ?")[0][0]
        abstract = dict(q(f"SELECT abstract_status, count(*) FROM {TABLE} WHERE {self.where} GROUP BY 1"))
        strata = Counter()
        for year, n in q(f"SELECT publication_year, count(*) FROM {TABLE} WHERE {self.where} GROUP BY 1"):
            strata[stratum(year)] += n
        decades = {str(k): v for k, v in sorted(q(f"SELECT (publication_year/10)*10, count(*) FROM {TABLE} WHERE {self.where} AND publication_year IS NOT NULL GROUP BY 1"))}
        language = dict(q(f"SELECT COALESCE(language,'null'), count(*) FROM {TABLE} WHERE {self.where} GROUP BY 1 ORDER BY 2 DESC LIMIT 12"))
        oa = dict(q(f"SELECT COALESCE(oa_status,'null'), count(*) FROM {TABLE} WHERE {self.where} GROUP BY 1"))
        h = hashlib.sha256()
        members = 0
        for (oid,) in self.db.execute(f"SELECT openalex_id FROM {TABLE} WHERE {self.where} ORDER BY openalex_id", (self.source,)):
            h.update(oid.encode()); h.update(b'\n'); members += 1
        if members != total:
            raise ValueError('Member digest count differs from count(*)')
        return dict(pool_id=self.run.id, source_run_id=self.source, scope_version=self.scope['version'], rule_sql=self.where,
                    source_index_rows=source_rows, excluded_xpac=excluded_xpac, member_count=total, members_sha256=h.hexdigest(),
                    abstract_status=abstract,
                    abstract_present=abstract.get('reconstructed', 0) + abstract.get('invalid_index', 0),
                    strata_recommended_2005_2018=dict(strata), by_decade=decades, language_top=language, oa_status=oa,
                    role='candidate_pool_not_corpus_membership', relevance_labels=0, corpus_memberships=0, confirmed_work_families=0)

    # ---- actions -------------------------------------------------------------------------------
    def build(self):
        if self.run.replay:
            raise ValueError('build writes the ledger; not allowed in replay')
        self.bind_source()
        self.db.executescript(POOLS_SCHEMA)
        if self.db.execute('SELECT 1 FROM candidate_pools WHERE pool_id=?', (self.run.id,)).fetchone():
            raise ValueError('Pool already built for this run')
        res = self.counts()
        with self.db:
            self.db.execute('INSERT INTO candidate_pools VALUES (?,?,?,?,?,?,?)',
                            (self.run.id, self.source, res['scope_version'], res['rule_sql'], res['member_count'], res['members_sha256'], now()))
        self.run.artifact('summary.json', res)
        self.run.manifest['checks'].append(dict(check='member_digest_matches_count', passed=True, at=now()))
        return res

    def verify(self):
        """Replay: recompute counts and digest from the current index and compare with the frozen summary."""
        stored = json.loads((self.run.directory / 'summary.json').read_text())
        self.bind_source()
        res = self.counts()
        keys = ('member_count', 'members_sha256', 'excluded_xpac', 'abstract_status', 'rule_sql', 'source_index_rows')
        diffs = {k: dict(stored=stored.get(k), current=res.get(k)) for k in keys if stored.get(k) != res.get(k)}
        out = dict(pool_id=self.run.id, identical=not diffs, differing=diffs, current=res)
        self.run.artifact('verify.json', out)
        return out

    def ids(self):
        for (oid,) in self.db.execute(f"SELECT openalex_id FROM {TABLE} WHERE {self.where} ORDER BY openalex_id", (self.source,)):
            yield oid

    def status(self):
        row = self.db.execute('SELECT member_count, members_sha256, created_at FROM candidate_pools WHERE pool_id=?', (self.run.id,)).fetchone() \
            if self.db.execute("SELECT name FROM sqlite_master WHERE name='candidate_pools'").fetchone() else None
        summary_path = self.run.directory / 'summary.json'
        return dict(pool_id=self.run.id, source_run_id=self.source, scope_version=self.scope['version'], rule_sql=self.where,
                    ledger=dict(member_count=row[0], members_sha256=row[1], created_at=row[2]) if row else None,
                    summary=json.loads(summary_path.read_text()) if summary_path.exists() else None, status=self.run.manifest.get('status'))


def stratum(year):
    if year is None:
        return 'year_missing'
    for name, lo, hi in STRATA[:3]:
        if (lo is None or year >= lo) and (hi is None or year <= hi):
            return name
    return 'year_missing'

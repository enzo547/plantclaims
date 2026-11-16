"""Named releases over the extraction store: a release pins (job selection, spec, derivation, answer watermark)."""
import csv
import gzip
import io
import json
import re
from pathlib import Path

from .derive import Deriver, unpack
from .records import digest, now, save_json

SCHEMA = '''
CREATE TABLE IF NOT EXISTS releases (
  name TEXT PRIMARY KEY, job_id TEXT NOT NULL, spec_key TEXT NOT NULL, derivation_id TEXT NOT NULL, answer_seq_max INTEGER NOT NULL,
  selection_sha256 TEXT NOT NULL, claim_ids_sha256 TEXT NOT NULL, n_claims INTEGER NOT NULL, stats_json TEXT NOT NULL, note TEXT, created_at TEXT NOT NULL);
'''
NAME = re.compile(r'[a-z0-9][a-z0-9._-]{0,63}')
REVIEW_COLUMNS = ('revision_id', 'paper_id', 'local_id', 'kind', 'epistemic_status', 'polarity', 'basis', 'statement', 'quotes', 'fields', 'missing',
                  'needs_schema_review', 'direction_inferred', 'human_verdict', 'human_note')


class Releases:
    def __init__(self, store):
        self.store, self.db, self.deriver = store, store.db, Deriver(store)
        if not store.readonly:
            self.db.executescript(SCHEMA)

    def _has_table(self):
        return self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='releases'").fetchone() is not None

    def create(self, name, job_id, derivation_id=None, note=None):
        if not NAME.fullmatch(name or ''):
            raise ValueError('Release names: lowercase letters, digits, . _ - (at most 64 characters)')
        if self.db.execute('SELECT 1 FROM releases WHERE name=?', (name,)).fetchone():
            raise ValueError('Release exists; a release is never redefined (pick a new name)')
        job = self.store.job(job_id)
        did = derivation_id or Deriver.current_id(job['spec_key'])
        d = self.deriver.derivation(did)
        if d['spec_key'] != job['spec_key']:
            raise ValueError('Derivation belongs to another spec than the job')
        if self.deriver.behind(did):
            raise ValueError('Derivation has not validated every stored answer of its spec yet; run `derive run` first')
        upto = self.db.execute('SELECT coalesce(max(answer_seq), 0) FROM derived_papers WHERE derivation_id=?', (did,)).fetchone()[0]
        stats = self.deriver.stats(did, job_id, upto=upto)
        sha, n = self.deriver.claim_ids_sha256(did, job_id, upto=upto)
        pin = dict(name=name, job_id=job_id, spec_key=job['spec_key'], derivation_id=did, validator_version=d['validator_version'],
                   validator_fingerprint=d['validator_fingerprint'], policy=d['policy'], answer_seq_max=upto, selection_sha256=job['selection_sha256'],
                   selection=json.loads(job['selection_json']), claim_ids_sha256=sha, n_claims=n, stats=stats, note=note, created_at=now(),
                   role='development_candidates_not_gold', human_reviewed=0)
        path = self.store.dir / 'releases' / (name + '.json')
        if path.exists():
            raise ValueError('Release file exists without an index row; look at it before doing anything else: ' + str(path))
        save_json(path, pin)
        with self.db:
            self.db.execute('INSERT INTO releases VALUES (?,?,?,?,?,?,?,?,?,?,?)', (name, job_id, job['spec_key'], did, upto, job['selection_sha256'], sha, n,
                                                                                 json.dumps(stats, sort_keys=True), note, pin['created_at']))
        return pin

    def get(self, name):
        row = self.db.execute('SELECT * FROM releases WHERE name=?', (name,)).fetchone() if self._has_table() else None
        if row is None:
            raise KeyError('No release ' + str(name))
        return dict(row, stats=json.loads(row['stats_json']))

    def list(self):
        if not self._has_table():
            return []
        return [{k: r[k] for k in ('name', 'job_id', 'derivation_id', 'answer_seq_max', 'n_claims', 'claim_ids_sha256', 'note', 'created_at')}
                for r in self.db.execute('SELECT * FROM releases ORDER BY created_at')]

    def verify(self, name):
        """Recompute the pinned claim set from the store. Later answers never change it; a lost derivation or a damaged store does."""
        r = self.get(name)
        sha, n = self.deriver.claim_ids_sha256(r['derivation_id'], r['job_id'], upto=r['answer_seq_max'])
        stats = self.deriver.stats(r['derivation_id'], r['job_id'], upto=r['answer_seq_max'])
        file_pin = json.loads((self.store.dir / 'releases' / (name + '.json')).read_text(encoding='utf-8'))
        ok = sha == r['claim_ids_sha256'] == file_pin['claim_ids_sha256'] and stats == r['stats'] == file_pin['stats']
        return dict(name=name, identical=ok, claims=n, claim_ids_sha256=sha, pinned=r['claim_ids_sha256'])

    def export(self, name, out_dir):
        """claims.jsonl.gz (accepted claims with the model provenance of their answer), a blank claim-review.csv and
        papers.csv. Written on demand; the store itself never keeps whole-job files."""
        r = self.get(name)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        scope, args = self.deriver._scope(r['derivation_id'], r['job_id'], False, 'derived_claims', r['answer_seq_max'])
        review = io.StringIO()
        writer = csv.writer(review)
        writer.writerow(REVIEW_COLUMNS)
        n = 0
        with gzip.open(out / 'claims.jsonl.gz', 'wt', encoding='utf-8') as fh:
            for row in self.db.execute("SELECT t.revision_id, t.request_key, t.data_z, (SELECT a.provenance_json FROM answers a WHERE a.request_key = t.request_key) AS prov "
                                       + scope + " AND t.disposition = 'accepted' ORDER BY t.paper_id, t.position", args):
                c = unpack(row['data_z'])
                fh.write(json.dumps(dict(c, model_provenance=json.loads(row['prov'])), ensure_ascii=False, sort_keys=True) + '\n')
                writer.writerow([c['revision_id'], c['paper_id'], c['local_id'], c['kind'], c['epistemic_status'], c['polarity'], c.get('basis', ''), c['statement'],
                                 ' || '.join(e['quote'] for e in c['evidence']), json.dumps({k: v['value'] for k, v in c['fields'].items()}, ensure_ascii=False),
                                 json.dumps(c['missing'], ensure_ascii=False), c.get('needs_schema_review') or '', int(bool(c.get('direction_inferred'))), '', ''])
                n += 1
        (out / 'claim-review.csv').write_text(review.getvalue(), encoding='utf-8')
        pscope, pargs = self.deriver._scope(r['derivation_id'], r['job_id'], False, 'derived_papers', r['answer_seq_max'])
        with open(out / 'papers.csv', 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(('paper_id', 'attempt', 'status', 'label', 'reason_code', 'plant_subject', 'buckets', 'input_issues', 'claims_proposed', 'claims_accepted', 'claims_review', 'claims_quarantined'))
            for p in self.db.execute('SELECT t.* ' + pscope + ' ORDER BY t.paper_id', pargs):
                w.writerow((p['paper_id'], p['attempt'], p['status'], p['label'], p['reason_code'], p['plant_subject'], p['buckets_json'], p['input_issues_json'],
                            p['claims_proposed'], p['claims_accepted'], p['claims_review'], p['claims_quarantined']))
        if n != r['n_claims']:
            raise ValueError(f"Exported {n} claims but the release pins {r['n_claims']}")
        manifest = dict(release=name, exported_at=now(), claims=n, claim_ids_sha256=r['claim_ids_sha256'],
                        files={f: digest((out / f).read_bytes()) for f in ('claims.jsonl.gz', 'claim-review.csv', 'papers.csv')},
                        role='development_candidates_not_gold', human_reviewed=0)
        save_json(out / 'export.json', manifest)
        return manifest

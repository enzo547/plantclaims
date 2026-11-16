"""Derivations over the extraction store: the validator run over stored answers, keyed by validator version and policy.
A derivation never calls a model."""
import json
import zlib
from collections import Counter

from .extract import VALIDATOR_POLICY_KEYS, components, paper_feedback, validate_paper
from .records import digest, now

SCHEMA = '''
CREATE TABLE IF NOT EXISTS derivations (
  derivation_id TEXT PRIMARY KEY, spec_key TEXT NOT NULL, validator_version TEXT NOT NULL, validator_fingerprint TEXT NOT NULL,
  policy_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS derived_papers (
  derivation_id TEXT NOT NULL REFERENCES derivations(derivation_id), request_key TEXT NOT NULL, paper_id TEXT NOT NULL, input_key TEXT NOT NULL,
  channel TEXT NOT NULL, replicate INTEGER NOT NULL, attempt INTEGER NOT NULL, answer_seq INTEGER NOT NULL, status TEXT NOT NULL, label TEXT, reason_code TEXT, plant_subject TEXT, buckets_json TEXT NOT NULL, input_issues_json TEXT NOT NULL,
  claims_proposed INTEGER NOT NULL, claims_accepted INTEGER NOT NULL, claims_review INTEGER NOT NULL, claims_quarantined INTEGER NOT NULL,
  errors_json TEXT NOT NULL, error TEXT, PRIMARY KEY (derivation_id, request_key));
CREATE INDEX IF NOT EXISTS derived_papers_series ON derived_papers(derivation_id, input_key, channel, replicate, attempt);
CREATE TABLE IF NOT EXISTS derived_claims (
  derivation_id TEXT NOT NULL REFERENCES derivations(derivation_id), request_key TEXT NOT NULL, position INTEGER NOT NULL, paper_id TEXT NOT NULL,
  attempt INTEGER NOT NULL, local_id TEXT, revision_id TEXT, legacy_revision_id TEXT, disposition TEXT NOT NULL, kind TEXT, epistemic_status TEXT,
  origin TEXT, polarity TEXT, statistical_conclusion TEXT, basis TEXT, buckets_json TEXT NOT NULL, quote_match TEXT, direction_inferred INTEGER,
  needs_schema_review TEXT, errors_json TEXT NOT NULL, data_z BLOB NOT NULL, PRIMARY KEY (derivation_id, request_key, position));
CREATE INDEX IF NOT EXISTS derived_claims_rev ON derived_claims(derivation_id, revision_id);
CREATE INDEX IF NOT EXISTS derived_claims_paper ON derived_claims(derivation_id, paper_id);
'''
COUNT_KEYS = ('quarantined_papers', 'claims_proposed', 'claims_accepted', 'claims_review', 'claims_quarantined')


def pack(value):
    """Claim detail is the bulk of the index (about 2 KB of JSON per claim, once per derivation); the columns beside it
    carry everything the statistics need, so the detail is stored deflated and opened only for export."""
    return zlib.compress(json.dumps(value, ensure_ascii=False, sort_keys=True).encode('utf-8'), 6)


def unpack(blob):
    return json.loads(zlib.decompress(blob).decode('utf-8'))


class Deriver:
    def __init__(self, store):
        self.store, self.db = store, store.db
        if not store.readonly:
            self.db.executescript(SCHEMA)
        self._specs = {}

    # ---- derivations ----
    @staticmethod
    def current_id(skey, policy=None):
        """Deterministic id of the derivation (spec, CURRENT validator, policy); nothing is written."""
        policy = dict(policy or {})
        extra = sorted(set(policy) - set(VALIDATOR_POLICY_KEYS))
        if extra:
            raise ValueError('Not validator policy (wire format, schema and prompt belong to the spec): ' + ', '.join(extra))
        if 'min_core_claims' in policy and (type(policy['min_core_claims']) is not int or policy['min_core_claims'] not in (0, 1)):
            raise ValueError('min_core_claims must be 0 or 1')
        return 'deriv-' + digest([skey, components()['validator']['fingerprint'], policy])[:16]

    def ensure(self, skey, policy=None):
        """Derivation id for (spec, CURRENT validator, policy); created when new."""
        policy = dict(policy or {})
        did = self.current_id(skey, policy)
        validator = components()['validator']
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO derivations VALUES (?,?,?,?,?,?)',
                            (did, skey, validator['version'], validator['fingerprint'], json.dumps(policy, sort_keys=True), now()))
        return did

    def exists(self, did):
        try:
            return self.db.execute('SELECT 1 FROM derivations WHERE derivation_id=?', (did,)).fetchone() is not None
        except Exception:      # a store that has never derived anything has no derivation tables yet (read-only open)
            return False

    def derivation(self, did):
        if not self.exists(did):
            raise KeyError('No derivation ' + did + ' (the validator or the policy differs from what was derived so far: run `derive` to catch up; no model is called)')
        row = self.db.execute('SELECT * FROM derivations WHERE derivation_id=?', (did,)).fetchone()
        return dict(row, policy=json.loads(row['policy_json']))

    def list(self, skey=None):
        if not self.exists_any():
            return []
        rows = self.db.execute('SELECT d.*, (SELECT count(*) FROM derived_papers p WHERE p.derivation_id = d.derivation_id) AS answers_derived FROM derivations d '
                               + ('WHERE d.spec_key = ? ' if skey else '') + 'ORDER BY d.created_at', ([skey] if skey else [])).fetchall()
        return [dict(r, policy=json.loads(r['policy_json'])) for r in rows]

    def drop(self, did):
        """Forget a derivation (its rows are recomputable from the stored answers at any time). Refused while a release pins it."""
        self.derivation(did)
        if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='releases'").fetchone():
            pinned = [r[0] for r in self.db.execute('SELECT name FROM releases WHERE derivation_id=?', (did,))]
            if pinned:
                raise ValueError('Derivation is pinned by release(s): ' + ', '.join(pinned))
        with self.db:
            n = self.db.execute('DELETE FROM derived_claims WHERE derivation_id=?', (did,)).rowcount
            self.db.execute('DELETE FROM derived_papers WHERE derivation_id=?', (did,))
            self.db.execute('DELETE FROM derivations WHERE derivation_id=?', (did,))
        return n

    def exists_any(self):
        return self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='derivations'").fetchone() is not None

    def _registry(self, did):
        d = self.derivation(did)
        if d['spec_key'] not in self._specs:
            self._specs[d['spec_key']] = self.store.get_spec(d['spec_key'])
        return dict(self._specs[d['spec_key']]['registry'], **d['policy'])

    def _validate(self, did, rkey):
        answer = self.store.get_answer(rkey)
        paper = self.store.get_input(answer['input_key'])
        # request_key takes the place the run id had: a claim's identity no longer depends on which run asked
        return answer, paper, validate_paper(answer['body']['result'], paper, self._registry(did), rkey)

    # ---- incremental derivation ----
    def derive(self, did, request_keys, chunk=500):
        """Validate these answers under the derivation; answers already derived are skipped. Returns how many were new."""
        if self.derivation(did)['validator_fingerprint'] != components()['validator']['fingerprint']:
            raise ValueError('This derivation belongs to another validator version; call ensure() for a derivation of the current one')
        done, todo = 0, []
        for k in request_keys:
            if not self.db.execute('SELECT 1 FROM derived_papers WHERE derivation_id=? AND request_key=?', (did, k)).fetchone():
                loc = self.db.execute('SELECT shard, offset, line_no FROM answers WHERE request_key=?', (k,)).fetchone()
                if loc is None:
                    raise KeyError('No answer ' + k)
                todo.append((tuple(loc), k))
        keys = [k for _, k in sorted(todo)]     # storage order: one member is decompressed once
        for start in range(0, len(keys), chunk):
            with self.db:
                for rkey in keys[start:start + chunk]:
                    answer, paper, v = self._validate(did, rkey)
                    self._write(did, answer, paper, v)
                    done += 1
        return done

    def derive_missing(self, did):
        skey = self.derivation(did)['spec_key']
        keys = [r[0] for r in self.db.execute('SELECT a.request_key FROM answers a LEFT JOIN derived_papers d ON d.request_key = a.request_key AND d.derivation_id = ? '
                                              'WHERE a.spec_key = ? AND d.request_key IS NULL', (did, skey)).fetchall()]
        return self.derive(did, keys)

    def _write(self, did, answer, paper, v):
        rel = v['relevance'] or {}
        subject = rel.get('plant_subject')
        if not (subject is None or isinstance(subject, str)):
            subject = json.dumps(subject, ensure_ascii=False)
        self.db.execute('INSERT INTO derived_papers VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (did, answer['request_key'], paper['paper_id'], answer['input_key'], answer['channel'], answer['replicate'], answer['attempt'], answer['answer_seq'], v['status'], rel.get('label'), rel.get('reason_code'), subject,
                         json.dumps(rel.get('buckets', [])), json.dumps(rel.get('input_issues', [])), v['claims_proposed'], len(v['claims_accepted']),
                         len(v['claims_review']), len(v['claims_quarantined']), json.dumps(v['errors']), v.get('error')))
        raw = answer['body']['result'].get('claims') if isinstance(answer['body']['result'], dict) else None
        raw = raw if isinstance(raw, list) else []
        first = {}
        for pos, c in enumerate(raw):
            if isinstance(c, dict):
                first.setdefault(c.get('id'), pos)
        legacy_run = answer['meta'].get('imported_from_run')

        def legacy(pos):
            return ('rev-' + digest([legacy_run, paper['paper_id'], paper['text_sha256'], raw[pos]])) if legacy_run and pos is not None and pos < len(raw) else None
        for disposition, items in (('accepted', v['claims_accepted']), ('review', v['claims_review'])):
            for c in items:
                pos = first.get(c['local_id'])
                self.db.execute('INSERT INTO derived_claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                (did, answer['request_key'], pos, paper['paper_id'], answer['attempt'], c['local_id'], c['revision_id'], legacy(pos), disposition,
                                 c['kind'], c['epistemic_status'], c['origin'], c['polarity'], c['statistical_conclusion'], c.get('basis'), json.dumps(c['buckets']),
                                 c['quote_match'], int(bool(c.get('direction_inferred'))), c.get('needs_schema_review'), '[]', pack(c)))
        for q in v['claims_quarantined']:
            c = q['raw'] if isinstance(q['raw'], dict) else {}
            self.db.execute('INSERT INTO derived_claims VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (did, answer['request_key'], q['response_position'], paper['paper_id'], answer['attempt'], q['local_id'], None, legacy(q['response_position']),
                             'quarantined', c.get('k'), c.get('es'), c.get('o'), c.get('pol'), c.get('stat'), c.get('basis'),
                             json.dumps(c.get('bk') if isinstance(c.get('bk'), list) else []), None, None, None, json.dumps(q['errors']),
                             pack(dict(raw=q['raw'], evidence_status=q.get('evidence_status') or {}))))

    # ---- statistics: SQL over the answer that counts (highest attempt) or over attempt 1 ----
    def _scope(self, did, job_id, first_pass, table, upto=None):
        """FROM/WHERE clause selecting the rows of `table` that belong to the counting answers. A question series is
        (input, channel, replicate 0); a job narrows it to its own inputs and its own channel, so an answer that was
        cached from another job still counts. `upto` (an
        answer_seq watermark) ignores every answer stored later: that is what makes a release reproducible."""
        args, join, channel = [], '', None
        if job_id:
            channel = self.store.job(job_id)['channel']
            join = 'JOIN job_papers j ON j.job_id = ? AND j.input_key = p.input_key '
            args.append(job_id)
        sub = 'SELECT p.request_key FROM derived_papers p ' + join + 'WHERE p.derivation_id = ? AND p.replicate = 0 '
        args.append(did)
        if channel:
            sub += 'AND p.channel = ? '
            args.append(channel)
        if upto is not None:
            sub += 'AND p.answer_seq <= ? '
            args.append(upto)
        sub += ('AND p.attempt = 1' if first_pass else
                'AND p.attempt = (SELECT max(e.attempt) FROM derived_papers e WHERE e.derivation_id = p.derivation_id AND e.input_key = p.input_key '
                'AND e.channel = p.channel AND e.replicate = 0' + (' AND e.answer_seq <= ?' if upto is not None else '') + ')')
        if upto is not None and not first_pass:
            args.append(upto)
        return f'FROM {table} t WHERE t.derivation_id = ? AND t.request_key IN ({sub})', [did] + args

    def behind(self, did):
        """Answers of the derivation's spec that it has not validated yet (a fresh derivation after a validator change starts at all of them)."""
        skey = self.derivation(did)['spec_key']
        return (self.db.execute('SELECT count(*) FROM answers WHERE spec_key=?', (skey,)).fetchone()[0]
                - self.db.execute('SELECT count(*) FROM derived_papers WHERE derivation_id=?', (did,)).fetchone()[0])

    def stats(self, did, job_id=None, first_pass=False, upto=None):
        scope, args = self._scope(did, job_id, first_pass, 'derived_papers', upto)
        row = self.db.execute("SELECT count(*), sum(status='assessed'), sum(status='execution_failed'), sum(status='quarantined'), sum(claims_proposed), "
                              "sum(claims_accepted), sum(claims_review), sum(claims_quarantined), sum(label='core' AND claims_accepted > 0), "
                              "sum(label='core' AND claims_proposed = 0) " + scope, args).fetchone()
        labels = dict(self.db.execute("SELECT t.label, count(*) " + scope + " AND t.status = 'assessed' GROUP BY t.label", args).fetchall())
        n = [x or 0 for x in row]
        total = self.db.execute('SELECT n_papers FROM jobs WHERE job_id=?', (job_id,)).fetchone()[0] if job_id else n[0]
        return dict(papers=total, records=n[0], pending=total - n[0], assessed=n[1], execution_failed=n[2], quarantined_papers=n[3], labels=labels,
                    core_with_claims=n[8], core_zero_claims=n[9], claims_proposed=n[4], claims_accepted=n[5], claims_review=n[6], claims_quarantined=n[7])

    def claim_ids_sha256(self, did, job_id=None, upto=None):
        scope, args = self._scope(did, job_id, False, 'derived_claims', upto)
        ids = [r[0] for r in self.db.execute("SELECT t.revision_id " + scope + " AND t.disposition = 'accepted' ORDER BY t.revision_id", args)]
        return digest(ids), len(ids)

    def report(self, did, job_id=None):
        d = self.derivation(did)
        final, first = self.stats(did, job_id), self.stats(did, job_id, first_pass=True)
        scope, args = self._scope(did, job_id, False, 'derived_claims')
        by = {k: Counter() for k in ('kind', 'epistemic_status', 'origin', 'polarity', 'statistical_conclusion', 'basis', 'bucket', 'quote_match')}
        dir_inf = rev = 0
        literal, quarantine = Counter(), Counter()
        for r in self.db.execute('SELECT t.disposition, t.kind, t.epistemic_status, t.origin, t.polarity, t.statistical_conclusion, t.basis, t.buckets_json, t.quote_match, '
                                 't.direction_inferred, t.needs_schema_review, t.errors_json, t.data_z ' + scope, args):
            if r['disposition'] == 'quarantined':
                quarantine.update(e.split(':')[0] for e in json.loads(r['errors_json'])); continue
            if r['disposition'] != 'accepted':
                continue
            for k in ('kind', 'epistemic_status', 'origin', 'polarity', 'statistical_conclusion', 'quote_match'):
                by[k][r[k]] += 1
            if r['basis']:
                by['basis'][r['basis']] += 1
            by['bucket'].update(json.loads(r['buckets_json']))
            dir_inf += r['direction_inferred'] or 0
            rev += 1 if r['needs_schema_review'] else 0
            literal.update(f['literal_match'] for f in unpack(r['data_z'])['fields'].values() if 'literal_match' in f)
        pscope, pargs = self._scope(did, job_id, False, 'derived_papers')
        issues, paper_errors = Counter(), Counter()
        for r in self.db.execute('SELECT t.input_issues_json, t.errors_json, t.status ' + pscope, pargs):
            issues.update(json.loads(r['input_issues_json']))
            if r['status'] == 'quarantined':
                paper_errors.update(json.loads(r['errors_json']))
        retried = self.db.execute('SELECT count(DISTINCT t.paper_id) ' + self._scope(did, job_id, False, 'derived_papers')[0] + ' AND t.attempt > 1', pargs).fetchone()[0]
        sha, n = self.claim_ids_sha256(did, job_id)
        return dict(derivation=dict(derivation_id=did, validator_version=d['validator_version'], validator_fingerprint=d['validator_fingerprint'], policy=d['policy']),
                    job_id=job_id, stats=final, first_pass_stats=first, attempts=dict(papers_retried=retried, first_pass={k: first[k] for k in COUNT_KEYS},
                                                                                        final={k: final[k] for k in COUNT_KEYS}),
                    claims_by={k: dict(v) for k, v in by.items() if v}, direction_inferred=dir_inf, needs_schema_review=rev, field_literal_match=dict(literal),
                    input_issues=dict(issues), quarantine_reasons=dict(paper=dict(paper_errors), claim=dict(quarantine)),
                    claim_ids_sha256=sha, claims_in_hash=n, role='development_candidates_not_gold', human_reviewed=0, semantic_accuracy=None)

    def paper_rows(self, did, request_keys):
        """What the validator made of these answers, per paper."""
        out = []
        for k in request_keys:
            r = self.db.execute('SELECT paper_id, attempt, status, label, claims_proposed, claims_accepted, claims_review, claims_quarantined, errors_json '
                                'FROM derived_papers WHERE derivation_id=? AND request_key=?', (did, k)).fetchone()
            claim_errors = [dict(local_id=c['local_id'], errors=json.loads(c['errors_json'])) for c in self.db.execute(
                "SELECT local_id, errors_json FROM derived_claims WHERE derivation_id=? AND request_key=? AND disposition='quarantined' ORDER BY position", (did, k))]
            out.append(dict(paper_id=r['paper_id'], attempt=r['attempt'], status=r['status'], label=r['label'], claims_proposed=r['claims_proposed'],
                            claims_accepted=r['claims_accepted'], claims_review=r['claims_review'], claims_quarantined=r['claims_quarantined'],
                            errors=json.loads(r['errors_json']), **({'claim_errors': claim_errors} if claim_errors else {})))
        return sorted(out, key=lambda x: (x['paper_id'], x['attempt']))

    # ---- validator feedback for a retry ----
    def feedback_for(self, did, rkey):
        return paper_feedback(self._validate(did, rkey)[2])

    def feedback(self, did, job_id, max_attempts):
        self.derivation(did)      # an absent derivation (validator changed since the last ingest) is an error here, not an empty list
        scope, args = self._scope(did, job_id, False, 'derived_papers')
        retry, exhausted = [], []
        for r in self.db.execute("SELECT t.request_key, t.paper_id, t.attempt " + scope + " AND (t.status = 'quarantined' OR t.claims_quarantined > 0 OR t.claims_review > 0) "
                                 "ORDER BY t.paper_id", args).fetchall():
            answer, paper, v = self._validate(did, r['request_key'])
            items = paper_feedback(v)
            if not items:
                continue
            entry = dict(paper_id=r['paper_id'], attempt=r['attempt'], next_attempt=r['attempt'] + 1, retry_of=r['request_key'], feedback=items)
            (retry if r['attempt'] < max_attempts else exhausted).append(entry)
        return dict(max_attempts=max_attempts, retry=retry, exhausted=exhausted)


__all__ = ['ACCEPTED_MATCH', 'COUNT_KEYS', 'Deriver', 'SCHEMA', 'pack', 'unpack']

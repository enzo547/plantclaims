"""Helpers shared by the annotation layers: paper ids as used in layer keys, and the check that a layer's derivation has
validated every answer it reads."""
import re

EXPORTABLE_POLICY_KEYS = {'bucket_rules'}


def site_id(paper_id):
    wid = paper_id.rsplit('/', 1)[-1]
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', wid):
        raise ValueError('Paper id is not URL-safe: ' + paper_id)
    return wid


def job_behind(deriver, did, job_id):
    """Answers to the job's own inputs under the derivation's spec that the derivation has not validated yet. A layer reads
    one job, so a chunk still being answered under the same spec does not hold back a finished chunk; every answer the layer
    can read must be validated."""
    skey = deriver.derivation(did)['spec_key']
    return deriver.db.execute('SELECT count(*) FROM job_papers j JOIN answers a ON a.input_key = j.input_key AND a.spec_key = ? WHERE j.job_id = ? '
                              'AND NOT EXISTS (SELECT 1 FROM derived_papers p WHERE p.derivation_id = ? AND p.request_key = a.request_key)',
                              (skey, job_id, did)).fetchone()[0]

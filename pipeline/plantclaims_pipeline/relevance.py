"""Provenance checks for recorded answers and write-once file helpers."""
from datetime import datetime
from pathlib import Path


ASSESSED_KEYS = {'paper_id', 'text_sha256', 'status', 'label', 'reason_code', 'rationale',
                 'plant_subject', 'research_question', 'evidence', 'buckets', 'missing_reason', 'next_action'}
MODEL_KEYS = {'schema_version', 'input_sha256', 'batch_id', 'method', 'model_provenance', 'assessments'}
REVIEW_KEYS = {'schema_version', 'input_sha256', 'method', 'reviewer_id', 'reviewed_at',
               'blind_review_declared', 'assessments'}
API_PROVENANCE_KEYS = {'provider', 'model_id', 'request_id', 'input_tokens', 'output_tokens', 'cost_usd',
                       'price_checked_on', 'effort', 'stop_reason', 'accounting_status', 'human_review', 'raw_sha256'}
API_ACCOUNTING = {'api_reported': True, 'api_reported_unpriced': False}  # value: cost_usd must be a number


def validate_api_provenance(p):
    """Actual service-reported provenance only; every field copied, nothing estimated."""
    if not isinstance(p, dict) or set(p) != API_PROVENANCE_KEYS:
        raise ValueError('model_api_provenance_keys')
    for key in ('provider', 'model_id', 'request_id', 'effort', 'stop_reason'):
        if not nonempty(p[key]):
            raise ValueError('model_api_provenance_missing_' + key)
    for key in ('input_tokens', 'output_tokens'):
        if not isinstance(p[key], int) or isinstance(p[key], bool) or p[key] < 0:
            raise ValueError('model_api_provenance_invalid_' + key)
    if p['human_review'] != 'not_performed':
        raise ValueError('model_api_must_not_claim_human_review')
    if p['accounting_status'] not in API_ACCOUNTING:
        raise ValueError('model_api_invalid_accounting_status')
    priced, cost = API_ACCOUNTING[p['accounting_status']], p['cost_usd']
    if priced and (not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost < 0):
        raise ValueError('model_api_priced_cost_required')
    if not priced and cost is not None:
        raise ValueError('model_api_unpriced_must_have_null_cost')
    try:
        datetime.fromisoformat(str(p['price_checked_on']))
    except (ValueError, TypeError):
        raise ValueError('model_api_price_checked_on_must_be_iso_date')
    if not isinstance(p['raw_sha256'], str) or len(p['raw_sha256']) != 64 or any(c not in '0123456789abcdef' for c in p['raw_sha256']):
        raise ValueError('model_api_raw_sha256_required')
    return p


def write_once(path, data):
    """Keep exact original bytes, including malformed model output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError('Immutable artifact differs')
    else:
        with path.open('xb') as stream:
            stream.write(data)


def nonempty(value):
    return isinstance(value, str) and bool(value.strip())



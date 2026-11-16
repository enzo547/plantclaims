"""OpenAlex record helpers: DOI normalisation, abstract reconstruction from the inverted index, and the record fields the pipeline keeps."""

import re
from urllib.parse import unquote


def normalize_doi(value):
    if not value:
        return None
    value = unquote(str(value).strip())
    value = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', value, flags=re.I)
    value = re.sub(r'^doi:\s*', '', value, flags=re.I)
    return value.lower()  # do not strip punctuation from DOI suffixes


def reconstruct_abstract(index):
    if index is None:
        return None
    if not isinstance(index, dict):
        raise ValueError('abstract_inverted_index must be object or null')
    tokens = {}
    for word, positions in index.items():
        for pos in positions:
            if not isinstance(pos, int) or pos < 0 or pos in tokens:
                raise ValueError('Invalid or overlapping abstract token positions')
            tokens[pos] = word
    if not tokens:
        return ''
    if set(tokens) != set(range(max(tokens) + 1)):
        raise ValueError('Abstract token gaps; keep original index for review')
    return ' '.join(tokens[i] for i in range(len(tokens)))


def oa_record(work):
    try:
        abstract = reconstruct_abstract(work.get('abstract_inverted_index'))
        abstract_status = 'reconstructed' if abstract else 'missing'
    except ValueError:
        abstract, abstract_status = None, 'invalid_index'
    return {'provider': 'openalex', 'native_id': work['id'], 'doi': normalize_doi(work.get('doi')),
            'title': work.get('title'), 'abstract': abstract, 'abstract_status': abstract_status,
            'year': work.get('publication_year'), 'publication_date': work.get('publication_date'),
            'language': work.get('language'), 'type': work.get('type'),
            'citations': work.get('cited_by_count'), 'reported_oa': work.get('open_access'),
            'locations': work.get('locations', []), 'primary_topic': work.get('primary_topic'),
            'topics': work.get('topics', []), 'relevance_decision': 'not_assessed',
            'dedup_status': 'not_assessed', 'role': 'p0_fixture'}



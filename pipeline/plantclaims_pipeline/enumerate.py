"""OpenAlex index helpers shared by the snapshot filter and the paper selection (subfield ids, index rows)."""

import json

from .adapters import normalize_doi, oa_record
from .records import digest

OA = 'https://api.openalex.org/works'
SCHEMA = '''
CREATE TABLE IF NOT EXISTS enumeration_records (
  run_id TEXT NOT NULL REFERENCES runs(run_id), openalex_id TEXT NOT NULL,
  doi TEXT, pmid TEXT, pmcid TEXT, publication_year INTEGER, type TEXT, language TEXT,
  has_abstract INTEGER NOT NULL, abstract_status TEXT NOT NULL, oa_status TEXT, is_oa INTEGER,
  primary_subfield_id TEXT, core_subfield_ids TEXT NOT NULL, is_retracted INTEGER,
  updated_date TEXT, raw_path TEXT NOT NULL, json_pointer TEXT NOT NULL, source_row_sha256 TEXT NOT NULL,
  PRIMARY KEY (run_id, openalex_id));
'''


def subfield_id(node):
    value = (node or {}).get('id') or ''
    return value.rsplit('/', 1)[-1] or None


def index_row(run_id, work, raw_path, pointer, core_ids):
    ids = work.get('ids') or {}
    rec = oa_record(work)  # abstract text is computed only to classify its status; never stored here
    topics = work.get('topics') or []
    core_hits = sorted({subfield_id(t.get('subfield')) for t in topics if subfield_id(t.get('subfield')) in core_ids})
    oa = work.get('open_access') or {}
    return (run_id, work['id'], normalize_doi(work.get('doi')),
            str(ids.get('pmid') or '').rstrip('/').rsplit('/', 1)[-1] or None,
            str(ids.get('pmcid') or '').rstrip('/').rsplit('/', 1)[-1] or None,
            work.get('publication_year'), work.get('type'), work.get('language'),
            int(bool(work.get('abstract_inverted_index'))), rec['abstract_status'],
            oa.get('oa_status'), None if oa.get('is_oa') is None else int(bool(oa.get('is_oa'))),
            subfield_id((work.get('primary_topic') or {}).get('subfield')), json.dumps(core_hits),
            None if work.get('is_retracted') is None else int(bool(work.get('is_retracted'))),
            work.get('updated_date'), raw_path, pointer, digest(work))



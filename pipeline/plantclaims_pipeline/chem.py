"""Compound normalisation against ChEBI: fetch the ChEBI flat files once, build an offline SQLite index, and look up names
exactly after normalisation. Used by the gene matcher for parts its gene rules leave untyped. No model call."""
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

from .records import digest, now, save_json
from .taxonomy import _download


def load_config(root):
    import yaml
    return yaml.safe_load((Path(root) / 'configs' / 'chem-v0.yaml').read_text(encoding='utf-8'))


def fetch(run):
    """Directory listings and files named in the run's choice (validated against configs/chem-v0.yaml), each one recorded GET."""
    if run.replay or run.readonly:
        raise ValueError('fetch needs a writable, non-replay run')
    if run.manifest.get('chebi'):
        raise ValueError('This run already holds ChEBI files; fetch once per run')
    src = run.config['chem']['source']
    chosen = run.config.get('chem_fetch') or {}
    wanted, listings = list(chosen.get('files') or src['files']), list(chosen.get('listings') if chosen.get('listings') is not None else src['listings'])
    if set(wanted) - set(src.get('available_files') or []) or set(listings) - set(src.get('listings') or []):
        raise ValueError('Files and listings must be named in configs/chem-v0.yaml (source.available_files / source.listings)')
    run.manifest['status'] = 'fetching'
    try:
        files, probed = {}, {}
        for rel in listings:
            path, attempt = _download(run, 'listing:' + rel, src['base_url'] + rel, src, allowed_host=src['host'], agent='chem')
            text = path.read_text(encoding='utf-8', errors='replace')
            plain = re.sub(r'<[^>]+>', ' ', text).replace('&nbsp;', ' ')      # Apache index pages: name, date, size once the tags are gone
            entries = re.findall(r'^\s*([^\s][^\n]*?)\s+(\d{4}-\d\d-\d\d \d\d:\d\d)\s+(\S+)', plain, re.M)
            probed[rel] = dict(raw=path.name, bytes=attempt['bytes'], sha256=attempt['sha256'], entries=[dict(file=f, modified=m, size_as_listed=z) for f, m, z in entries])
        for rel in wanted:
            if '..' in rel.split('/') or rel.startswith('/'):
                raise ValueError('Source files must be relative paths under base_url')
            path, attempt = _download(run, rel, src['base_url'] + rel, src, allowed_host=src['host'], agent='chem')
            files[rel] = dict(raw=path.name, bytes=attempt['bytes'], sha256=attempt['sha256'], retrieved_at=attempt['finished_at'])
        run.manifest['chebi'] = dict(source=src['name'], base_url=src['base_url'], licence_as_reported=src['licence_as_reported'], files=files, listings=probed)
        run.finish('fetched')
        return dict(run_id=run.id, files=files, listings=probed)
    except Exception as exc:
        run.manifest['failure'] = f'{type(exc).__name__}: {exc}'[:300]
        run.finish('failed')
        raise


# ---- index -----------------------------------------------------------------------------------------------------------

INDEX_VERSION = 'chem-index-v0.2'                      # v0.2: a class is an entity without a molecular formula (chemical_data), not one with members
GREEK = {'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon', 'κ': 'kappa', 'λ': 'lambda', 'ω': 'omega', 'ρ': 'rho', 'σ': 'sigma'}
HYPHENS = dict.fromkeys(map(ord, '‐‑‒–—―−'), '-')


def ckey(text):
    """Lookup key of a chemical name: tags out, NFKC, Greek letters spelled out, case-folded, spaces and hyphens removed
    ("gallic acid" = "gallic-acid", "β-myrcene" = "beta-myrcene"). The same function builds the index and reads the claims."""
    t = re.sub(r'<[^>]+>', '', unicodedata.normalize('NFKC', text or '')).translate(HYPHENS)
    t = ''.join(GREEK.get(ch, GREEK.get(ch.lower(), ch)) for ch in t)
    return re.sub(r'[\s\-]+', '', t.casefold())


def index_path(root, fetch_run_id):
    return Path(root) / 'data' / 'chem' / fetch_run_id / 'index.sqlite'


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _tsv(path):
    csv.field_size_limit(1 << 30)
    with gzip.open(path, 'rt', encoding='utf-8', errors='replace', newline='') as f:
        yield from csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)


def build_index(root, fetch_run_id, force=False):
    """ChEBI flat files → SQLite: entities (id, name, stars, status, class flag) and names (key, id, class priority). Idempotent."""
    root = Path(root)
    manifest = json.loads((root / 'data' / 'runs' / fetch_run_id / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind') != 'chem_fetch' or manifest.get('status') != 'fetched' or 'flat_files/compounds.tsv.gz' not in manifest['chebi']['files']:
        raise ValueError('Not a completed chem_fetch run with the flat files: ' + fetch_run_id)
    cfg = load_config(root)['index']                      # a build parameter (recorded in the index identity), not part of the fetch
    raw = root / 'data' / 'raw' / fetch_run_id
    files = manifest['chebi']['files']
    for rel, info in files.items():
        if digest((raw / info['raw']).read_bytes()) != info['sha256']:
            raise ValueError(f'{rel} differs from the fetch manifest; the raw file was modified')
    release = re.search(r'ChEBI Release:\s*(\d+)', (raw / 'README').read_text(encoding='utf-8', errors='replace'))
    path = index_path(root, fetch_run_id)
    meta_path = path.with_name('index.json')
    identity = dict(index_version=INDEX_VERSION, fetch_run=fetch_run_id, files={k: v['sha256'] for k, v in sorted(files.items())}, index_config=cfg)
    if path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if {k: meta.get(k) for k in identity} == identity and _file_sha(path) == meta['index_sha256']:
            return dict(path=str(path), existing=True, counts=meta['counts'])
    statuses = {r['id'] for r in _tsv(raw / 'status.tsv.gz') if r['name'] in cfg['statuses']}
    rel_type = {r['code']: r['id'] for r in _tsv(raw / 'relation_type.tsv.gz')}[cfg['class_relation']]
    entities, counts = {}, Counter()
    for r in _tsv(raw / 'compounds.tsv.gz'):
        counts['compounds_rows'] += 1
        if r['status_id'] in statuses and not r['parent_id']:
            entities[int(r['id'])] = dict(name=r['ascii_name'] or r['name'], stars=int(r['stars'] or 0))
    parents = set()
    for r in _tsv(raw / 'relation.tsv.gz'):
        if r['relation_type_id'] == rel_type and r['status_id'] in statuses and int(r['init_id']) in entities:
            parents.add(int(r['final_id']))
    if cfg.get('class_rule') != 'no_formula' or 'flat_files/chemical_data.tsv.gz' not in files:
        raise ValueError('chem-index-v0.2 needs class_rule no_formula and chemical_data.tsv.gz in the fetch run')
    with_formula = {int(r['compound_id']) for r in _tsv(raw / 'chemical_data.tsv.gz') if (r['formula'] or '').strip()}
    classes = {i for i in entities if i not in with_formula}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('index.sqlite.tmp')
    if temp.exists():
        temp.unlink()
    db = sqlite3.connect(temp)
    db.executescript('''
        CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT NOT NULL, stars INTEGER, is_class INTEGER NOT NULL, has_members INTEGER NOT NULL);
        CREATE TABLE names (key TEXT NOT NULL, id INTEGER NOT NULL, cls INTEGER NOT NULL, name TEXT NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    types = list(cfg['name_types'])
    with db:
        db.executemany('INSERT INTO entities VALUES (?,?,?,?,?)', ((i, e['name'], e['stars'], int(i in classes), int(i in parents)) for i, e in sorted(entities.items())))
        seen = set()

        def rows():
            for i, e in entities.items():
                k = ckey(e['name'])
                if k and (k, i) not in seen:
                    seen.add((k, i)); counts['names:NAME'] += 1
                    yield k, i, types.index('NAME'), e['name']
            for r in _tsv(raw / 'names.tsv.gz'):
                cid = int(r['compound_id'])
                if cid in entities and r['type'] in types and r['status_id'] in statuses:
                    name = r['ascii_name'] or r['name']
                    k = ckey(name)
                    if k and (k, cid) not in seen:
                        seen.add((k, cid)); counts['names:' + r['type']] += 1
                        yield k, cid, types.index(r['type']), name
        db.executemany('INSERT INTO names VALUES (?,?,?,?)', rows())
        db.execute('CREATE INDEX names_key ON names(key)')
        counts.update(entities=len(entities), classes=len(classes), with_members=sum(1 for i in entities if i in parents))
        meta = dict(identity, release=release.group(1) if release else None, name_type_order=types, counts=dict(sorted(counts.items())), built_at=now())
        db.executemany('INSERT INTO meta VALUES (?,?)', ((k, json.dumps(v, sort_keys=True)) for k, v in meta.items()))
    db.close()
    temp.replace(path)
    meta['index_sha256'] = _file_sha(path)
    save_json(meta_path, meta)
    return dict(path=str(path), existing=False, counts=meta['counts'], release=meta['release'])


class ChemIndex:
    """Read-only view of a built ChEBI index, with the lookup the gene matcher uses for parts its gene rules leave untyped."""

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise ValueError('No ChEBI index at ' + str(path) + ' (run `chem build` first)')
        self.db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.meta = {r['key']: json.loads(r['value']) for r in self.db.execute('SELECT * FROM meta')}
        self.types = list(self.meta['name_type_order'])
        self.min_synonym_key = 4

    def close(self):
        self.db.close()

    def lookup(self, text):
        """Entities whose name has the key of `text`, strongest name type first: [(id, type)] of that type only."""
        key = ckey(text)
        rows = self.db.execute('SELECT id, cls FROM names WHERE key=? ORDER BY cls, id', (key,)).fetchall()
        if len(key) < self.min_synonym_key:           # a short abbreviation ("Pi-b", "Q") counts only as an entity's own ChEBI name
            rows = [r for r in rows if self.types[r['cls']] == 'NAME']
        if not rows:
            return []
        best = rows[0]['cls']
        return sorted({(r['id'], self.types[r['cls']]) for r in rows if r['cls'] == best})

    def entity(self, cid):
        r = self.db.execute('SELECT * FROM entities WHERE id=?', (cid,)).fetchone()
        return dict(id=r['id'], name=r['name'], stars=r['stars'], is_class=bool(r['is_class'])) if r else None

    def place(self, text):
        """One untyped part → None, or {status resolved|ambiguous, ids, type metabolite|compound_class, via, name}. The written form is
        tried first, then — for a plural — the singular (flavonoids → flavonoid, glycosides → glycoside)."""
        tries = [(text, None)]
        if re.search(r'[a-z]s$', text):
            tries.append((text[:-1], 'plural'))
            if text.endswith('es'):
                tries.append((text[:-2], 'plural'))
        for candidate, how in tries:
            hits = self.lookup(candidate)
            if hits:
                ids = sorted({i for i, _ in hits})
                via = ['chebi:' + hits[0][1].lower().replace(' ', '_')] + (['plural_to_singular'] if how else [])
                if len(ids) == 1:
                    e = self.entity(ids[0])
                    return dict(status='resolved', ids=ids, type='compound_class' if e['is_class'] else 'metabolite', via=via, name=e['name'], key=ckey(candidate))
                return dict(status='ambiguous', ids=ids, type='metabolite', via=via, name=candidate, key=ckey(candidate))
        return None

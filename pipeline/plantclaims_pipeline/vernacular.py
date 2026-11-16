"""Common-name lists for organism normalisation: fetch a list once and propose common-name mappings for spellings the
matcher leaves unresolved."""
import csv
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .records import digest
from .taxonomy import Taxonomy, _download, index_path, norm


def fetch(run, source_id):
    if run.replay or run.readonly:
        raise ValueError('fetch needs a writable, non-replay run')
    if run.manifest.get('vernacular_source'):
        raise ValueError('This run already holds a source; fetch once per run')
    src = run.config['vernacular']['sources'].get(source_id)
    if src is None:
        raise ValueError('Unknown source: ' + source_id)
    run.manifest['status'] = 'fetching'
    try:
        path, attempt = _download(run, 'archive', src['url'], src, allowed_host=src['host'], agent='vernacular')
        members, metadata = {}, {}
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                data = z.read(info)
                members[info.filename] = dict(bytes=info.file_size, sha256=hashlib.sha256(data).hexdigest())
                if Path(info.filename).name.lower() in ('metadata.yaml', 'metadata.yml', 'metadata.json', 'eml.xml'):
                    text = data.decode('utf-8', 'replace')
                    metadata[info.filename] = dict(licence_lines=[l.strip() for l in text.splitlines() if re.search(r'licen[cs]e|rights|cc0|creative', l, re.I)][:10])
        run.manifest['vernacular_source'] = dict(source=source_id, name=src['name'], url=src['url'], archive=path.name, archive_sha256=attempt['sha256'],
                                                 archive_bytes=attempt['bytes'], members=members, metadata=metadata, licence_as_reported=src['licence_as_reported'],
                                                 retrieved_at=attempt['finished_at'])
        run.finish('fetched')
        return dict(run_id=run.id, archive_bytes=attempt['bytes'], archive_sha256=attempt['sha256'], members=sorted(members), metadata=metadata)
    except Exception as exc:
        run.manifest['failure'] = f'{type(exc).__name__}: {exc}'[:300]
        run.finish('failed')
        raise


def open_archive(root, fetch_run_id):
    """The retained archive of a completed fetch run, checked against its manifest."""
    run_dir = Path(root) / 'data' / 'runs' / fetch_run_id
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind') != 'vernacular_fetch' or manifest.get('status') != 'fetched':
        raise ValueError('Not a completed vernacular_fetch run: ' + fetch_run_id)
    info = manifest['vernacular_source']
    path = Path(root) / 'data' / 'raw' / fetch_run_id / info['archive']
    if digest(path.read_bytes()) != info['archive_sha256']:
        raise ValueError('The retained archive differs from its fetch manifest')
    return zipfile.ZipFile(path), manifest


def compact(text):
    return re.sub(r'[\s\-‐‑]', '', norm(text))


def singulars(key):
    out = []
    for pattern, repl in ((r'ies$', 'y'), (r'(ch|sh|s|x|z)es$', r'\1'), (r'oes$', 'o'), (r's$', '')):
        v = re.sub(pattern, repl, key)
        if v != key and v not in out:
            out.append(v)
    return out


def read_tsv(z, name):
    return list(csv.DictReader(io.StringIO(z.read(name).decode('utf-8')), delimiter='\t', quoting=csv.QUOTE_NONE))


class Grin:
    """GRIN ColDP: English common names (comma-joined group strings split into names) → GRIN taxa, each placed on the NCBI backbone."""

    def __init__(self, z, tax):
        self.usage = {r['col:ID']: r for r in read_tsv(z, 'NameUsage.tsv')}
        self.by_key, self.by_compact, self.strings = defaultdict(set), defaultdict(set), defaultdict(set)
        for r in read_tsv(z, 'VernacularName.tsv'):
            if r['col:language'] != 'eng' or r['col:taxonID'] not in self.usage:
                continue
            for part in r['col:name'].split(','):
                if part.strip():
                    self.by_key[norm(part)].add(r['col:taxonID'])
                    self.by_compact[compact(part)].add(r['col:taxonID'])
                    self.strings[(norm(part), r['col:taxonID'])].add(r['col:name'])
        self.tax, self._placed = tax, {}

    def find(self, spelling):
        """(form used, GRIN taxon ids): the whole spelling, then singular forms, then the same ignoring spaces and hyphens."""
        key = norm(spelling)
        for form in [key] + singulars(key):
            if form in self.by_key:
                return form, sorted(self.by_key[form])
        for form in [key] + singulars(key):
            if compact(form) in self.by_compact:
                return 'compact:' + compact(form), sorted(self.by_compact[compact(form)])
        return None, []

    def describe(self, gid):
        u = self.usage[gid]
        name = u['col:scientificName']
        accepted = self.usage.get(u['col:parentID']) if u['col:status'] and u['col:status'] != 'accepted' else None
        return dict(grin_id=gid, name=name, rank=u['col:rank'], status=u['col:status'] or 'accepted', accepted=accepted['col:scientificName'] if accepted else None,
                    genus=name.split()[0], ncbi=self.place(name, accepted['col:scientificName'] if accepted else None))

    def place(self, name, accepted=None):
        """NCBI placement of a GRIN scientific name: exact name (scientific / synonym / equivalent), GRIN's accepted name, a
        "Genus spp." as the genus; homonyms prefer the green plant. None when the backbone has no such name."""
        if (name, accepted) in self._placed:
            return self._placed[(name, accepted)]
        out = None
        tries = [(name, 'exact'), (accepted, 'grin_accepted_name')] + ([(name.split()[0], 'genus_of_spp')] if re.search(r'\sspp?\.$', name) else [])
        for text, how in tries:
            if not text:
                continue
            ids = sorted({h['taxid'] for h in self.tax.lookup(text) if h['cls'] in PLACING_CLASSES})
            if len(ids) > 1:
                ids = [t for t in ids if self.tax.group(t) == 'plant'] or ids
            if len(ids) == 1:
                d = self.tax.describe(ids[0])
                out = dict(taxid=ids[0], name=d['name'], rank=d['rank'], group=d['group'], via=how)
                break
            if len(ids) > 1:
                out = dict(taxid=None, candidates=ids, via=how + ':homonym')
                break
        self._placed[(name, accepted)] = out
        return out


PLACING_CLASSES = ('scientific name', 'synonym', 'equivalent name')
SPECIES_RANKS = ('species',)
INFRA_RANKS = ('subspecies', 'variety', 'f.', 'group', 'race')


def classify(cands):
    """single — one GRIN taxon; nested — all in one genus (a genus entry and/or one species with its infraspecific taxa): a species
    option and a genus option; conflicting — several genera, or several species of one genus: no proposal."""
    genera = {c['genus'] for c in cands}
    if len(cands) == 1:
        return 'single', dict(species=cands[0] if cands[0]['rank'] in SPECIES_RANKS + INFRA_RANKS and not re.search(r'\sspp?\.$', cands[0]['name']) else None,
                              genus=cands[0] if cands[0]['rank'] == 'genus' or re.search(r'\sspp?\.$', cands[0]['name']) else None)
    if len(genera) != 1:
        return 'conflicting', {}
    species = {' '.join(c['name'].split()[:2]) for c in cands if c['rank'] in SPECIES_RANKS + INFRA_RANKS and not re.search(r'\sspp?\.$', c['name'])}
    if len(species) > 1:
        return 'conflicting', {}
    sp = [c for c in cands if c['rank'] in SPECIES_RANKS and ' '.join(c['name'].split()[:2]) in species] or [c for c in cands if c['rank'] in INFRA_RANKS]
    gen = [c for c in cands if c['rank'] == 'genus' or re.search(r'\sspp?\.$', c['name'])]
    return 'nested', dict(species=sp[0] if sp else None, genus=gen[0] if gen else None)


def ncbi_compact(tax):
    """NCBI's own common names keyed ignoring spaces and hyphens (46K names): what a spacing/plural rule inside the backbone would find."""
    out = defaultdict(set)
    for r in tax.db.execute("SELECT taxid, name FROM names WHERE class IN ('common name', 'genbank common name')"):
        out[compact(r['name'])].add((r['taxid'], r['name']))
    return out


COLUMNS = ['field', 'spelling', 'values', 'claims', 'program', 'grin_form', 'pattern', 'species_option', 'genus_option', 'grin_candidates', 'ncbi_spacing_or_plural',
           'contexts']


def candidates_run(run):
    """Review table: every leftover spelling of the bound match run with its GRIN candidates placed on NCBI, the species and genus
    options and what NCBI's own common names give when spaces, hyphens and plurals are ignored.
    Proposes; never applies."""
    from .store import Store
    from .taxonomy import fold, store_claims, store_layers
    if run.replay or run.readonly:
        raise ValueError('candidates needs a writable, non-replay run')
    bound = run.config['vernacular_candidates']
    z, fetch_manifest = open_archive(run.root, bound['fetch_run'])
    match_dir = run.root / 'data' / 'runs' / bound['match_run']
    match_manifest = json.loads((match_dir / 'manifest.json').read_text(encoding='utf-8'))
    if match_manifest.get('kind') != 'taxon_match' or match_manifest.get('status') != 'matched':
        raise ValueError('Not a completed taxon_match run: ' + bound['match_run'])
    data = (match_dir / 'matches.jsonl').read_bytes()
    if digest(data) != match_manifest['taxon_match']['outputs']['matches.jsonl']['sha256']:
        raise ValueError('matches.jsonl differs from its run manifest')
    rows = [json.loads(l) for l in data.decode('utf-8').splitlines() if l.strip()]
    tax = Taxonomy(index_path(run.root, bound['taxonomy_run']))
    try:
        grin, own = Grin(z, tax), ncbi_compact(tax)
        store = Store(run.root, readonly=True)
        try:
            layers = [dict(job=l['job'], derivation=l['derivation']) for l in match_manifest['taxon_match']['claims_source']['layers']]
            papers, _ = store_layers(store, layers)
            statement = {c['id']: c['statement'] for _, c in store_claims(store, layers, papers)}
        finally:
            store.close()
        groups = defaultdict(list)
        for r in rows:
            if r['status'] != 'resolved':
                groups[(r['field'], fold(r['clean']))].append(r)
        table, counts = [], Counter()
        for (field, _), group in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            r0 = group[0]
            form, ids = grin.find(r0['clean'])
            cands = [grin.describe(g) for g in ids]
            pattern, options = classify(cands) if cands else ('none', {})
            key = norm(r0['clean'])
            own_hits = next((sorted(own[compact(f)]) for f in [key] + singulars(key) + [key + 's', key + 'es'] if compact(f) in own), [])
            counts[pattern] += 1
            counts['values:' + pattern] += len(group)

            def show(c):
                if not c:
                    return ''
                n = c['ncbi'] or {}
                return f"{c['name']} [{c['rank']}] → " + (f"NCBI {n['taxid']} {n['name']} ({n['rank']})" if n.get('taxid') else 'NCBI: ' + (n.get('via') or 'not in backbone'))
            table.append(dict(field=field, spelling=r0['clean'], values=len(group), claims=len({g['claim'] for g in group}), program=r0['status'] + (':' + r0['reason'] if r0.get('reason') else ''),
                              grin_form=form or '', pattern=pattern, species_option=show(options.get('species')), genus_option=show(options.get('genus')),
                              grin_candidates='; '.join(show(c) for c in cands), ncbi_spacing_or_plural='; '.join(f'{t} {n}' for t, n in own_hits),
                              contexts=' | '.join(dict.fromkeys(statement.get(g['claim'], '') for g in group))[:400]))
    finally:
        tax.close()
        z.close()
    with (run.directory / 'candidates.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(table)
    summary = dict(spellings=len(table), values=sum(t['values'] for t in table), patterns={k: v for k, v in sorted(counts.items()) if ':' not in k},
                   values_by_pattern={k.split(':', 1)[1]: v for k, v in sorted(counts.items()) if ':' in k},
                   ncbi_spacing_or_plural=sum(1 for t in table if t['ncbi_spacing_or_plural']))
    run.artifact('summary.json', summary)
    run.manifest['vernacular_candidates'] = dict(fetch_run=bound['fetch_run'], archive_sha256=fetch_manifest['vernacular_source']['archive_sha256'],
                                                 taxonomy_run=bound['taxonomy_run'], index_sha256=json.loads(index_path(run.root, bound['taxonomy_run']).with_name('index.json').read_text(encoding='utf-8'))['index_sha256'],
                                                 match_run=bound['match_run'], matches_sha256=digest(data), summary=summary,
                                                 outputs={n: dict(sha256=digest((run.directory / n).read_bytes())) for n in ('candidates.csv', 'summary.json')})
    run.finish('proposed')
    return dict(run_id=run.id, **summary, directory=str(run.directory))


__all__ = ['candidates_run', 'fetch', 'open_archive']

"""Gene normalisation against NCBI Gene: fetch gene_info, build an offline index, and place the literal gene values of the
claims under the organisms named in the same claims. No model call."""
import csv
import gzip
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .records import digest, now, save_json
from .taxonomy import Taxonomy, _download, clean_value, fold, index_path as taxonomy_index_path

INDEX_VERSION = 'gene-index-v0.1'


# ---- step 1: fetch ---------------------------------------------------------------------------------------------------
def fetch(run):
    """One recorded GET per configured file into data/raw/<run>/ (the gzip is kept as downloaded, never extracted in place)."""
    if run.replay or run.readonly:
        raise ValueError('fetch needs a writable, non-replay run')
    if run.manifest.get('gene_info'):
        raise ValueError('This run already holds gene_info files; fetch once per run')
    src = run.config['genes']['source']
    chosen = run.config.get('gene_fetch') or {}
    wanted, listings = list(chosen.get('files') or src['files']), list(chosen.get('listings') or [])
    allowed = set(src.get('available_files') or src['files'])
    if set(wanted) - allowed or set(listings) - set(src.get('listings') or []):
        raise ValueError('Files and listings must be named in configs/genes-v0.yaml (source.available_files / source.listings)')
    run.manifest['status'] = 'fetching'
    try:
        files, probed = {}, {}
        for rel in listings:                                # a directory listing only: file names and sizes, to size a download before making it
            path, attempt = _download(run, 'listing:' + rel, src['base_url'] + rel, src, agent='genes')
            text = path.read_text(encoding='utf-8', errors='replace')
            entries = re.findall(r'href="([^"?/][^"]*\.gene_info\.gz)"[^\n]*?\s(\d[\d.]*[KMG]?)\s*$', text, re.M)
            probed[rel] = dict(raw=path.name, bytes=attempt['bytes'], sha256=attempt['sha256'], entries=[dict(file=f, size_as_listed=size) for f, size in entries])
        for rel in wanted:
            if '..' in rel.split('/') or rel.startswith('/'):
                raise ValueError('Source files must be relative paths under base_url')
            path, attempt = _download(run, rel, src['base_url'] + rel, src, agent='genes')
            with gzip.open(path, 'rt', encoding='utf-8', errors='replace') as f:
                header = f.readline().rstrip('\n')
                rows = sum(1 for _ in f)
            files[rel] = dict(raw=path.name, bytes=attempt['bytes'], sha256=attempt['sha256'], header=header, rows=rows, retrieved_at=attempt['finished_at'])
        run.manifest['gene_info'] = dict(source=src['name'], base_url=src['base_url'], licence_as_reported=src['licence_as_reported'], files=files,
                                         **({'listings': probed} if probed else {}))
        run.finish('fetched')
        return dict(run_id=run.id, files=files, listings=probed)
    except Exception as exc:
        run.manifest['failure'] = f'{type(exc).__name__}: {exc}'[:300]
        run.finish('failed')
        raise


# ---- names -----------------------------------------------------------------------------------------------------------
HYPHENS = dict.fromkeys(map(ord, '‐‑‒–—―−'), '-')


def gkey(text, greek):
    """Lookup key of a gene name: NFKC, Greek letters spelled out (β → beta), case-folded, hyphens, underscores, dots, apostrophes and
    spaces removed ("PR-1" = "PR1", "β‐tubulin" = "beta-tubulin"). The same function builds the index and reads the claims."""
    t = unicodedata.normalize('NFKC', text or '').translate(HYPHENS)
    t = ''.join(greek.get(ch, greek.get(ch.lower(), ch)) for ch in t)
    return re.sub(r"[\s\-_.'’]+", '', t.casefold())


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def run_set(runs):
    """Fetch runs as one id: a list or a '+'-joined string → the '+'-joined string (run ids never hold '+')."""
    ids = runs.split('+') if isinstance(runs, str) else list(runs)
    if not ids or any(not r.startswith('genedump-') for r in ids) or len(set(ids)) != len(ids):
        raise ValueError('Gene fetch runs must be distinct genedump-… ids')
    return '+'.join(ids)


def index_path(root, fetch_runs):
    return Path(root) / 'data' / 'genes' / run_set(fetch_runs) / 'index.sqlite'


def _rank_ancestor(tax, taxid, rank):
    for t in tax.lineage(taxid):
        node = tax.node(t)
        if node and node['rank'] == rank:
            return t
    return None


def build_index(root, fetch_runs, taxonomy_run, force=False):
    """gene_info → SQLite: genes (id, taxid, species, genus, symbol, locus tag, description, type), names (lookup key, species,
    gene, class priority) and species (taxid, genus, name, genes, prefixes). Idempotent: an index built from the same files, the
    same taxonomy index and the same config is reused. Genes whose taxid the taxonomy index does not hold (after following merged
    taxids) are counted and left out."""
    root = Path(root)
    fetch_run_id = run_set(fetch_runs)
    files, cfg = {}, None
    for rid in fetch_run_id.split('+'):                  # several fetch runs (plants, then fungi and viruses) make one index
        manifest = json.loads((root / 'data' / 'runs' / rid / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('kind') != 'gene_fetch' or manifest.get('status') != 'fetched':
            raise ValueError('Not a completed gene_fetch run: ' + rid)
        run_cfg = manifest['config_snapshot']['genes']
        if cfg is not None and (run_cfg['index'], run_cfg['match']['greek']) != (cfg['index'], cfg['match']['greek']):
            raise ValueError('The fetch runs were made under different index settings')
        cfg = cfg or run_cfg
        for rel, info in manifest['gene_info']['files'].items():
            if rel in files:
                raise ValueError(f'{rel} is in two fetch runs')
            path = root / 'data' / 'raw' / rid / info['raw']
            if digest(path.read_bytes()) != info['sha256']:
                raise ValueError(f'{rel} differs from the fetch manifest; the raw file was modified')
            files[rel] = dict(info, path=str(path))
    tax_meta = json.loads(taxonomy_index_path(root, taxonomy_run).with_name('index.json').read_text(encoding='utf-8'))
    path = index_path(root, fetch_run_id)
    meta_path = path.with_name('index.json')
    identity = dict(index_version=INDEX_VERSION, fetch_run=fetch_run_id, files={k: v['sha256'] for k, v in sorted(files.items())}, taxonomy_run=taxonomy_run,
                    taxonomy_index_sha256=tax_meta['index_sha256'], index_config=cfg['index'], greek=cfg['match']['greek'],
                    prefix_rule=dict(not_epithets=sorted(NOT_EPITHETS)))     # the prefixes table follows initials: a rule change forces a rebuild
    if path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if {k: meta.get(k) for k in identity} == identity and file_sha256(path) == meta['index_sha256']:
            return dict(path=str(path), existing=True, counts=meta['counts'])
    classes = list(cfg['index']['name_classes'].items())
    skip_symbol, skip_desc = re.compile(cfg['index']['skip_symbols']), re.compile(cfg['index']['skip_descriptions'], re.I)
    greek = cfg['match']['greek']
    tax = Taxonomy(taxonomy_index_path(root, taxonomy_run))
    merged = {r[0]: r[1] for r in tax.db.execute('SELECT old, new FROM merged')}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('index.sqlite.tmp')
    if temp.exists():
        temp.unlink()
    db = sqlite3.connect(temp)
    db.executescript('''
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE genes (gene_id INTEGER PRIMARY KEY, taxid INTEGER NOT NULL, species INTEGER NOT NULL, genus INTEGER, symbol TEXT NOT NULL,
                            locus_tag TEXT, description TEXT, type TEXT);
        CREATE TABLE names (key TEXT NOT NULL, species INTEGER NOT NULL, gene_id INTEGER NOT NULL, cls INTEGER NOT NULL);
        CREATE TABLE species (taxid INTEGER PRIMARY KEY, genus INTEGER, name TEXT NOT NULL, rank TEXT, genes INTEGER NOT NULL);
        CREATE TABLE prefixes (prefix TEXT NOT NULL, species INTEGER NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    counts, place, species_genes = Counter(), {}, Counter()

    def placed(taxid):
        if taxid not in place:
            t = merged.get(taxid, taxid)
            if tax.node(t) is None:
                place[taxid] = None
            else:
                place[taxid] = (_rank_ancestor(tax, t, 'species') or t, _rank_ancestor(tax, t, 'genus'))
        return place[taxid]

    def rows():
        for rel, info in sorted(files.items()):
            with gzip.open(info['path'], 'rt', encoding='utf-8', errors='replace') as f:
                header = f.readline().lstrip('#').rstrip('\n').split('\t')
                col = {name: i for i, name in enumerate(header)}
                for line in f:
                    r = line.rstrip('\n').split('\t')
                    counts['rows'] += 1
                    where = placed(int(r[col['tax_id']]))
                    if where is None:
                        counts['taxid_not_in_backbone'] += 1
                        continue
                    species, genus = where
                    gid = int(r[col['GeneID']])
                    val = lambda name: None if r[col[name]] in ('-', '') else r[col[name]]
                    species_genes[species] += 1
                    yield ('gene', (gid, int(r[col['tax_id']]), species, genus, r[col['Symbol']], val('LocusTag'), val('description'), val('type_of_gene')))
                    seen = set()
                    for priority, (cls, column) in enumerate(classes):
                        raw_value = val(column)
                        if raw_value is None:
                            continue
                        for name in (raw_value.split('|') if cls == 'synonym' else [raw_value]):
                            if cls in ('symbol', 'nomenclature_symbol', 'synonym') and skip_symbol.match(name):
                                continue
                            if cls in ('description', 'full_name') and skip_desc.match(name):
                                continue
                            k = gkey(name, greek)
                            if k and k not in seen:
                                seen.add(k)
                                counts['names:' + cls] += 1
                                yield ('name', (k, species, gid, priority))
    batch_g, batch_n = [], []
    with db:
        for kind, row in rows():
            (batch_g if kind == 'gene' else batch_n).append(row)
            if len(batch_n) >= 200000:
                db.executemany('INSERT INTO genes VALUES (?,?,?,?,?,?,?,?)', batch_g)
                db.executemany('INSERT INTO names VALUES (?,?,?,?)', batch_n)
                batch_g, batch_n = [], []
        db.executemany('INSERT INTO genes VALUES (?,?,?,?,?,?,?,?)', batch_g)
        db.executemany('INSERT INTO names VALUES (?,?,?,?)', batch_n)
        for sp, n in sorted(species_genes.items()):
            name = tax.scientific_name(sp) or ''
            node = tax.node(sp) or {}
            db.execute('INSERT INTO species VALUES (?,?,?,?,?)', (sp, _rank_ancestor(tax, sp, 'genus'), name, node.get('rank'), n))
            for p in initials(name):
                db.execute('INSERT INTO prefixes VALUES (?,?)', (p, sp))
        db.execute('CREATE INDEX names_key ON names(key, species)')
        db.execute('CREATE INDEX prefixes_prefix ON prefixes(prefix)')
        db.execute('CREATE INDEX species_genus ON species(genus)')
        counts.update(genes=db.execute('SELECT count(*) FROM genes').fetchone()[0], names=db.execute('SELECT count(*) FROM names').fetchone()[0],
                      species=db.execute('SELECT count(*) FROM species').fetchone()[0])
        meta = dict(identity, name_class_order=[c for c, _ in classes], counts=dict(sorted(counts.items())), built_at=now())
        db.executemany('INSERT INTO meta VALUES (?,?)', ((k, json.dumps(v, sort_keys=True)) for k, v in meta.items()))
    db.close()
    tax.close()
    temp.replace(path)
    meta['index_sha256'] = file_sha256(path)
    save_json(meta_path, meta)
    return dict(path=str(path), existing=False, counts=meta['counts'])


NOT_EPITHETS = {'x', '×', 'hybrid', 'cultivar', 'group', 'sp', 'spp'}      # "Saccharum hybrid cultivar" gives no prefix


def initials(name):
    """Species prefixes a gene symbol may carry, case-folded: genus initial + first one or two letters of the epithet (Malus domestica
    → md, mdo; Brassica napus → bn, bna). Only binomials (two words, both alphabetic after the genus) give prefixes."""
    words = (name or '').split()
    if len(words) < 2 or not words[0].isalpha() or not words[1].isalpha() or words[1].casefold() in NOT_EPITHETS:
        return []
    g, e = words[0][0].casefold(), words[1].casefold()
    return sorted({g + e[:1], g + e[:2]})


class GeneIndex:
    """Read-only view of a built gene index."""

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise ValueError('No gene index at ' + str(path) + ' (run `gene build` first)')
        self.db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.meta = {r['key']: json.loads(r['value']) for r in self.db.execute('SELECT * FROM meta')}
        self.classes = list(self.meta['name_class_order'])          # the class priority of the build (meta JSON sorts dict keys)
        self._genus = {}

    def close(self):
        self.db.close()

    def species_of_genus(self, genus):
        if genus not in self._genus:
            self._genus[genus] = [r[0] for r in self.db.execute('SELECT taxid FROM species WHERE genus=? ORDER BY taxid', (genus,))]
        return self._genus[genus]

    def has_species(self, taxid):
        return self.db.execute('SELECT 1 FROM species WHERE taxid=?', (taxid,)).fetchone() is not None

    def prefix_species(self, prefix):
        return [r[0] for r in self.db.execute('SELECT species FROM prefixes WHERE prefix=? ORDER BY species', (prefix,))]

    def lookup(self, key, species, classes):
        """Genes whose name has `key` inside `species` (a list of species taxids), restricted to the name classes given; returns the
        hits of the strongest class present: [(gene_id, species, class)]."""
        if not key or not species:
            return []
        allowed = [self.classes.index(c) for c in classes]
        marks = ','.join('?' * len(species))
        rows = self.db.execute(f'SELECT gene_id, species, cls FROM names WHERE key=? AND species IN ({marks}) ORDER BY cls, gene_id', (key, *species)).fetchall()
        rows = [r for r in rows if r['cls'] in allowed]
        if not rows:
            return []
        best = rows[0]['cls']
        return [(r['gene_id'], r['species'], self.classes[r['cls']]) for r in rows if r['cls'] == best]

    def holders(self, key, species, classes):
        """Every gene of `species` (one taxid) that has `key` in any of the given name classes (a symbol that is also another gene's synonym)."""
        allowed = {self.classes.index(c) for c in classes}
        return sorted({r['gene_id'] for r in self.db.execute('SELECT gene_id, cls FROM names WHERE key=? AND species=?', (key, species)) if r['cls'] in allowed})

    def gene(self, gene_id):
        r = self.db.execute('SELECT * FROM genes WHERE gene_id=?', (gene_id,)).fetchone()
        return dict(r) if r else None


# ---- step 2: matching ------------------------------------------------------------------------------------------------
SYMBOL_CLASSES = ('symbol', 'nomenclature_symbol', 'synonym', 'locus_tag')
NAME_OF_A_GENE = ('symbol', 'nomenclature_symbol', 'synonym')          # classes whose clash between two genes of a species makes a symbol ambiguous (genes-v0.6)
ACRONYM_PREFIX = re.compile(r'^(?P<acr>[A-Z][A-Z0-9]{1,9})\s+(?P<sym>\S+)$')   # "BCTIV V2": an organism acronym, then the gene
CONTEXT_STOPWORDS = {'protein', 'proteins', 'family', 'like', 'putative', 'probable', 'domain', 'containing', 'gene', 'genes', 'type', 'homolog',
                     'isoform', 'subunit', 'chain', 'related', 'member', 'partial', 'precursor', 'uncharacterized', 'with', 'from', 'that', 'this'}
NAME_CLASSES = ('full_name', 'description', 'synonym')
NODE_TYPES = ('gene', 'symbol', 'qtl_marker')
MAX_CANDIDATES = 10


class Placer:
    """The genes-v0.1 rules for one literal gene value under the claim's organisms (see configs/genes-v0.yaml and the module
    docstring). Every part carries `via` — the rules that fired — so each placement can be reviewed."""

    def __init__(self, genes, tax, cfg, chem=None):
        self.genes, self.tax, self.cfg, self.chem = genes, tax, cfg, chem      # chem: a ChemIndex (genes-v0.7) for the parts left untyped
        self.greek = cfg['greek']
        self.separators = list(cfg['list_separators'])
        self.trailing = sorted(cfg['trailing_words'], key=len, reverse=True)
        self.plural = {w.casefold() for w in cfg.get('plural_words') or []}
        self.prefix = re.compile(cfg['prefix']['pattern'])
        self.min_rest = int(cfg['prefix']['min_rest'])
        self.symbol_form = re.compile(cfg['symbol_form'])
        self.types = [(name, re.compile(rx) if name == 'qtl_marker' else re.compile(rx, re.I)) for name, rx in cfg['types'].items()]
        self._species, self._prefixes = {}, {}
        self.context_used = False

    # -- organisms --
    def species_level(self, taxid):
        """A taxon at or below species → its species; above species → itself (a genus, a family ...)."""
        if taxid not in self._species:
            row = self.tax.db.execute('SELECT new FROM merged WHERE old=?', (taxid,)).fetchone()
            current = row[0] if row else taxid                         # a merged taxid is read as its current one
            self._species[taxid] = _rank_ancestor(self.tax, current, 'species') or current
        return self._species[taxid]

    def organisms(self, taxids):
        """The claim's organism taxids at species level, without those that are ancestors of another one (Oryza + Oryza sativa → Oryza sativa)."""
        out = sorted({self.species_level(t) for t in taxids})
        return [t for t in out if not any(o != t and t in self.tax.lineage(o) for o in out)]

    def search_space(self, organism):
        """Species taxids of the gene index an organism stands for: itself when it is a species in the index, the index's species
        of a genus, nothing otherwise (an organism above genus is too broad for a gene symbol)."""
        node = self.tax.node(organism) or {}
        if node.get('rank') == 'genus':
            return self.genes.species_of_genus(organism)
        return [organism] if self.genes.has_species(organism) else []

    def prefixes_of(self, taxid):
        """Prefixes a gene symbol of this organism may carry: initials of its scientific name and of its synonyms (Pennisetum glaucum
        → Pg for Cenchrus americanus); a genus stands for its species in the gene index, each with its own prefixes."""
        if taxid not in self._prefixes:
            node = self.tax.node(taxid) or {}
            species = self.genes.species_of_genus(taxid) if node.get('rank') == 'genus' else [taxid]
            out = {}
            for sp in species:
                names = [r[0] for r in self.tax.db.execute("SELECT name FROM names WHERE taxid=? AND class IN ('scientific name', 'synonym')", (sp,))]
                for n in names:
                    for p in initials(n):
                        out.setdefault(p, set()).add(sp)
            self._prefixes[taxid] = out
        return self._prefixes[taxid]

    def prefix_holders(self, prefix, taxids):
        return sorted({sp for t in taxids for sp in self.prefixes_of(t).get(prefix, ())})

    # -- text --
    def split(self, text):
        """List parts outside parentheses; the split is abandoned when a part holds no letter (chemical locants "1, 3-dihydroxy")."""
        blank = _outside(text)
        cuts = [(0, len(text))]
        for sep in self.separators:
            nxt = []
            for a, b in cuts:
                start, i = a, blank.find(sep, a, b)
                while i != -1:
                    nxt.append((start, i))
                    start = i + len(sep)
                    i = blank.find(sep, start, b)
                nxt.append((start, b))
            cuts = nxt
        parts = [re.sub(r'^(?:and|or|&)\s+', '', text[a:b].strip(' ,;')) for a, b in cuts]
        parts = [p for p in parts if p]
        if len(parts) > 1 and any(not re.search(r'[A-Za-zα-ωΑ-Ω]', p) for p in parts):
            return [text], False
        return parts, len(parts) > 1

    def strip_trailing(self, text):
        for w in self.trailing:
            m = re.match(r'^(?P<head>.+?)\s+' + re.escape(w) + r'$', text, re.I)
            if m:
                return m.group('head').strip(), w
        return text, None

    def type_of(self, text):
        for name, rx in self.types:
            if rx.search(text):
                return name
        return None

    def is_symbol(self, text):
        return bool(self.symbol_form.match(text))

    # -- one part --
    def acronym_organisms(self, acronym, interactors):
        """Organisms an acronym before a gene symbol names: NCBI Taxonomy's acronyms (TYLCV), else — for a virus-style acronym ending in V
        that the backbone lacks (BCTIV) — the claim's interacting viruses."""
        hits = sorted({h['taxid'] for h in self.tax.lookup(acronym) if h['cls'] in ('acronym', 'genbank acronym')})
        if hits:
            return hits
        if acronym.endswith('V') and len(acronym) >= 3:
            return [t for t in interactors if self.tax.group(t) == 'virus']
        return []

    def by_context(self, gene_ids, context):
        """The one candidate whose NCBI description shares a distinctive word with the claim's own text (statement and quotes): words of
        four letters or more, minus generic ones, minus words every candidate's description has. None unless exactly one candidate scores."""
        text = set(re.findall(r'[a-z]{4,}', context.casefold()))
        words = {g: set(re.findall(r'[a-z]{4,}', ((self.genes.gene(g) or {}).get('description') or '').casefold())) - CONTEXT_STOPWORDS for g in gene_ids}
        scores = {}
        for g, ws in words.items():
            others = set().union(*(words[o] for o in gene_ids if o != g))
            scores[g] = sorted((ws - others) & text)
        winners = [g for g, w in scores.items() if w]
        return (winners[0], scores[winners[0]]) if len(winners) == 1 else (None, [])

    def place_part(self, part, organisms, interactors, list_type=None, context=''):
        via, r = [], dict(part=part)
        word_type = self.type_of(part) or list_type
        if list_type and not self.type_of(part):
            via.append('type_from_list_tail')
        core, trailing = self.strip_trailing(part)
        plural = (bool(trailing) and trailing.casefold() in self.plural) or bool(list_type and 'type_from_list_tail' in via)
        if trailing:
            via.append('trailing_word:' + trailing.lower())
        if plural and not word_type:
            word_type = 'qtl_marker' if trailing.casefold() in ('loci', 'markers') else 'gene_family'
        forced = None                                      # genes-v0.6: "BCTIV V2" → the gene V2 of the virus the acronym names
        am = ACRONYM_PREFIX.match(core)
        if am and not plural and self.is_symbol(am.group('sym')):
            orgs = self.acronym_organisms(am.group('acr'), interactors)
            if orgs:
                forced, core = self.organisms(orgs), am.group('sym')
                via.append('acronym_prefix:' + am.group('acr'))
        symbol, full, alias = None, None, None
        m = re.match(r'^(?P<a>[^()]+?)\s*\((?P<b>[^()]+)\)$', core)
        if m:
            a, b = m.group('a').strip(), m.group('b').strip()
            sa, sb = self.is_symbol(a), self.is_symbol(b)
            if sb and not sa:
                symbol, full = b, a
            elif sa and not sb:
                symbol, full = a, b
            elif sa and sb:
                symbol, alias = a, b
            else:
                full = a
            via.append('parenthesis')
        elif self.is_symbol(core):
            symbol = core
        else:
            full = core
        # species prefix, checked against the claim's organisms, then the interactors, then the whole index (another species' gene)
        species, prefix_state, rest = None, None, None
        if symbol:
            pm = self.prefix.match(symbol)
            if pm and len(pm.group('rest')) >= self.min_rest:
                p, rest = pm.group('prefix').casefold(), pm.group('rest')
                own = self.prefix_holders(p, organisms)
                inter = self.prefix_holders(p, interactors)
                if len(own) > 1:                          # a genus-level organism: several of its species share the prefix
                    holding = [sp for sp in own if self.genes.lookup(gkey(rest, self.greek), [sp], SYMBOL_CLASSES)]
                    owners = [o for o in organisms if set(own) <= set(self.prefixes_of(o).get(p, ()))]
                    if len(holding) == 1:
                        own = holding
                    elif len(owners) == 1:
                        species, prefix_state = owners[0], 'claim_genus'
                if prefix_state:
                    pass
                elif len(own) == 1:
                    species, prefix_state = own[0], 'claim_organism'
                elif not own and len(inter) == 1:
                    species, prefix_state = inter[0], 'interactor'
                elif not own and not inter and pm.group('prefix')[0].isupper():
                    holders = []
                    for sp in self.genes.prefix_species(p):
                        if self.genes.lookup(gkey(rest, self.greek), [sp], SYMBOL_CLASSES) or self.genes.lookup(gkey(symbol, self.greek), [sp], SYMBOL_CLASSES):
                            holders.append(sp)
                    if len(holders) == 1:
                        species, prefix_state = holders[0], 'other_species'
                    else:
                        rest = None
                        via.append('prefix_unverified')
                else:
                    rest = None
                    via.append('prefix_unverified')
            if prefix_state:
                via.append('prefix:' + prefix_state)
        targets = [species] if species else (forced or organisms)
        # lookup inside the species (a genus-level organism: its species in the index); genes-v0.6: when the claim's organisms hold no gene
        # of that name, the interacting organisms are tried (a viral, fungal or bacterial gene named in a plant study)
        hits, hit_species = [], set()
        tries = []
        if symbol:
            tries += [(symbol, SYMBOL_CLASSES)] + ([(rest, SYMBOL_CLASSES)] if rest else []) + ([(alias, SYMBOL_CLASSES)] if alias else [])
        if full:
            tries += [(full, SYMBOL_CLASSES if len(full.split()) == 1 else NAME_CLASSES)]     # one word ("Wx", "rbcS"): read as a symbol too
        stages = [targets] + ([[i for i in interactors if i not in targets]] if interactors and not species and not forced else [])
        for stage, orgs in enumerate([] if plural else stages):      # a plural names a group of genes: never looked up as one gene
            for t in orgs:
                space = self.search_space(t)
                for text, classes in tries:
                    found = self.genes.lookup(gkey(text, self.greek), space, classes)
                    if found:
                        hits += [(g, sp, cls, t, text) for g, sp, cls in found]
                        break
            if hits:
                if stage == 1:
                    via.append('interactor_organism')
                break
        hit_organisms = sorted({h[3] for h in hits})
        gene_ids = sorted({h[0] for h in hits})
        if hits:
            via.append('ncbi_gene:' + hits[0][2])
            if len(gene_ids) == 1 and hits[0][2] in NAME_OF_A_GENE:     # genes-v0.6: the same name held by another gene of the species (KAT2)
                others = [g for g in self.genes.holders(gkey(hits[0][4], self.greek), hits[0][1], NAME_OF_A_GENE) if g != gene_ids[0]]
                if others:
                    gene_ids = sorted(gene_ids + others)
                    via.append('name_collision')
            if 1 < len(gene_ids) <= MAX_CANDIDATES and context:          # genes-v0.6: the claim's own words choose between candidates, or nothing
                self.context_used = True
                pick, words = self.by_context(gene_ids, context)
                if pick is not None:
                    gene_ids = [pick]
                    via.append('context:' + '+'.join(words))
            if len(gene_ids) == 1:                       # one gene: file it under the species NCBI gives it (a genus-level organism → its species)
                species = self.genes.gene(gene_ids[0])['species']
            elif len(hit_organisms) == 1:
                species = species or hit_organisms[0]
        # type: established by the program only
        if plural:
            gtype = word_type
        elif gene_ids:
            gtype = 'gene'
        elif prefix_state:
            gtype = 'gene'
        elif word_type:
            gtype = word_type
        elif symbol:
            gtype = 'symbol'
        else:
            gtype = 'untyped'
        if species is None and forced and len(forced) == 1:
            species = forced[0]
        if species is None and len(organisms) == 1:
            species = organisms[0]
        label = rest if (prefix_state and rest) else (symbol or full or core)
        key = gkey(label, self.greek)
        if len(gene_ids) == 1 and len(hit_organisms) == 1:
            status, node = 'resolved', f'ncbigene:{gene_ids[0]}'
        elif gene_ids:                              # a symbol NCBI holds several times in the species: the symbol is still the node
            status, node = 'ambiguous', (f'sym:{species}:{key}' if species is not None and key else None)
        elif gtype in NODE_TYPES and species is not None and symbol and key:
            status, node = 'placed', f'sym:{species}:{key}'
        elif species is None and gtype in NODE_TYPES and symbol:
            status, node = 'no_organism', None
        else:
            status, node = 'typed' if gtype != 'untyped' else 'untyped', None
        chem = None
        if gtype == 'untyped' and self.chem is not None and not plural:     # genes-v0.7: a name the gene rules leave untyped may be a ChEBI entity
            chem = self.chem.place(full or core)
            if chem:
                gtype, status, species, key = chem['type'], chem['status'], None, chem['key']
                node = f"chebi:{chem['ids'][0]}" if chem['status'] == 'resolved' else f"chem:{chem['key']}"
                via += chem['via']
        r.update(type=gtype, status=status, node=node, species=species, label=label, key=key, symbol=symbol, full=full, via=via)
        if chem:
            r['chebi_ids'] = chem['ids'][:MAX_CANDIDATES]
            if chem['status'] == 'resolved':
                r['chebi'] = dict(id=chem['ids'][0], name=chem['name'], is_class=chem['type'] == 'compound_class')
        if alias:
            r['alias'] = alias
        if gene_ids:
            r['gene_ids'] = gene_ids[:MAX_CANDIDATES]
            r['gene_candidates'] = len(gene_ids)
            if status == 'resolved':
                g = self.genes.gene(gene_ids[0])
                r['ncbi'] = dict(symbol=g['symbol'], description=g['description'], taxid=g['taxid'], species=g['species'])
        return r

    def place_compound(self, value):
        """genes-v0.9: a literal `compound` value (claims-v0.7+) is looked up in ChEBI only — the prompt keeps genes, proteins and
        enzymes out of that field. Lists split as for genes; each part gets the ChEBI rule of genes-v0.7, else stays untyped (no node)."""
        if self.chem is None:
            raise ValueError('A compound value needs the ChEBI index: bind --chem-run')
        clean = clean_value(value)
        parts, split = self.split(clean.translate(HYPHENS))
        placed = []
        for part in parts:
            r = dict(part=part, type='untyped', status='untyped', node=None, species=None, label=part, key=gkey(part, self.greek),
                     symbol=None, full=part, via=['compound_field'])
            chem = self.chem.place(part)
            if chem:
                r.update(type=chem['type'], status=chem['status'], key=chem['key'], via=r['via'] + chem['via'],
                         node=f"chebi:{chem['ids'][0]}" if chem['status'] == 'resolved' else f"chem:{chem['key']}")
                r['chebi_ids'] = chem['ids'][:MAX_CANDIDATES]
                if chem['status'] == 'resolved':
                    r['chebi'] = dict(id=chem['ids'][0], name=chem['name'], is_class=chem['type'] == 'compound_class')
            placed.append(r)
        return dict(clean=clean, split=split, parts=placed)

    def place(self, value, organisms, interactors, context=''):
        clean = clean_value(value)
        text = clean.translate(HYPHENS)
        parts, split = self.split(text)
        list_type = None
        if split:
            tail = self.type_of(parts[-1])
            core, trailing = self.strip_trailing(parts[-1])
            if tail and trailing and trailing.casefold() in self.plural:
                list_type = tail          # "LIM, MYB and C2H2 gene families": the tail's plural word types every symbol of the list
        placed = [self.place_part(p, organisms, interactors, list_type if (i < len(parts) - 1 and self.is_symbol(p)) else None, context) for i, p in enumerate(parts)]
        return dict(clean=clean, split=split, parts=placed)


def _outside(text):
    out, depth = [], 0
    for ch in text:
        if ch in '([（':
            depth += 1
        elif ch in ')]）':
            depth = max(0, depth - 1)
        out.append(' ' if depth or ch in '()[]（）' else ch)
    return ''.join(out)


def claim_organisms(layer_rows, claim, fields):
    """(organism taxids, interactor taxids) of one claim from the taxa layer. A claim with an organism field that the layer does not
    cover at the same revision is refused: the lookup would silently run without its species."""
    present = [f for f in fields if f in claim['fields']]
    if not present:
        return [], []
    v = layer_rows.get(claim['id'])
    if v is None or v['revision_id'] != claim['revision_id']:
        raise ValueError(f"Taxa layer does not cover claim {claim['id']} (revision {claim['revision_id']})")
    ids = {f: list((v['value'].get(f) or {}).get('ids') or []) for f in present}
    return ids.get(fields[0], []), [t for f in fields[1:] for t in ids.get(f, [])]


def match_claims(claims, placer, cfg):
    """claims yields (site paper id, claim id, literal gene value, organism taxids, interactor taxids) → one row per claim with a gene
    value. Results are reused by (cleaned value, organisms, interactors), except a result the claim's own words decided (genes-v0.6
    `context:`), which is computed again for every claim; the optional sixth element is that context (statement and quotes), the
    optional seventh the field (default cfg['field']; genes-v0.9: cfg['compound_field'] is placed in ChEBI only, one row per field)."""
    memo, counts, rows = {}, Counter(), []
    for item in claims:
        paper, claim_id, value, org_ids, inter_ids = item[:5]
        context = item[5] if len(item) > 5 else ''
        field = item[6] if len(item) > 6 else cfg['field']
        compound = field != cfg['field']
        if compound and field != cfg.get('compound_field'):
            raise ValueError('Not a gene or compound field: ' + field)
        organisms, interactors = placer.organisms(org_ids), placer.organisms(inter_ids)
        key = (field, clean_value(value)) if compound else (clean_value(value), tuple(organisms), tuple(interactors))
        hit = memo.get(key)
        if hit is None and compound:
            hit = memo[key] = placer.place_compound(value)
            counts['computed'] += 1
        elif hit is None:
            placer.context_used = False
            hit = placer.place(value, organisms, interactors, context)
            if placer.context_used:                      # the claim's own words chose between candidates: exact for this claim only
                counts['context_dependent'] += 1
            else:
                memo[key] = hit
                counts['computed'] += 1
        else:
            counts['reused'] += 1
        rows.append(dict(claim=claim_id, paper=paper, field=field, value=value, organisms=organisms, interactors=interactors, **hit))
    counts['spellings_reusable'] = len(memo)
    return rows, dict(sorted(counts.items()))


def summarize(rows, top=30):
    """Counts of values (claim × field) and parts by type and status; nodes and how many distinct written parts fold into them.
    `placed` means the program put the part on a (species, normalised symbol) node or an NCBI Gene id — never that it is right."""
    parts = [p for r in rows for p in r['parts']]
    nodes = defaultdict(set)
    for p in parts:
        if p['node']:
            nodes[p['node']].add(p['part'])
    values_with_node = sum(1 for r in rows if any(p['node'] for p in r['parts']))
    values_with_id = sum(1 for r in rows if any((p['node'] or '').startswith('ncbigene:') for p in r['parts']))
    values_with_chebi = sum(1 for r in rows if any((p['node'] or '').startswith('chebi:') for p in r['parts']))
    spell = {}
    for r in rows:
        spell.setdefault((fold(r['clean']), tuple(r['organisms'])), r)
    return dict(values=len(rows), spellings=len({fold(r['clean']) for r in rows}), spellings_with_organisms=len(spell),
                values_split=sum(1 for r in rows if r['split']), parts=len(parts),
                values_with_node=values_with_node, values_with_node_share=round(values_with_node / len(rows), 4) if rows else None,
                values_with_ncbi_id=values_with_id, values_with_ncbi_id_share=round(values_with_id / len(rows), 4) if rows else None,
                values_with_chebi_id=values_with_chebi, values_with_chebi_id_share=round(values_with_chebi / len(rows), 4) if rows else None,
                part_status=dict(sorted(Counter(p['status'] for p in parts).items())), part_type=dict(sorted(Counter(p['type'] for p in parts).items())),
                type_by_status={t: dict(sorted(Counter(p['status'] for p in parts if p['type'] == t).items())) for t in sorted({p['type'] for p in parts})},
                prefix=dict(sorted(Counter(v for p in parts for v in p['via'] if v.startswith('prefix')).items())),
                lookup_class=dict(sorted(Counter(v for p in parts for v in p['via'] if v.startswith('ncbi_gene:')).items())),
                values_without_organism=sum(1 for r in rows if not r['organisms']),
                nodes=len(nodes), nodes_with_several_spellings=sum(1 for v in nodes.values() if len({fold(x) for x in v}) > 1),
                written_parts_folded=sum(len({fold(x) for x in v}) - 1 for v in nodes.values()),
                folded_examples=sorted([sorted(v) for v in nodes.values() if len({fold(x) for x in v}) > 1])[:top])


SPELLING_COLUMNS = ['spelling', 'organisms', 'organism_names', 'values', 'claims', 'part', 'type', 'status', 'node', 'label', 'gene_ids', 'ncbi_symbol', 'ncbi_description', 'via']


def spelling_rows(rows, tax):
    """One reviewable row per (spelling, organisms, part): what the program decided and why."""
    groups = defaultdict(list)
    for r in rows:
        groups[(fold(r['clean']), tuple(r['organisms']))].append(r)
    out = []
    for (_, orgs), group in sorted(groups.items()):
        r = group[0]
        for p in r['parts']:
            out.append(dict(spelling=r['clean'], organisms=';'.join(map(str, orgs)), organism_names=';'.join(tax.scientific_name(t) or '' for t in orgs),
                            values=len(group), claims=len({g['claim'] for g in group}), part=p['part'], type=p['type'], status=p['status'], node=p['node'] or '',
                            label=p['label'], gene_ids=';'.join(map(str, p.get('gene_ids') or [])), ncbi_symbol=(p.get('ncbi') or {}).get('symbol') or '',
                            ncbi_description=(p.get('ncbi') or {}).get('description') or '', via=';'.join(p['via'])))
    return out


def open_indexes(root, gene_run, taxonomy_run):
    genes = GeneIndex(index_path(root, gene_run))
    meta = json.loads(index_path(root, gene_run).with_name('index.json').read_text(encoding='utf-8'))
    if meta['taxonomy_run'] != taxonomy_run:
        genes.close()
        raise ValueError(f"Gene index {gene_run} was built with {meta['taxonomy_run']}, not {taxonomy_run}")
    return genes, Taxonomy(taxonomy_index_path(root, taxonomy_run)), meta


def match_run(run):
    """The gene_match run: accepted claims of the bound layers read from the extraction store, their organisms from the bound taxa
    layer, every gene value placed with the genes-v0.1 rules; matches.jsonl, spellings.csv, summary.json in the run directory."""
    from .annotate import Annotations, KIND_TAXA
    from .layers import site_id
    from .store import Store
    from .taxonomy import store_claims, store_layers
    if run.replay or run.readonly:
        raise ValueError('match needs a writable, non-replay run')
    if (run.directory / 'summary.json').exists():
        raise ValueError('Run already matched')
    bound, cfg = run.config['gene_match'], run.config['genes']['match']
    genes, tax, meta = open_indexes(run.root, bound['gene_run'], bound['taxonomy_run'])
    chem, chem_meta = None, None
    if bound.get('chem_run'):                            # genes-v0.7: ChEBI for the untyped parts
        from .chem import ChemIndex, index_path as chem_index_path
        chem = ChemIndex(chem_index_path(run.root, bound['chem_run']))
        chem_meta = json.loads(chem_index_path(run.root, bound['chem_run']).with_name('index.json').read_text(encoding='utf-8'))
    store = Store(run.root, readonly=True)
    try:
        ann = Annotations(store)
        layer = ann.layer(bound['taxa_layer'])
        if layer['kind'] != KIND_TAXA:
            raise ValueError(f"{bound['taxa_layer']} is a {layer['kind']} layer, not {KIND_TAXA}")
        if layer['vocabulary']['taxonomy_run'] != bound['taxonomy_run']:
            raise ValueError(f"Taxa layer {bound['taxa_layer']} uses {layer['vocabulary']['taxonomy_run']}, not {bound['taxonomy_run']}")
        layer_rows = ann.claim_rows(bound['taxa_layer'])
        papers, info = store_layers(store, bound['layers'])

        def claims():
            for pid, c in store_claims(store, bound['layers'], papers):
                value = (c['fields'].get(cfg['field']) or {}).get('value')
                if value:
                    orgs, inters = claim_organisms(layer_rows, dict(c, fields={f: v['value'] for f, v in c['fields'].items()}), cfg['organism_fields'])
                    context = ' '.join([c.get('statement') or ''] + [e.get('quote') or '' for e in c.get('evidence') or []])
                    yield site_id(pid), c['id'], value, orgs, inters, context
                compound = (c['fields'].get(cfg['compound_field']) or {}).get('value') if cfg.get('compound_field') else None
                if compound:                               # genes-v0.9: ChEBI only, no organism conditioning
                    yield site_id(pid), c['id'], compound, [], [], '', cfg['compound_field']
        rows, reuse = match_claims(claims(), Placer(genes, tax, cfg, chem), cfg)
        summary = summarize(rows)
        with (run.directory / 'matches.jsonl').open('w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
        with (run.directory / 'spellings.csv').open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=SPELLING_COLUMNS)
            w.writeheader()
            w.writerows(spelling_rows(rows, tax))
    finally:
        store.close()
        genes.close()
        tax.close()
        if chem is not None:
            chem.close()
    run.artifact('summary.json', summary)
    run.manifest['gene_match'] = dict(gene_run=bound['gene_run'], gene_index_version=meta['index_version'], gene_index_sha256=meta['index_sha256'],
                                      taxonomy_run=bound['taxonomy_run'], taxa_layer=bound['taxa_layer'], rules=run.config['genes']['version'],
                                      **({'chem_run': bound['chem_run'], 'chem_index_sha256': chem_meta['index_sha256'], 'chebi_release': chem_meta['release']} if chem_meta else {}),
                                      claims_source=dict(source='extraction_store', layers=info, layers_from=bound.get('layers_from'), papers=len(papers)),
                                      values=len(rows), reuse=reuse,
                                      outputs={n: dict(sha256=digest((run.directory / n).read_bytes()), bytes=(run.directory / n).stat().st_size) for n in ('matches.jsonl', 'spellings.csv', 'summary.json')})
    run.finish('matched')
    return dict(run_id=run.id, values=len(rows), reuse=reuse, **{k: summary[k] for k in ('values_with_node_share', 'values_with_ncbi_id_share', 'values_with_chebi_id_share', 'part_status', 'part_type')},
                directory=str(run.directory))


def lookup_values(root, gene_run, taxonomy_run, values, taxids):
    """Offline check of the rules on literal values under the given organism taxids (no run, nothing written)."""
    import yaml
    cfg = yaml.safe_load((Path(root) / 'configs' / 'genes-v0.yaml').read_text(encoding='utf-8'))['match']
    genes, tax, _ = open_indexes(root, gene_run, taxonomy_run)
    try:
        placer = Placer(genes, tax, cfg)
        orgs = placer.organisms(taxids)
        return [dict(value=v, organisms=orgs, **placer.place(v, orgs, [])) for v in values]
    finally:
        genes.close()
        tax.close()


GENE_LAYER_RULES = dict(version='genes-import-v0.1', source='program_rules_only',
                        node='ncbi_gene_id_when_unique_in_the_species_else_species_and_normalised_symbol',
                        ambiguous='node_by_symbol_candidate_ids_kept', no_node='typed_untyped_or_without_organism')
PART_KEYS = ('part', 'type', 'status', 'node', 'label', 'species', 'via')


def import_genes_layer(store, root, match_run_id, name=None):
    """One gene_match run → one write-once `genes` layer (deterministic id; importing again is a no-op). The layer stores, per claim,
    the placed parts of its gene value, and a vocabulary of every node (label = most frequent written label, written forms, species,
    type, NCBI symbol / description / id when resolved) and every species used (lineage from the taxonomy index), so a site export
    needs no index."""
    from .annotate import Annotations, KIND_GENES
    from .layers import site_id
    from .taxonomy import lineage_info
    if store.readonly:
        raise ValueError('import writes the store')
    root = Path(root)
    run_dir = root / 'data' / 'runs' / match_run_id
    if not (run_dir / 'manifest.json').exists():
        raise ValueError('Match run does not exist: ' + match_run_id)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind') != 'gene_match' or manifest.get('status') != 'matched':
        raise ValueError('Not a completed gene_match run: ' + match_run_id)
    bound = manifest['gene_match']
    data = (run_dir / 'matches.jsonl').read_bytes()
    matches_sha = bound['outputs']['matches.jsonl']['sha256']
    if digest(data) != matches_sha:
        raise ValueError('matches.jsonl differs from the run manifest')
    meta = json.loads(index_path(root, bound['gene_run']).with_name('index.json').read_text(encoding='utf-8'))
    if meta['index_sha256'] != bound['gene_index_sha256']:
        raise ValueError('The gene index differs from the one the match run used')
    rows = [json.loads(l) for l in data.decode('utf-8').splitlines() if l.strip()]
    ann = Annotations(store)
    layer_id = 'ann-' + digest([KIND_GENES, match_run_id, matches_sha, meta['index_sha256'], GENE_LAYER_RULES])[:16]
    if ann.exists_any() and store.db.execute('SELECT 1 FROM annotation_layers WHERE layer_id=?', (layer_id,)).fetchone():
        return dict(layer_id=layer_id, existing=True, **ann.layer(layer_id)['stats'])
    fetched = {}                                          # gene_info file → retrieval date, over every fetch run of the index
    for rid in run_set(bound['gene_run']).split('+'):
        m = json.loads((root / 'data' / 'runs' / rid / 'manifest.json').read_text(encoding='utf-8'))
        fetched.update({rel: info['retrieved_at'][:10] for rel, info in m['gene_info']['files'].items()})
    values, paper_of, labels, info, species = {}, {}, defaultdict(Counter), {}, set()
    for r in rows:
        if r['field'] in values.get(r['claim'], {}):
            raise ValueError(f"Two rows for claim {r['claim']} field {r['field']}")
        parts = []
        for p in r['parts']:
            e = {k: p[k] for k in PART_KEYS if p.get(k) is not None}
            if p.get('gene_ids'):
                e['ids'] = list(p['gene_ids'])
                e['candidates'] = p['gene_candidates']
            if p.get('chebi_ids'):
                e['chebi'] = list(p['chebi_ids'])
            parts.append(e)
            if p['node']:
                labels[p['node']][p['label']] += 1
                info.setdefault(p['node'], dict(type=p['type'], status=p['status'], species=p['species'],
                                                **({'ncbi': dict(p['ncbi'], id=p['gene_ids'][0])} if p['status'] == 'resolved' and p.get('ncbi') else {}),
                                                **({'chebi': p['chebi']} if p.get('chebi') else {}),
                                                **({'chebi_candidates': p['chebi_ids']} if p.get('chebi_ids') and not p.get('chebi') else {})))
            if p.get('species') is not None:
                species.add(p['species'])
        values.setdefault(r['claim'], {})[r['field']] = dict(parts=parts)
        paper_of[r['claim']] = r['paper']
    tax = Taxonomy(taxonomy_index_path(root, bound['taxonomy_run']))
    try:
        species_info = {str(t): lineage_info(tax, t) for t in sorted(species)}
    finally:
        tax.close()
    nodes = {}
    for node, forms in sorted(labels.items()):
        nodes[node] = dict(info[node], label=sorted(forms, key=lambda f: (-forms[f], f))[0], forms=sorted(forms))
    resolved = []
    for claim_id in sorted(values):
        rev, paper = ann.resolve_claim(claim_id)
        if site_id(paper) != paper_of[claim_id]:
            raise ValueError(f'{claim_id} belongs to {paper}, not to {paper_of[claim_id]}')
        resolved.append((claim_id, rev, paper))
    parts = [p for v in values.values() for f in v.values() for p in f['parts']]
    stats = dict(claims=len(values), values=sum(len(v) for v in values.values()), parts=len(parts),
                 **({'values_by_field': dict(sorted(Counter(f for v in values.values() for f in v).items()))} if any(len(v) > 1 or 'gene' not in v for v in values.values()) else {}), nodes=len(nodes), species=len(species_info),
                 by_status=dict(sorted(Counter(p['status'] for p in parts).items())), by_type=dict(sorted(Counter(p['type'] for p in parts).items())),
                 values_with_node=sum(1 for v in values.values() for f in v.values() if any(p.get('node') for p in f['parts'])),
                 values_with_ncbi_id=sum(1 for v in values.values() for f in v.values() if any((p.get('node') or '').startswith('ncbigene:') for p in f['parts'])),
                 values_with_chebi_id=sum(1 for v in values.values() for f in v.values() if any((p.get('node') or '').startswith('chebi:') for p in f['parts'])))
    vocabulary = dict(backbone='NCBI Gene', gene_run=bound['gene_run'], backbone_retrieved=' / '.join(sorted(set(fetched.values()))),
                      backbone_files=dict(sorted(fetched.items())),
                      index_version=meta['index_version'], index_sha256=meta['index_sha256'], taxonomy_run=bound['taxonomy_run'], rules=GENE_LAYER_RULES,
                      match_rules=bound['rules'], fields=[f for f in (manifest['config_snapshot']['genes']['match']['field'], manifest['config_snapshot']['genes']['match'].get('compound_field')) if f], nodes=nodes, species=species_info,
                      chem=(dict(source='ChEBI', chem_run=bound['chem_run'], release=bound['chebi_release'], index_sha256=bound['chem_index_sha256'],
                                 licence='CC BY 4.0 (flat_files/LICENSE; the flat-file README says CC BY-SA 4.0); attribution: ChEBI, https://www.ebi.ac.uk/chebi, release ' + str(bound['chebi_release']))
                            if bound.get('chem_run') else None))
    source = dict(match_run=match_run_id, matches_sha256=matches_sha, taxa_layer=bound['taxa_layer'],
                  claims_source=dict(source=bound['claims_source']['source'], layers=[{k: l[k] for k in ('job', 'derivation', 'claim_ids_sha256')} for l in bound['claims_source']['layers']]))
    prov = dict(judge=None, program=dict(module='plantclaims_pipeline/genes.py', rules=bound['rules']), human_review='not_performed')
    with store.db:
        store.db.execute('INSERT INTO annotation_layers VALUES (?,?,?,?,?,?,?,?)',
                         (layer_id, KIND_GENES, name or f'gene field placed on NCBI Gene ({vocabulary["backbone_retrieved"]}) by program rules {bound["rules"]}', now(),
                          json.dumps(source, sort_keys=True), json.dumps(prov, sort_keys=True), json.dumps(vocabulary, sort_keys=True), json.dumps(stats, sort_keys=True)))
        for claim_id, rev, paper in resolved:
            store.db.execute('INSERT INTO annotation_claims VALUES (?,?,?,?,?,?)', (layer_id, claim_id, rev, paper, json.dumps(values[claim_id], sort_keys=True), None))
    return dict(layer_id=layer_id, existing=False, **stats)

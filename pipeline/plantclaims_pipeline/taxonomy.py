"""Organism normalisation against NCBI Taxonomy: fetch the taxonomy dump, build an offline index, and match the literal
organism fields of the claims with deterministic rules."""
import csv
import hashlib
import json
import re
import sqlite3
import tarfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import requests

from . import textclean
from .annotate import Annotations, JUDGE_DECLARATIONS, JUDGE_PROVENANCE_KEYS, KIND_TAXA
from .records import digest, now, save_json

INDEX_VERSION = 'taxonomy-index-v0.1'
ALLOWED_HOST = 'ftp.ncbi.nlm.nih.gov'
CLASS_PRIORITY = {'exact': 0, 'synonym': 1, 'common_name': 2, 'acronym': 3}
RANK_FORMS = {'var.': 'var.', 'subsp.': 'subsp.', 'ssp.': 'subsp.', 'f. sp.': 'f. sp.', 'f.sp.': 'f. sp.', 'f.': 'f.', 'pv.': 'pv.', 'subvar.': 'subvar.'}
LATIN = re.compile(r'^(?P<genus>[A-Z][a-z]{2,})\s+(?P<hyb>[x×]\s*)?(?P<epithet>[a-z][a-z-]{2,})(?P<rest>.*)$')
ABBREV = re.compile(r'^(?P<init>[A-Z])\s*\.\s*(?P<hyb>[x×]\s*)?(?P<epithet>[a-z][a-z-]{2,})(?P<rest>.*)$')
GENUS = re.compile(r'^(?P<genus>[A-Z][a-z]{2,})(?:\s+(?:sp\.|spp\.|sp|spp|species))?$')
RANK = re.compile(r'(?:^|\s)(?P<rank>var\.|subsp\.|ssp\.|f\.\s?sp\.|f\.|pv\.|subvar\.)\s+(?P<sub>[a-z][a-z-]{2,})')
CAPITALISED = re.compile(r'\b([A-Z][a-z]{2,})\b')
# a fragment that qualifies the organism named before it (cultivar, line, strain ...) rather than naming another one
QUALIFIER = re.compile(r"^(?:cvs?\.?|cultivars?|var\.|subsp\.|ssp\.|f\.\s?sp\.|pv\.|genotypes?|lines?|races?|strains?|isolates?|accessions?|hybrids?|['‘\"])", re.I)
QUALIFIER_TAIL = re.compile(r"^(?P<head>.+?)[,;]?\s+(?:cvs?\.?|cultivars?|genotypes?|lines?|races?|strains?|isolates?|accessions?|variety|varieties|['‘\"])(?:\s|$)", re.I)
LEADING_CONJUNCTION = re.compile(r'^(?:and|or|&)\s+', re.I)
COMMON_CLASSES = ('common name', 'genbank common name')
EMBEDDED_BINOMIAL = re.compile(r'(?<![A-Za-z])[A-Z][a-z]{2,}\s+(?:[x×]\s*)?[a-z][a-z-]{2,}')
HYPHEN_IN_WORD = re.compile(r'(?<=[a-z])[-‐‑](?=[a-z])')


def SPACING_FIXES(text):
    """Typographic spacing of Latin names: "P . cinnamomi" → "P. cinnamomi", "T urticae" → "T. urticae" (an initial without its
    period, at the start), "Cynodon sp ." → "Cynodon sp.", "Begonia X tuberhybrida" → "Begonia x tuberhybrida" (capital hybrid sign)."""
    t = re.sub(r'\b([A-Z])\s+\.\s*(?=[a-z])', r'\1. ', text)
    t = re.sub(r'^([A-Z])\s+(?=[a-z]{3,})', r'\1. ', t)
    t = re.sub(r'\b(spp?)\s+\.', r'\1.', t)
    return re.sub(r'\b([A-Z][a-z]{2,})\s+X\s+(?=[a-z])', r'\1 x ', t)


def edit_distance_one(a, b):
    """True when a and b differ by exactly one substitution, insertion or deletion."""
    if abs(len(a) - len(b)) > 1 or a == b:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1:]


def _outside_parens(text):
    """The text with every parenthesised span blanked to spaces (same length), so separators inside parentheses do not split."""
    out, depth = [], 0
    for ch in text:
        if ch in '(（':
            depth += 1
        elif ch in ')）':
            depth = max(0, depth - 1)
        out.append(' ' if depth or ch in '()（）' else ch)
    return ''.join(out)


def load_config(root):
    """configs/taxonomy-v0.yaml with the common-name table (match.rescue.common_names_file, taxonomy-v0.3) inlined as
    match.rescue.common_names, so a run's config snapshot and hash hold the table it used."""
    import yaml
    root = Path(root)
    cfg = yaml.safe_load((root / 'configs' / 'taxonomy-v0.yaml').read_text(encoding='utf-8'))
    rescue = (cfg.get('match') or {}).get('rescue') or {}
    if rescue.get('common_names_file'):
        rescue['common_names'] = yaml.safe_load((root / rescue['common_names_file']).read_text(encoding='utf-8'))
    return cfg


# ---- names ----------------------------------------------------------------------------------------------------------
def norm(text):
    """Lookup key: NFKC, case-folded, the multiplication sign as "x" (NCBI writes hybrids "Populus x canadensis"), one space."""
    t = unicodedata.normalize('NFKC', text or '').casefold().replace('×', 'x')
    return re.sub(r'\s+', ' ', t).strip(' \t"\'“”‘’')


def clean_value(value):
    """A field value as the matcher sees it: provider markup removed and entities decoded (textclean), quotes and trailing
    punctuation stripped, whitespace collapsed. The literal value itself is kept unchanged in every result."""
    text = textclean.clean_indexed(value or '')[0]
    text = re.sub(r'\s+', ' ', text).strip().strip('"\'“”‘’').strip()
    return text.rstrip(' .,;:').strip()


# ---- step 1: fetch ---------------------------------------------------------------------------------------------------
def _download(run, step, url, cfg, allowed_host=ALLOWED_HOST, agent='taxonomy'):
    """One recorded GET streamed to the run's raw directory. Intent is written before sending; every outcome is an attempt row.
    Only https to `allowed_host` (NCBI's FTP host by default; the vernacular fetch passes its configured host)."""
    if urlparse(url).scheme != 'https' or urlparse(url).hostname != allowed_host:
        raise ValueError('Transport only allows https://' + allowed_host)
    seq = len(run.manifest['attempts']) + 1
    target = run.raw / Path(urlparse(url).path).name
    attempt = dict(seq=seq, step=step, url=url, request_hash=digest(url), started_at=now(), status=None, bytes=0, sha256=None,
                   raw_path=str(target.relative_to(run.root)), finished_at=None, error=None)
    run.manifest['attempts'].append(attempt)
    run.flush()
    sha, size = hashlib.sha256(), 0
    temp = target.with_name(target.name + '.part')
    try:
        with requests.get(url, stream=True, timeout=cfg['timeout_seconds'], headers={'User-Agent': 'PlantClaims/0.1 ' + agent}) as r:
            attempt['status'] = r.status_code
            if r.status_code != 200:
                attempt['error'] = f'http_{r.status_code}'
            else:
                with temp.open('wb') as out:
                    for chunk in r.iter_content(1 << 16):
                        size += len(chunk)
                        if size > cfg['max_bytes']:
                            attempt['error'] = 'response_size_limit'
                            break
                        out.write(chunk)
                        sha.update(chunk)
    except requests.RequestException as exc:
        attempt['error'] = type(exc).__name__          # never the message: it can carry proxy details
    finally:
        attempt['finished_at'] = now()
        attempt['bytes'] = size
        attempt['sha256'] = sha.hexdigest() if size else None
        if attempt['error'] is None and temp.exists():
            temp.replace(target)
        elif temp.exists():
            temp.unlink()
        run.flush()
    if attempt['error']:
        raise RuntimeError(f"{step}: {attempt['error']}")
    return target, attempt


def fetch(run):
    """Download checksum + archive, verify the md5, extract the member files next to the archive; the run manifest records it all."""
    if run.replay or run.readonly:
        raise ValueError('fetch needs a writable, non-replay run')
    if run.manifest.get('taxdump'):
        raise ValueError('This run already holds a taxdump; fetch once per run')
    src = run.config['taxonomy']['source']
    run.manifest['status'] = 'fetching'
    try:
        checksum_path, _ = _download(run, 'checksum', src['base_url'] + src['checksum'], src)
        expected = checksum_path.read_text(encoding='utf-8').split()[0].lower()
        if not re.fullmatch(r'[0-9a-f]{32}', expected):
            raise RuntimeError('checksum: no md5 digest in the checksum file')
        archive_path, attempt = _download(run, 'archive', src['base_url'] + src['archive'], src)
        md5 = hashlib.md5()
        with archive_path.open('rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                md5.update(chunk)
        if md5.hexdigest() != expected:
            raise RuntimeError('archive: md5 differs from the checksum file')
        members = {}
        with tarfile.open(archive_path, 'r:gz') as tar:
            for name in src['members']:
                if '/' in name or name.startswith('.'):
                    raise ValueError('Member names must be plain file names')
                member = tar.getmember(name)
                if not member.isfile():
                    raise RuntimeError(f'archive: {name} is not a regular file')
                out = run.raw / name
                with tar.extractfile(member) as f, out.open('wb') as o:
                    h = hashlib.sha256()
                    for chunk in iter(lambda: f.read(1 << 20), b''):
                        o.write(chunk)
                        h.update(chunk)
                members[name] = dict(bytes=member.size, sha256=h.hexdigest(), mtime=member.mtime)
        run.manifest['taxdump'] = dict(source=src['name'], archive=src['archive'], archive_sha256=attempt['sha256'], archive_bytes=attempt['bytes'],
                                       md5=expected, md5_ok=True, members=members, retrieved_at=attempt['finished_at'])
        run.finish('fetched')
        return dict(run_id=run.id, archive_bytes=attempt['bytes'], archive_sha256=attempt['sha256'], md5_ok=True, members=members)
    except Exception as exc:
        run.manifest['failure'] = f'{type(exc).__name__}: {exc}'[:300]
        run.finish('failed')
        raise


# ---- step 1b: index --------------------------------------------------------------------------------------------------
def index_path(root, fetch_run_id):
    return Path(root) / 'data' / 'taxonomy' / fetch_run_id / 'index.sqlite'


def _dmp_rows(path):
    """NCBI dmp format: fields separated by "\\t|\\t", every line terminated by "\\t|"."""
    with Path(path).open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.endswith('\t|'):
                line = line[:-2]
            yield line.split('\t|\t')


def build_index(root, fetch_run_id, force=False):
    """names.dmp + nodes.dmp + merged.dmp → SQLite. Idempotent: an index built from the same member files is reused."""
    root = Path(root)
    manifest_path = root / 'data' / 'runs' / fetch_run_id / 'manifest.json'
    if not manifest_path.exists():
        raise ValueError('Fetch run does not exist: ' + fetch_run_id)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('kind') != 'taxonomy_fetch' or manifest.get('status') != 'fetched':
        raise ValueError('Not a completed taxonomy_fetch run: ' + fetch_run_id)
    cfg = manifest['config_snapshot']['taxonomy']
    members = manifest['taxdump']['members']
    raw = root / 'data' / 'raw' / fetch_run_id
    for name, info in members.items():
        if digest((raw / name).read_bytes()) != info['sha256']:
            raise ValueError(f'{name} differs from the fetch manifest; the raw dump was modified')
    path = index_path(root, fetch_run_id)
    meta_path = path.with_name('index.json')
    identity = dict(index_version=INDEX_VERSION, fetch_run=fetch_run_id, members=members, groups=cfg['groups'])
    if path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if {k: meta.get(k) for k in identity} == identity and digest(path.read_bytes()) == meta['index_sha256']:
            return dict(path=str(path), existing=True, **{k: meta[k] for k in ('counts', 'group_taxids')})
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('index.sqlite.tmp')
    if temp.exists():
        temp.unlink()
    db = sqlite3.connect(temp)
    db.executescript('''
        CREATE TABLE nodes (taxid INTEGER PRIMARY KEY, parent INTEGER NOT NULL, rank TEXT NOT NULL);
        CREATE TABLE names (norm TEXT NOT NULL, taxid INTEGER NOT NULL, class TEXT NOT NULL, name TEXT NOT NULL);
        CREATE TABLE merged (old INTEGER PRIMARY KEY, new INTEGER NOT NULL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    counts = Counter()
    with db:
        db.executemany('INSERT INTO nodes VALUES (?,?,?)', ((int(r[0]), int(r[1]), r[2]) for r in _dmp_rows(raw / 'nodes.dmp')))
        classes = cfg['match']['name_classes']

        def name_rows():
            for r in _dmp_rows(raw / 'names.dmp'):
                if r[3] in classes:
                    counts[r[3]] += 1
                    yield norm(r[1]), int(r[0]), r[3], r[1]
        db.executemany('INSERT INTO names VALUES (?,?,?,?)', name_rows())
        db.executemany('INSERT INTO merged VALUES (?,?)', ((int(r[0]), int(r[1])) for r in _dmp_rows(raw / 'merged.dmp')))
        db.execute('CREATE INDEX names_norm ON names(norm)')
        db.execute('CREATE INDEX names_taxid ON names(taxid, class)')
        group_taxids = {}
        for label, name in cfg['groups'].items():
            rows = db.execute("SELECT taxid FROM names WHERE norm=? AND class='scientific name'", (norm(name),)).fetchall()
            if len(rows) != 1:
                raise ValueError(f'Group anchor {name!r} resolves to {len(rows)} taxa in the dump')
            group_taxids[label] = rows[0][0]
        counts.update(nodes=db.execute('SELECT count(*) FROM nodes').fetchone()[0], names=db.execute('SELECT count(*) FROM names').fetchone()[0],
                      merged=db.execute('SELECT count(*) FROM merged').fetchone()[0])
        meta = dict(identity, counts=dict(sorted(counts.items())), group_taxids=group_taxids, built_at=now())
        db.executemany('INSERT INTO meta VALUES (?,?)', ((k, json.dumps(v, sort_keys=True)) for k, v in meta.items()))
    db.execute('VACUUM')
    db.close()
    temp.replace(path)
    meta['index_sha256'] = digest(path.read_bytes())
    save_json(meta_path, meta)
    return dict(path=str(path), existing=False, counts=meta['counts'], group_taxids=group_taxids)


class Taxonomy:
    """Read-only view of a built index."""

    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise ValueError('No taxonomy index at ' + str(path) + ' (run `taxon build` first)')
        self.path = path
        self.db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        self.db.row_factory = sqlite3.Row
        self.meta = {r['key']: json.loads(r['value']) for r in self.db.execute('SELECT * FROM meta')}
        self.groups = {int(t): label for label, t in self.meta['group_taxids'].items()}
        self._lineage = {}
        self._suffixes = None

    def close(self):
        self.db.close()

    def lookup(self, text):
        key = norm(text)
        if not key:
            return []
        return [dict(taxid=r['taxid'], cls=r['class'], name=r['name']) for r in self.db.execute('SELECT taxid, class, name FROM names WHERE norm=? ORDER BY taxid, class', (key,))]

    def suffix_lookup(self, text, limit=200):
        """Common names ending in the words of `text` ("lettuce" → "garden lettuce", "prickly lettuce" ...), used for hints only.
        Same answer as `suffix_scan` (the SQL LIKE scan, about 0.25 s per call) from an in-memory map of every word suffix of the
        backbone's common names, built once per index (46K names)."""
        key = norm(text)
        if not key:
            return []
        if '%' in key or '_' in key:                       # LIKE wildcards: only the scan answers these exactly
            return self.suffix_scan(key, limit)
        if self._suffixes is None:
            suffixes = defaultdict(set)
            for r in self.db.execute('SELECT taxid, class, name, norm FROM names WHERE class IN (?, ?)', COMMON_CLASSES):
                words = r['norm'].split(' ')
                for i in range(1, len(words)):
                    suffixes[' '.join(words[i:])].add((r['name'], r['taxid'], r['class']))
            self._suffixes = {k: sorted(v) for k, v in suffixes.items()}
        return [dict(taxid=t, cls=c, name=n) for n, t, c in self._suffixes.get(key, [])[:limit]]

    def suffix_scan(self, text, limit=200):
        key = norm(text)
        return [dict(taxid=r['taxid'], cls=r['class'], name=r['name']) for r in self.db.execute(
            "SELECT DISTINCT taxid, class, name FROM names WHERE class IN (?, ?) AND norm LIKE ? ORDER BY name, taxid, class LIMIT ?", (*COMMON_CLASSES, '% ' + key, limit))]

    def node(self, taxid):
        r = self.db.execute('SELECT parent, rank FROM nodes WHERE taxid=?', (taxid,)).fetchone()
        return dict(parent=r['parent'], rank=r['rank']) if r else None

    def scientific_name(self, taxid):
        r = self.db.execute("SELECT name FROM names WHERE taxid=? AND class='scientific name'", (taxid,)).fetchone()
        return r['name'] if r else None

    def lineage(self, taxid):
        if taxid in self._lineage:
            return self._lineage[taxid]
        out, seen, current = [], set(), taxid
        while current and current not in seen:
            seen.add(current)
            out.append(current)
            node = self.node(current)
            if node is None or node['parent'] == current:
                break
            current = node['parent']
        self._lineage[taxid] = out
        return out

    def group(self, taxid):
        for t in self.lineage(taxid):
            if t in self.groups:
                return self.groups[t]
        return 'other'

    def describe(self, taxid):
        node = self.node(taxid) or {}
        return dict(taxid=taxid, name=self.scientific_name(taxid), rank=node.get('rank'), group=self.group(taxid))


# ---- step 2: matching ------------------------------------------------------------------------------------------------
class Matcher:
    def __init__(self, tax, cfg):
        self.tax, self.cfg = tax, cfg
        self.methods = cfg['name_classes']
        self.trailing = set(cfg['trailing_words'])
        self.separators = list(cfg['list_separators'])
        self.preference = cfg.get('group_preference') or {}
        self.leading = set(cfg.get('leading_words') or [])
        self.hint_limit = int(cfg.get('hint_limit') or 0)
        self._hints = {}
        self.context_used = False
        rescue = cfg.get('rescue') or {}                  # taxonomy-v0.2: a second pass for values the rules leave unresolved (absent = v0.1 behaviour)
        self.rescue_enabled = bool(rescue)
        self.rescue_separators = list(rescue.get('list_separators') or [])
        self.near_limit = int(rescue.get('near_name_limit') or 0)
        self.spacing_plural = bool(rescue.get('ncbi_spacing_plural'))    # taxonomy-v0.4: NCBI common names ignoring spaces, hyphens and plural endings
        self.spacing_only = bool(rescue.get('ncbi_spacing'))              # taxonomy-v0.5: spaces and hyphens only
        self._compact = None
        self.table, self.table_version = {}, None
        table = rescue.get('common_names')
        if table:                                         # taxonomy-v0.3: approved common names, consulted last in the rescue pass
            self.table_version = table['version']
            for e in table['entries']:
                if tax.scientific_name(e['taxid']) != e['ncbi_name']:
                    raise ValueError(f"Common-name table {table['version']}: {e['name']!r} → {e['taxid']} is not {e['ncbi_name']!r} in this backbone")
                key = norm(e['name'])
                if key in self.table:
                    raise ValueError(f"Common-name table {table['version']}: {e['name']!r} listed twice")
                self.table[key] = e
        self.rescuing, self._fired, self._tried = False, set(), set()

    # -- one lookup with candidate ranking --
    def direct(self, text, field, hits=None):
        """Whole-text lookup (or the given backbone `hits`) → one result: the best name class, then lineage-ancestor and group-preference
        disambiguation, else ambiguous."""
        hits = [dict(h, method=self.methods[h['cls']]) for h in (self.tax.lookup(text) if hits is None else hits) if h['cls'] in self.methods]
        if not hits:
            return None
        best = min(CLASS_PRIORITY[h['method']] for h in hits)
        top = [h for h in hits if CLASS_PRIORITY[h['method']] == best]
        taxids = sorted({h['taxid'] for h in top})
        method = top[0]['method']
        if len(taxids) == 1:
            return self._hit(taxids[0], method, text)
        # NCBI lists one common name at a species and again at its subspecies/variety ("maize"): take the ancestor, the least specific claim
        ancestors = [t for t in taxids if all(t in self.tax.lineage(other) for other in taxids)]
        if len(ancestors) == 1:
            return dict(self._hit(ancestors[0], method, text), disambiguation='lineage_ancestor', candidates=[self.tax.describe(t) for t in taxids])
        preferred = self.preference.get(field)
        if preferred:
            keep = [t for t in taxids if self.tax.group(t) == preferred]
            if len(keep) == 1:
                return dict(self._hit(keep[0], method, text), disambiguation='group_preference:' + preferred,
                            candidates=[self.tax.describe(t) for t in taxids])
        return dict(status='ambiguous', method=method, matched=text, taxids=[], taxa=[], candidates=[self.tax.describe(t) for t in taxids])

    def _hit(self, taxid, method, text):
        return dict(status='resolved', method=method, matched=text, taxids=[taxid], taxa=[self.tax.describe(taxid)])

    @staticmethod
    def _unresolved(reason, **extra):
        return dict(status='unresolved', method=None, matched=None, taxids=[], taxa=[], reason=reason, **extra)

    # -- the rules --
    def latin(self, text, field, genus=None):
        """Binomial with optional hybrid sign, rank marker and trailing author/cultivar text; `genus` replaces an abbreviated one."""
        m = LATIN.match(text) if genus is None else ABBREV.match(text)
        if not m:
            return None
        g = genus or m.group('genus')
        epithet = m.group('epithet')
        hyb = 'x ' if m.group('hyb') else ''
        rest = m.group('rest') or ''
        if self.continues_with_another(rest):
            return None                                    # "A. vesicaria and K. sedifolia": a list, handled by `resolve`
        forms = []
        rank = RANK.search(rest)
        if rank:
            marker = RANK_FORMS[re.sub(r'\s+', ' ', rank.group('rank'))]
            forms.append(f"{g} {hyb}{epithet} {marker} {rank.group('sub')}")
        forms.append(f'{g} {hyb}{epithet}')
        for form in forms:
            r = self.direct(form, field)
            if r:
                return dict(r, via='latin_form' if form == text else 'latin_prefix')
        genus_hit = self.direct(g, field)
        if genus_hit and genus_hit['status'] == 'resolved' and genus_hit['taxa'][0]['rank'] == 'genus':
            return self._unresolved('species_not_in_backbone', tried=forms, genus=genus_hit['taxa'][0])
        return None

    def continues_with_another(self, rest):
        """True when the text after a name introduces another organism: a list separator followed by anything but a qualifier."""
        flat = _outside_parens(rest)
        for sep in self.separators:
            if sep.strip() == '&':
                continue                                   # after a Latin name "&" joins authors (Snyder & Hansen), not organisms
            i = flat.find(sep)
            if i >= 0:
                tail = LEADING_CONJUNCTION.sub('', flat[i + len(sep):].strip(' ,;'))
                if tail and (tail[0].isalpha() or tail[0] in '\'‘"') and not QUALIFIER.match(tail):     # "2n=2x=14: AA" is not an organism
                    return True
        return False

    def qualifier_stripped(self, text, field):
        """"tomato, cv. Moneymaker", "rice line IR64": the organism is what precedes the qualifier."""
        m = QUALIFIER_TAIL.match(text)
        if not m:
            return None
        head = m.group('head').strip(' ,;')
        if not head or norm(head) == norm(text):
            return None
        for rule in (lambda: self.direct(head, field), lambda: self.latin(head, field), lambda: self.common_variants(head, field)):
            r = rule()
            if r and r['status'] != 'unresolved':
                return dict(r, via=(r.get('via') + '+' if r.get('via') else '') + 'qualifier_stripped:' + head)
        return None

    def genus_only(self, text, field):
        m = GENUS.match(text)
        if not m:
            return None
        r = self.direct(m.group('genus'), field)
        if r and r['status'] == 'resolved' and r['taxa'][0]['rank'] == 'genus':
            return dict(r, method='genus', via='genus_only')
        return r

    def abbreviated(self, text, field, context):
        m = ABBREV.match(text)
        if not m:
            return None
        initial = m.group('init')
        self.context_used = True                           # the result now depends on the paper's text: `match_claims` does not reuse it for another paper
        if callable(context):
            context = context()
        genera = sorted({w for w in CAPITALISED.findall(context or '') if w[0] == initial and w != initial})
        found, tried = {}, []
        for g in genera:
            r = self.latin(text, field, genus=g)
            tried.append(g)
            if r and r['status'] == 'resolved':
                found[r['taxids'][0]] = dict(r, genus=g)
        if len(found) == 1:
            r = next(iter(found.values()))
            return dict(r, method='abbrev_context', via='genus_from_paper_text:' + r['genus'])
        if len(found) > 1:
            return dict(status='ambiguous', method='abbrev_context', matched=text, taxids=[], taxa=[], candidates=[r['taxa'][0] for r in found.values()])
        return self._unresolved('abbreviation_unresolved', genera_in_text=tried)

    def common_variants(self, text, field, lookup=None):
        lookup = lookup or self.direct
        words = text.split()
        candidates = []
        while words:
            phrase = ' '.join(words)
            candidates.append(phrase)
            for singular in (re.sub(r'ies$', 'y', phrase), re.sub(r'(ch|sh|s|x|z)es$', r'\1', phrase), re.sub(r'oes$', 'o', phrase), re.sub(r's$', '', phrase)):
                if singular != phrase:
                    candidates.append(singular)
            if words[-1].lower() in self.trailing and len(words) > 1:
                words = words[:-1]
                continue
            if (words[0].lower() == 'the' or words[0].lower() in self.leading) and len(words) > 1:
                words = words[1:]
                continue
            break
        seen = {norm(text)} if lookup == self.direct else set()     # the whole value was already looked up by `direct`
        for cand in candidates:
            key = norm(cand)
            if key in seen:
                continue
            seen.add(key)
            r = lookup(cand, field)
            if r:
                return dict(r, via='common_variant:' + cand)
        return None

    def compact_direct(self, text, field):
        """NCBI common names equal to `text` once spaces and hyphens are ignored (v0.4 also with a plural ending -s / -es added);
        the usual class ranking and disambiguation of `direct` decide between several. Common-name classes only."""
        if self._compact is None:
            self._compact = defaultdict(list)
            for r in self.tax.db.execute('SELECT taxid, class, name, norm FROM names WHERE class IN (?, ?)', COMMON_CLASSES):
                self._compact[re.sub(r'[\s\-‐‑]', '', r['norm'])].append(dict(taxid=r['taxid'], cls=r['class'], name=r['name']))
        key = re.sub(r'[\s\-‐‑]', '', norm(text))
        for form in ((key, key + 's', key + 'es') if self.spacing_plural else (key,)):
            if form in self._compact:
                r = self.direct(text, field, hits=self._compact[form])
                if r:
                    return dict(r, matched=self._compact[form][0]['name'])
        return None

    def table_entry(self, text, field):
        """One approved common-name table entry for exactly this text (spaces and hyphens ignored as well), as a match result."""
        e = self.table.get(norm(text)) or next((v for k, v in self.table.items() if re.sub(r'[\s\-‐‑]', '', k) == re.sub(r'[\s\-‐‑]', '', norm(text))), None)
        if e is None:
            return None
        return dict(self._hit(e['taxid'], 'common_name_table', text), table=dict(version=self.table_version, name=e['name'], level=e['level']))

    def simple(self, text, field, context):
        """Everything except parentheses and lists, in a fixed order of rules; the first rule that returns decides."""
        r = self.direct(text, field)
        if r:
            return r
        # genus_only before latin: "Fusarium spp." must not be read as the species "spp"
        for rule in (lambda: self.genus_only(text, field), lambda: self.latin(text, field), lambda: self.abbreviated(text, field, context),
                     lambda: self.common_variants(text, field), lambda: self.qualifier_stripped(text, field)):
            r = rule()
            if r:
                return r
        return self._unresolved('no_match')

    def split_list(self, text):
        """Parts of a list value, split at separators outside parentheses; a qualifier fragment stays with the part before it."""
        flat = _outside_parens(text)
        positions = []
        for sep in self.separators + (self.rescue_separators if self.rescuing else []):
            start = 0
            while (i := flat.find(sep, start)) >= 0:
                positions.append((i, len(sep), sep not in self.separators))
                start = i + len(sep)
        parts, last, extra = [], 0, False
        for i, n, is_extra in sorted(positions):
            if i < last:
                continue
            parts.append(text[last:i])
            last = i + n
            extra |= is_extra
        parts.append(text[last:])
        parts = [LEADING_CONJUNCTION.sub('', p.strip(' ,;')).strip() for p in parts]
        merged = []
        for p in parts:
            if not p:
                continue
            if merged and QUALIFIER.match(p):
                merged[-1] = merged[-1] + ', ' + p
            else:
                merged.append(p)
        if len(merged) > 1 and extra:
            self._fired.add('list_separator')
        return merged if len(merged) > 1 else None

    def hints(self, text, field):
        """For an unresolved short value: backbone common names that end in it, within the field's preferred group — material for a judge."""
        key = norm(text)
        if not self.hint_limit or len(key) < 4 or len(key.split()) > 3:
            return []
        cache = (key, field)
        if cache not in self._hints:
            preferred = self.preference.get(field)
            seen, out = set(), []
            for h in self.tax.suffix_lookup(key):
                if h['taxid'] in seen:
                    continue
                seen.add(h['taxid'])
                d = self.tax.describe(h['taxid'])
                if preferred and d['group'] != preferred:
                    continue
                out.append(dict(d, common_name=h['name']))
            self._hints[cache] = out[:self.hint_limit]
        return self._hints[cache]

    @staticmethod
    def _first_paren(text):
        """(before, inner, after) around the first top-level parenthesised span (nested parentheses stay inside `inner`), or None."""
        depth, start = 0, None
        for i, ch in enumerate(text):
            if ch in '(（':
                if depth == 0:
                    start = i
                depth += 1
            elif ch in ')）' and depth:
                depth -= 1
                if depth == 0:
                    return text[:start].strip(), text[start + 1:i].strip(), text[i + 1:].strip()
        return None

    def single(self, text, field, context):
        """A value naming one organism: the rules, then the parenthesised alternative (the text inside first, then the text around it)."""
        r = self.simple(text, field, context)
        if r['status'] != 'unresolved':
            return r
        paren = self._first_paren(text)
        if paren:
            before, inner, after = paren
            for cand, via in ((inner, 'parentheses_inner'), (f'{before} {after}'.strip(), 'parentheses_outer')):
                if cand and norm(cand) != norm(text):
                    alt = self.single(cand, field, context)          # shorter text: terminates on nested parentheses
                    if alt['status'] != 'unresolved':
                        merged = dict(alt)
                        merged['via'] = (alt.get('via') + '+' if alt.get('via') else '') + via
                        return merged
        if self.rescuing and text not in self._tried:
            self._tried.add(text)
            alt = self.rescued(text, field, context)
            if alt:
                return alt
        return r

    # -- the rescue pass (taxonomy-v0.2): only for values the rules above leave unresolved; every rule names itself in `rescue` --
    def _mark(self, r, rule):
        self._fired.add(rule)
        return dict(r, via=(r.get('via') + '+' if r.get('via') else '') + 'rescue:' + rule)

    def rescued(self, text, field, context):
        """Typographic repairs of one organism's text, then a Latin name embedded in it. None when nothing places it."""
        if '[' in text and ']' in text:                                            # "Aronia [Aronia melanocarpa (Michaux) Elliot]"
            alt = self.single(text.replace('[', '(').replace(']', ')'), field, context)
            if alt['status'] != 'unresolved':
                return self._mark(alt, 'brackets')
        spaced = SPACING_FIXES(text)                                               # "P . cinnamomi", "T urticae", "Cynodon sp .", "Begonia X tuberhybrida"
        if spaced != text:
            alt = self.single(spaced, field, context)
            if alt['status'] != 'unresolved':
                return self._mark(alt, 'spacing')
        if HYPHEN_IN_WORD.search(text):                                            # "soy-bean", "root‐knot"
            for variant in (HYPHEN_IN_WORD.sub('', text), HYPHEN_IN_WORD.sub(' ', text)):
                alt = self.direct(variant, field) or self.common_variants(variant, field)
                if alt and alt['status'] != 'unresolved':
                    return self._mark(dict(alt, matched=variant), 'hyphen_variant')
        alt = self.embedded_latin(text, field)
        if alt:
            return alt
        if self.table:                                                             # the approved common-name table, with the usual
            alt = self.common_variants(text, field, lookup=self.table_entry)       # plural / trailing-word / season-word variants
            if alt:
                return self._mark(alt, 'common_name_table')
        if self.spacing_plural or self.spacing_only:                               # last: NCBI's own common names written apart ("groundnut" =
            alt = self.common_variants(text, field, lookup=self.compact_direct)    # "ground-nut"; v0.4 also plural: "pepper" = "peppers"); after the
            if alt:                                                                # table, whose approved species rows ("sunflower" → H. annuus)
                return self._mark(alt, 'ncbi_spacing_plural' if self.spacing_plural else 'ncbi_spacing')   # would otherwise fall to NCBI's genus plurals
        return None

    def embedded_latin(self, text, field):
        """A Latin name inside descriptive text ("wheat Triticum aestivum", "fungal pathogen Botrytis cinerea"): every capitalised
        binomial is tried as a backbone name. Only for a value that is not a list, and only when exactly one taxon is found — two
        names are a list the splitter missed ("Calamogrostis canadensis, and for the cattail, Typha latifolia"), left to the judge."""
        if self.split_list(text):
            return None
        found, names = {}, set()
        for m in EMBEDDED_BINOMIAL.finditer(text):
            r = self.latin(text[m.start():], field)
            if r and r['status'] == 'resolved':
                found.setdefault(r['taxids'][0], r)
                names.add(r['taxids'][0])
            elif r and r.get('reason') == 'species_not_in_backbone':
                names.add(norm(m.group(0)))                                        # a second organism the backbone lacks still makes it a list
        if len(found) != 1 or len(names) != 1:
            return None
        r = next(iter(found.values()))
        return self._mark(dict(r, method='embedded_latin'), 'embedded_latin')

    def near_names(self, text):
        """For an unresolved binomial: species of the same backbone genus whose epithet is one edit away ("Populus trichacarpa" →
        Populus trichocarpa). Material for the judge only — the program cannot tell a typo from a species the backbone lacks."""
        m = LATIN.match(text)
        if not self.near_limit or not m:
            return []
        genus, epithet = m.group('genus'), m.group('epithet').lower()
        if len(epithet) < 5:
            return []
        prefix = norm(genus) + ' '
        out, seen = [], set()
        for row in self.tax.db.execute("SELECT taxid, name, norm FROM names WHERE norm >= ? AND norm < ? AND class = 'scientific name'", (prefix, prefix[:-1] + '!')):
            words = row['norm'].split(' ')
            if len(words) == 2 and words[1] != epithet and edit_distance_one(words[1], epithet) and row['taxid'] not in seen:
                seen.add(row['taxid'])
                out.append(self.tax.describe(row['taxid']))
        return sorted(out, key=lambda d: d['name'])[:self.near_limit]

    def resolve(self, value, field, context=''):
        """One literal field value → result dict (status resolved / ambiguous / unresolved, method, taxids, taxa, via, reason ...).
        Order: the whole value; a list (separators outside parentheses, every part as a single organism); the whole value with its
        parenthesised alternative. With a `rescue` config (taxonomy-v0.2) an unresolved value gets a second pass with the repairs
        of `rescued` and the extra list separators; a value the first pass placed or found ambiguous is never touched by it, and
        a second pass that places nothing leaves the first result as it was (plus `near_names` for a binomial)."""
        r = self._resolve(value, field, context)
        if not self.rescue_enabled or r['status'] != 'unresolved':
            return r
        self.rescuing, self._fired, self._tried = True, set(), set()
        try:
            second = self._resolve(value, field, context)
        finally:
            self.rescuing = False
        if second['status'] != 'unresolved':
            return dict(second, rescue=sorted(self._fired))
        near = self.near_names(r['clean'])
        return dict(r, near_names=near) if near else r

    def _resolve(self, value, field, context):
        clean = clean_value(value)
        base = dict(value=value, clean=clean, field=field)
        if not clean:
            return dict(base, **self._unresolved('empty'))
        r = self.simple(clean, field, context)
        if r['status'] != 'unresolved':
            return dict(base, **r)
        parts = self.split_list(clean)
        if parts:
            results = [self.single(p, field, context) for p in parts]
            if all(x['status'] == 'resolved' for x in results):
                taxids = sorted({t for x in results for t in x['taxids']})
                return dict(base, status='resolved', method='list', matched=clean, taxids=taxids, taxa=[self.tax.describe(t) for t in taxids],
                            via='list', parts=[dict(text=p, method=x['method'], via=x.get('via'), taxids=x['taxids']) for p, x in zip(parts, results)])
            if any(x['status'] != 'unresolved' for x in results):
                return dict(base, **self._unresolved('list_partial', parts=[dict(text=p, status=x['status'], method=x['method'], taxids=x['taxids'],
                                                                                reason=x.get('reason')) for p, x in zip(parts, results)]))
        r = dict(base, **self.single(clean, field, context))
        if r.get('reason') == 'no_match':
            r['hints'] = self.hints(clean, field)
        return r


def fold(value):
    return ' '.join((value or '').split()).casefold()


def store_layers(store, layers):
    """The accepted claims of (job, derivation) layers, merged as the site export merges them (a later layer replaces WHOLE papers),
    read straight from the extraction store. Returns (papers, info): papers = paper_id -> (layer index, input_key) of the last layer
    holding the paper; info = one record per layer (job, derivation, spec, accepted-claim digest, papers and claims used). Refused:
    a derivation of another spec, one that has not validated every stored answer, a policy overlay other than the area rules (those
    change area lists only, never acceptance or field values; the export's plain-vs-overlay id check is left to the export)."""
    from .layers import EXPORTABLE_POLICY_KEYS, job_behind
    from .derive import Deriver
    deriver, papers, info = Deriver(store), {}, []
    for i, layer in enumerate(layers):
        d, job = deriver.derivation(layer['derivation']), store.job(layer['job'])
        if d['spec_key'] != job['spec_key']:
            raise ValueError('Derivation belongs to another spec than the job: ' + layer['derivation'])
        if set(d['policy']) - EXPORTABLE_POLICY_KEYS:
            raise ValueError('Derivation carries a validator-policy overlay that changes acceptance: ' + layer['derivation'])
        if job_behind(deriver, layer['derivation'], layer['job']):
            raise ValueError('Derivation has not validated every stored answer of the job; run `derive run` first: ' + layer['derivation'])
        scope, args = deriver._scope(layer['derivation'], layer['job'], False, 'derived_papers')
        for row in store.db.execute('SELECT t.paper_id, t.input_key ' + scope, args):
            papers[row['paper_id']] = (i, row['input_key'])
        sha, n = deriver.claim_ids_sha256(layer['derivation'], layer['job'])
        info.append(dict(job=layer['job'], derivation=layer['derivation'], spec_key=d['spec_key'], validator_version=d['validator_version'],
                         policy_keys=sorted(d['policy']), claim_ids_sha256=sha, claims_accepted_in_layer=n))
    for i, entry in enumerate(info):
        entry['papers_used'] = sum(1 for layer, _ in papers.values() if layer == i)
    return papers, info


def store_claims(store, layers, papers):
    """Accepted claims of the winning (layer, paper) pairs, streamed one at a time in layer, paper and response order (the export's order)."""
    from .derive import Deriver, unpack
    deriver = Deriver(store)
    for i, layer in enumerate(layers):
        scope, args = deriver._scope(layer['derivation'], layer['job'], False, 'derived_claims')
        for row in store.db.execute("SELECT t.paper_id, t.data_z " + scope + " AND t.disposition = 'accepted' ORDER BY t.paper_id, t.position", args):
            if papers[row['paper_id']][0] == i:
                yield row['paper_id'], unpack(row['data_z'])


def paper_context(store, input_key):
    """The paper's own text as the model read it (cleaned title + abstract blocks), tags out and entities decoded: the only context
    the matcher consults (to expand an abbreviated genus)."""
    blocks = {b['block_id']: b['text'] for b in store.get_input(input_key)['blocks']}
    return textclean.clean_indexed((blocks.get('title') or '') + '\n' + (blocks.get('abstract') or ''))[0]


def match_claims(claims, context_of, tax, cfg):
    """Every (claim, field) with a value → one result row. `claims` yields (site paper id, claim id, {field: literal value});
    `context_of(site paper id)` returns the paper's text and is called only when a rule needs it. Results are reused by spelling:
    the matcher reads nothing but the cleaned value, the field and — only inside the abbreviated-genus rule — the paper's text, so a
    result computed without the text is exact for every other occurrence of the same (field, cleaned value); a result that used the
    text is computed again for each claim. Returns (rows, reuse counts)."""
    matcher, memo, counts = Matcher(tax, cfg), {}, Counter()
    rows, last = [], {}                                    # last: the most recent paper's text (claims arrive grouped by paper)

    def text_of(paper):
        if last.get('paper') != paper:
            last.update(paper=paper, text=context_of(paper))
        return last['text']
    for paper, claim_id, fields in claims:
        context = lambda paper=paper: text_of(paper)
        for field in cfg['fields']:
            if field not in fields:
                continue
            value = fields[field]
            key = (field, clean_value(value))
            hit = memo.get(key)
            if hit is not None:
                r = dict(hit, value=value)
                counts['reused'] += 1
            else:
                matcher.context_used = False
                r = matcher.resolve(value, field, context)
                if matcher.context_used:
                    counts['context_dependent'] += 1
                else:
                    memo[key] = {k: v for k, v in r.items() if k != 'value'}
                    counts['computed'] += 1
            rows.append(dict(claim=claim_id, paper=paper, **r))
    counts['spellings_reusable'] = len(memo)
    return rows, dict(sorted(counts.items()))


def summarize(rows, top=25):
    """Counts per field and overall: values (claim × field), distinct spellings, methods, reasons, groups, the largest unresolved and
    ambiguous spellings. Coverage means "the program placed the value in the backbone", never "the placement is right"."""
    out = dict(values=len(rows), fields={})
    for field in sorted({r['field'] for r in rows}):
        sub = [r for r in rows if r['field'] == field]
        spell = defaultdict(list)
        for r in sub:
            spell[fold(r['clean'])].append(r)
        status = Counter(r['status'] for r in sub)
        by_spelling = Counter(v[0]['status'] for v in spell.values())
        methods = Counter(r['method'] for r in sub if r['status'] == 'resolved')
        reasons = Counter(r.get('reason') for r in sub if r['status'] == 'unresolved')
        groups = Counter(g for r in sub if r['status'] == 'resolved' for g in sorted({t['group'] for t in r['taxa']}))
        ranks = Counter(g for r in sub if r['status'] == 'resolved' for g in sorted({t['rank'] for t in r['taxa']}))
        unresolved = sorted(((len(v), v[0]['clean'], v[0].get('reason')) for v in spell.values() if v[0]['status'] == 'unresolved'), key=lambda x: (-x[0], x[1]))
        ambiguous = sorted(((len(v), v[0]['clean'], [c['name'] for c in v[0]['candidates']]) for v in spell.values() if v[0]['status'] == 'ambiguous'), key=lambda x: (-x[0], x[1]))
        out['fields'][field] = dict(values=len(sub), spellings=len(spell), status=dict(sorted(status.items())), spellings_by_status=dict(sorted(by_spelling.items())),
                                    resolved_share=round(status['resolved'] / len(sub), 4) if sub else None,
                                    methods=dict(sorted(methods.items())), reasons=dict(sorted(reasons.items())), groups=dict(sorted(groups.items())), ranks=dict(sorted(ranks.items())),
                                    top_unresolved=[dict(values=n, spelling=s, reason=why) for n, s, why in unresolved[:top]],
                                    top_ambiguous=[dict(values=n, spelling=s, candidates=c) for n, s, c in ambiguous[:top]])
    out['status'] = dict(sorted(Counter(r['status'] for r in rows).items()))
    return out


def spelling_rows(rows):
    """One reviewable row per (field, spelling): what the program decided and why — the input of a later judge round."""
    spell = defaultdict(list)
    for r in rows:
        spell[(r['field'], fold(r['clean']))].append(r)
    out = []
    for (field, _), group in sorted(spell.items()):
        r = group[0]
        out.append(dict(field=field, spelling=r['clean'], values=len(group), claims=len({g['claim'] for g in group}), status=r['status'], method=r['method'],
                        via=r.get('via'), taxids=';'.join(str(t) for t in r['taxids']), names=';'.join(t['name'] or '' for t in r['taxa']),
                        ranks=';'.join(t['rank'] or '' for t in r['taxa']), groups=';'.join(t['group'] for t in r['taxa']), reason=r.get('reason'),
                        candidates=json.dumps(r.get('candidates') or [], ensure_ascii=False) if r.get('candidates') else '',
                        hints=';'.join(f"{h['name']} [{h['common_name']}]" for h in r.get('hints') or []),
                        detail=json.dumps({k: r[k] for k in DETAIL_KEYS if k in r}, ensure_ascii=False)
                        if any(k in r for k in DETAIL_KEYS) else ''))
    return out


DETAIL_KEYS = ('parts', 'genus', 'tried', 'genera_in_text', 'disambiguation', 'rescue', 'near_names')
SPELLING_COLUMNS = ['field', 'spelling', 'values', 'claims', 'status', 'method', 'via', 'taxids', 'names', 'ranks', 'groups', 'reason', 'candidates', 'hints', 'detail']


def match_run(run):
    """The taxon_match run: the accepted claims of the bound (job, derivation) layers read straight from the extraction store, their
    organism fields matched against the bound taxonomy index; matches.jsonl, spellings.csv and summary.json in the run directory.
    Nothing is written to the store or the website."""
    from .layers import site_id
    from .store import Store
    if run.replay or run.readonly:
        raise ValueError('match needs a writable, non-replay run')
    if (run.directory / 'summary.json').exists():
        raise ValueError('Run already matched')
    bound = run.config['taxon_match']
    layers = bound['layers']
    tax = Taxonomy(index_path(run.root, bound['taxonomy_run']))
    meta = json.loads(index_path(run.root, bound['taxonomy_run']).with_name('index.json').read_text(encoding='utf-8'))
    store = Store(run.root, readonly=True)
    try:
        papers, info = store_layers(store, layers)
        input_of = {site_id(pid): ikey for pid, (_, ikey) in papers.items()}
        claims = ((site_id(pid), c['id'], {f: v['value'] for f, v in c['fields'].items()}) for pid, c in store_claims(store, layers, papers))
        rows, reuse = match_claims(claims, lambda paper: paper_context(store, input_of[paper]), tax, run.config['taxonomy']['match'])
    finally:
        store.close()
        tax.close()
    summary = summarize(rows)
    with (run.directory / 'matches.jsonl').open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    with (run.directory / 'spellings.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SPELLING_COLUMNS)
        w.writeheader()
        w.writerows(spelling_rows(rows))
    run.artifact('summary.json', summary)
    run.manifest['taxon_match'] = dict(taxonomy_run=bound['taxonomy_run'], index_version=meta['index_version'], index_sha256=meta['index_sha256'],
                                       claims_source=dict(source='extraction_store', layers=info, layers_from=bound.get('layers_from'), papers=len(papers),
                                                          claims=len({r['claim'] for r in rows}) if rows else 0),
                                       fields=run.config['taxonomy']['match']['fields'], values=len(rows), reuse=reuse, status=summary['status'],
                                       outputs={n: dict(sha256=digest((run.directory / n).read_bytes()), bytes=(run.directory / n).stat().st_size) for n in ('matches.jsonl', 'spellings.csv', 'summary.json')})
    run.finish('matched')
    return dict(run_id=run.id, values=len(rows), reuse=reuse, status=summary['status'],
                fields={k: dict(values=v['values'], spellings=v['spellings'], resolved_share=v['resolved_share']) for k, v in summary['fields'].items()}, directory=str(run.directory))


# ---- step 3: the taxa annotation layer ------------------------------------------------------------------------------
# Program results + the judge's answers for what the program left → one write-once layer (annotate.py tables, kind `taxa`).
# Rules executed at import: a judged name must be a scientific name or synonym in the backbone;
# a homonym is taken by the field's group preference (taxon → plant) or left unresolved; a judged SPECIES absent from the
# backbone falls to its genus and the judged name is kept as text; an empty answer is `none`. Judged values are never edited.
TAXA_RULES = dict(version='taxa-import-v0.3', judged_names='scientific_name_or_synonym_in_the_backbone', homonym='group_preference_of_the_field_else_unresolved',
                  species_absent='genus_taxid_and_the_judged_name_kept_as_text', empty_answer='none',
                  lineage='species_id_family_id_order_stored_per_taxon',       # v0.2
                  precedence='judged_answer_over_program_result')             # v0.3: a spelling a packet answers takes the judge's
                                                                                # answer even where the (newer) program rules place it; the program's placement is kept as `program`
JUDGE_CLASSES = ('scientific name', 'synonym', 'equivalent name')
JUDGE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9 .'\-()×/]{1,120}$")


def read_taxon_packet(directory):
    """Frozen input + judge output of a taxon judgement packet → (items by id, answers by id, source, provenance). Structural
    validation only, the same rules as the packet's own check_output.py: every item answered once in order, `names` a list of
    distinct name strings, an empty list explained by a note, provenance keys and declarations complete."""
    d = Path(directory)
    manifest = json.loads((d / 'packet-manifest.json').read_text(encoding='utf-8'))
    if digest((d / 'INSTRUCTIONS.md').read_bytes()) != manifest['instructions_sha256']:
        raise ValueError('INSTRUCTIONS.md changed since the packet was frozen')
    prov_path = d / 'output' / 'provenance.json'
    if not prov_path.exists():
        raise ValueError('output/provenance.json is missing: the packet is not finished')
    prov = json.loads(prov_path.read_text(encoding='utf-8'))
    if not isinstance(prov, dict) or set(prov) != JUDGE_PROVENANCE_KEYS:
        raise ValueError('provenance.json keys differ from the packet contract')
    if prov['packet_manifest_sha256'] != digest((d / 'packet-manifest.json').read_bytes()) or prov['instructions_sha256'] != manifest['instructions_sha256']:
        raise ValueError('provenance.json does not name the frozen packet')
    if prov['human_review'] != 'not_performed' or not (isinstance(prov['judge_model_as_stated'], str) and prov['judge_model_as_stated'].strip()):
        raise ValueError('provenance.json: judge_model_as_stated must be stated and human_review must be not_performed')
    if any(not isinstance(prov[k], bool) for k in JUDGE_DECLARATIONS):
        raise ValueError('provenance.json: declarations must be true or false')
    items, answers, files = {}, {}, {}
    for name, info in sorted(manifest['files'].items()):
        if digest((d / 'input' / name).read_bytes()) != info['sha256']:
            raise ValueError(f'input/{name} changed since the packet was frozen')
        out = d / 'output' / name
        if not out.exists():
            raise ValueError(f'output/{name} is missing: the packet is not finished')
        files[name] = digest(out.read_bytes())
        expected = [json.loads(l) for l in (d / 'input' / name).read_text(encoding='utf-8').splitlines() if l.strip()]
        got = [json.loads(l) for l in out.read_text(encoding='utf-8').splitlines() if l.strip()]
        if [r.get('item_id') if isinstance(r, dict) else None for r in got] != [it['item_id'] for it in expected]:
            raise ValueError(f'output/{name}: items must be exactly the input items, once each, in input order')
        for it, row in zip(expected, got):
            if set(row) != {'item_id', 'names', 'note'} or not isinstance(row['names'], list) or not all(isinstance(n, str) and JUDGE_NAME.match(n) and n == n.strip() for n in row['names']) \
               or len({n.casefold() for n in row['names']}) != len(row['names']) or not (row['note'] is None or (isinstance(row['note'], str) and row['note'].strip())):
                raise ValueError(f"output/{name}: {it['item_id']}: needs item_id, names (distinct name strings), note (null or text)")
            if not row['names'] and row['note'] is None:
                raise ValueError(f"output/{name}: {it['item_id']}: an empty names list needs a note")
            items[it['item_id']] = it
            answers[it['item_id']] = dict(names=list(row['names']), note=row['note'])
    if len(items) != manifest['items']:
        raise ValueError('Item count differs from the packet manifest')
    source = dict(directory=str(d), packet_manifest_sha256=digest((d / 'packet-manifest.json').read_bytes()), instructions_sha256=manifest['instructions_sha256'],
                  output_files=files, provenance_sha256=digest(prov_path.read_bytes()), match_run=manifest['source']['match_run'],
                  packet_site_export_data_sha256=manifest['source'].get('site_export_data_sha256'), blinded=manifest.get('blinded'))
    return items, answers, source, prov


def lineage_info(tax, taxid):
    """name, rank, group and the backbone lineage of one taxon: genus, family (+ id), order (names; None when the lineage has none) and the
    species it belongs to (`species_id` / `species_name`: itself for a species, its parent species for a subspecies, variety, forma,
    strain or unranked node below a species, None above species) — what a site needs to file the taxon without the index."""
    d = tax.describe(taxid)
    info = dict(name=d['name'], rank=d['rank'], group=d['group'], genus=None, family=None, family_id=None, order=None, species_id=None, species_name=None)
    for t in tax.lineage(taxid):                       # self first, then ancestors: the nearest node of each rank wins
        r = (tax.node(t) or {}).get('rank')
        if r == 'species' and info['species_id'] is None:
            info['species_id'], info['species_name'] = t, tax.scientific_name(t)
        elif r == 'genus' and info['genus'] is None:
            info['genus'] = tax.scientific_name(t)
        elif r == 'family' and info['family'] is None:
            info['family'], info['family_id'] = tax.scientific_name(t), t
        elif r == 'order' and info['order'] is None:
            info['order'] = tax.scientific_name(t)
    return info


def find_judged_name(tax, name, preferred):
    """('found', taxid, candidates) | ('homonym', None, candidates) | ('absent', None, [])."""
    hits = [h for h in tax.lookup(name) if h['cls'] in JUDGE_CLASSES]
    if not hits and not name.lower().startswith('candidatus '):
        hits = [h for h in tax.lookup('Candidatus ' + name) if h['cls'] in JUDGE_CLASSES]
    sci = [h for h in hits if h['cls'] == 'scientific name'] or hits
    taxids = sorted({h['taxid'] for h in sci})
    if not taxids:
        return 'absent', None, []
    if len(taxids) == 1:
        return 'found', taxids[0], taxids
    keep = [t for t in taxids if tax.group(t) == preferred] if preferred else []
    if len(keep) == 1:
        return 'found', keep[0], taxids
    return 'homonym', None, taxids


def judge_entry(tax, answer, preferred, item_id):
    ids, absent, unresolved, homonyms = [], [], [], []
    for n in answer['names']:
        status, taxid, candidates = find_judged_name(tax, n, preferred)
        if status == 'found':
            ids.append(taxid)
        elif status == 'homonym':
            unresolved.append(n); homonyms.append(dict(name=n, candidates=candidates))
        else:
            words = n.split()
            gstatus, gid, _ = find_judged_name(tax, words[0], preferred) if len(words) > 1 else ('absent', None, [])
            if gstatus == 'found' and (tax.node(gid) or {}).get('rank') == 'genus':
                ids.append(gid); absent.append(n)
            else:
                unresolved.append(n)
    ids = sorted(set(ids))
    status = 'none' if not answer['names'] else 'resolved' if ids and not unresolved else 'partly_resolved' if ids else 'unresolved'
    entry = dict(status=status, source='judge', method='judge', ids=ids, judged_names=list(answer['names']), note=answer['note'], item_id=item_id)
    if absent:
        entry['absent_species'] = absent
    if unresolved:
        entry['unresolved_names'] = unresolved
    if homonyms:
        entry['homonyms'] = homonyms
    return entry


def import_taxa_layer(store, root, match_run_id, packet_dirs, name=None, allow_unjudged=False):
    """One match run + answered judgement packets → one write-once `taxa` layer (deterministic id; importing again is a no-op).
    Every claim value of the run gets an entry per organism field: the program's placement, or the judge's answer resolved
    under TAXA_RULES. The layer also stores name/rank/group/family/genus of every taxon it uses, so a site export needs no index.

    Judgement is by spelling: a packet answers (field, spelling) items, and an answer serves every claim of the run whose value
    has that cleaned spelling, whichever run the packet was built from. Several packets form one ledger in the order given; a
    later packet's answer to the same (field, spelling) replaces an earlier one (listed in the layer's source). A packet built
    from THIS run must be used completely; items of packets from earlier runs may match nothing here (counted). A value the
    program left unresolved and no packet answers stops the import, unless `allow_unjudged`: it is then stored as the program
    left it (status unresolved, source program, no ids) and counted, so a grown claim set can be imported before its new
    spellings are judged."""
    from .layers import site_id
    if store.readonly:
        raise ValueError('import writes the store')
    root = Path(root)
    run_dir = root / 'data' / 'runs' / match_run_id
    if not (run_dir / 'manifest.json').exists():
        raise ValueError('Match run does not exist: ' + match_run_id)
    manifest = json.loads((run_dir / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('kind') != 'taxon_match' or manifest.get('status') != 'matched':
        raise ValueError('Not a completed taxon_match run: ' + match_run_id)
    bound = manifest['taxon_match']
    matches_sha = bound['outputs']['matches.jsonl']['sha256']
    data = (run_dir / 'matches.jsonl').read_bytes()
    if digest(data) != matches_sha:
        raise ValueError('matches.jsonl differs from the run manifest')
    rows = [json.loads(l) for l in data.decode('utf-8').splitlines() if l.strip()]
    dirs = [packet_dirs] if isinstance(packet_dirs, (str, Path)) else list(packet_dirs)
    if not dirs and not allow_unjudged:
        raise ValueError('No judgement packet given (pass --allow-unjudged to store the program results alone)')
    packets = [read_taxon_packet(d) for d in dirs]
    if len({Path(d).resolve() for d in dirs}) != len(dirs):
        raise ValueError('The same packet is given twice')
    meta = json.loads(index_path(root, bound['taxonomy_run']).with_name('index.json').read_text(encoding='utf-8'))
    if meta['index_sha256'] != bound['index_sha256']:
        raise ValueError('The taxonomy index differs from the one the match run used')
    fetch_manifest = json.loads((root / 'data' / 'runs' / bound['taxonomy_run'] / 'manifest.json').read_text(encoding='utf-8'))
    preference = (manifest['config_snapshot']['taxonomy']['match'].get('group_preference') or {})
    fields = list(manifest['config_snapshot']['taxonomy']['match']['fields'])
    judge_as = manifest['config_snapshot']['taxonomy']['match'].get('judge_as') or {}      # taxonomy-v0.6: an interactor_2 spelling takes the interactor item
    jkey = lambda r: (judge_as.get(r['field'], r['field']), fold(r['clean']))
    ann = Annotations(store)
    if len(packets) == 1 and not allow_unjudged:
        source, prov = packets[0][2], packets[0][3]
        layer_id = 'ann-' + digest([KIND_TAXA, match_run_id, matches_sha, source['output_files'], source['provenance_sha256'], meta['index_sha256'], TAXA_RULES])[:16]
    else:
        layer_id = 'ann-' + digest([KIND_TAXA, match_run_id, matches_sha, [[p[2]['output_files'], p[2]['provenance_sha256']] for p in packets], meta['index_sha256'], TAXA_RULES,
                                    dict(allow_unjudged=bool(allow_unjudged))])[:16]
        source = dict(packets=[p[2] for p in packets], allow_unjudged=bool(allow_unjudged))
        prov = dict(packets=[p[3] for p in packets])
    if ann.exists_any() and store.db.execute('SELECT 1 FROM annotation_layers WHERE layer_id=?', (layer_id,)).fetchone():
        return dict(layer_id=layer_id, existing=True, **ann.layer(layer_id)['stats'])
    items, answers, item_of, replaced = {}, {}, {}, []
    for pi, (p_items, p_answers, p_source, _) in enumerate(packets):
        tag = (lambda iid: iid) if len(packets) == 1 else (lambda iid, pi=pi: f'p{pi + 1}:{iid}')
        for iid, it in p_items.items():
            key = (it['field'], fold(it['value']))
            if key in item_of:
                replaced.append(dict(field=key[0], spelling=it['value'], earlier=item_of[key], later=tag(iid)))
            item_of[key] = tag(iid)
            items[tag(iid)], answers[tag(iid)] = dict(it, _packet=pi), p_answers[iid]
    strict = {tag for tag, it in items.items() if packets[it['_packet']][2]['match_run'] == match_run_id}
    tax = Taxonomy(index_path(root, bound['taxonomy_run']))
    try:
        values, paper_of, used_items, used_ids = defaultdict(dict), {}, set(), set()
        for r in rows:
            judged_iid = item_of.get(jkey(r)) if TAXA_RULES['precedence'] == 'judged_answer_over_program_result' else None
            if r['status'] == 'resolved' and judged_iid is not None:          # the judge's answer wins; the program's placement stays visible
                entry = judge_entry(tax, answers[judged_iid], preference.get(r['field']), judged_iid)
                entry['program'] = dict(method=r['method'], ids=list(r['taxids']), **({'via': r['via']} if r.get('via') else {}))
                used_items.add(judged_iid)
            elif r['status'] == 'resolved':
                entry = dict(status='resolved', source='program', method=r['method'], ids=list(r['taxids']))
                if r.get('via'):
                    entry['via'] = r['via']
            else:
                iid = item_of.get(jkey(r))
                if iid is None:
                    if not allow_unjudged:
                        raise ValueError(f"No packet has an item for the {r['field']} value {r['clean']!r} (judge it, or pass --allow-unjudged)")
                    entry = dict(status='unresolved', source='program', method=None, ids=[], program_status=r['status'], reason=r.get('reason'))
                else:
                    entry = judge_entry(tax, answers[iid], preference.get(r['field']), iid)
                    used_items.add(iid)
            if r['field'] in values[r['claim']]:
                raise ValueError(f"Two rows for {r['claim']} / {r['field']}")
            values[r['claim']][r['field']] = entry
            paper_of[r['claim']] = r['paper']
            used_ids.update(entry['ids'])
        unused = sorted(set(items) - used_items)
        superseded = {r['earlier'] for r in replaced}                           # replaced by a later packet: not a mismatch with the run
        if (set(unused) & strict) - superseded:
            raise ValueError('Items of a packet built from this run match no value of the run: ' + ', '.join(sorted((set(unused) & strict) - superseded)))
        taxa = {str(t): lineage_info(tax, t) for t in sorted(used_ids)}
    finally:
        tax.close()
    resolved = []
    derivations = sorted({l['derivation'] for l in bound['claims_source']['layers']}) if 'claims_source' in bound else None   # indexed lookup for store-based runs
    for claim_id in sorted(values):
        rev, paper = ann.resolve_claim(claim_id, derivations)
        if site_id(paper) != paper_of[claim_id]:
            raise ValueError(f'{claim_id} belongs to {paper}, not to {paper_of[claim_id]}')
        resolved.append((claim_id, rev, paper))
    entries = [(f, e) for v in values.values() for f, e in v.items()]
    stats = dict(claims=len(values), values=len(entries), judge_items=len(items), taxa=len(taxa),
                 by_field={f: dict(sorted(Counter(e['status'] for g, e in entries if g == f).items())) for f in fields if any(g == f for g, _ in entries)},   # a field no claim has (interactor_2 before claims-v0.8) is left out
                 by_source=dict(sorted(Counter(e['source'] for _, e in entries).items())),
                 absent_species=sum(len(e.get('absent_species') or []) for _, e in entries), unresolved_names=sum(len(e.get('unresolved_names') or []) for _, e in entries),
                 none=sum(1 for _, e in entries if e['status'] == 'none'),
                 unjudged=sum(1 for _, e in entries if e['source'] == 'program' and e['status'] != 'resolved'),
                 judge_items_unused=len(unused), packets=len(packets), answers_replaced=len(replaced),
                 judged_over_program=sum(1 for _, e in entries if 'program' in e))
    vocabulary = dict(backbone='NCBI Taxonomy', taxonomy_run=bound['taxonomy_run'], backbone_retrieved=fetch_manifest['taxdump']['retrieved_at'][:10], index_version=meta['index_version'],
                      index_sha256=meta['index_sha256'], groups=meta['group_taxids'], group_preference=preference, rules=TAXA_RULES, fields=fields, taxa=taxa)
    claims_of = (dict(site_export_data_sha256=bound['site_export']['data_sha256'], site_export_version=bound['site_export']['version']) if 'site_export' in bound else
                 dict(claims_source=dict(source=bound['claims_source']['source'], layers=[{k: l[k] for k in ('job', 'derivation', 'claim_ids_sha256')} for l in bound['claims_source']['layers']])))
    if 'match_run' in source and source['match_run'] != match_run_id:        # one packet built from another run: keep which run it came from
        source = dict(source, packet_match_run=source['match_run'])
    source = dict(source, match_run=match_run_id, matches_sha256=matches_sha, **claims_of, **({'answers_replaced': replaced} if replaced else {}))
    with store.db:
        store.db.execute('INSERT INTO annotation_layers VALUES (?,?,?,?,?,?,?,?)',
                         (layer_id, KIND_TAXA, name or f'organism fields normalised to {vocabulary["backbone"]} ({vocabulary["backbone_retrieved"]}): program rules + judged leftovers', now(),
                          json.dumps(source, sort_keys=True), json.dumps(prov, sort_keys=True), json.dumps(vocabulary, sort_keys=True), json.dumps(stats, sort_keys=True)))
        for claim_id, rev, paper in resolved:
            store.db.execute('INSERT INTO annotation_claims VALUES (?,?,?,?,?,?)', (layer_id, claim_id, rev, paper, json.dumps(values[claim_id], sort_keys=True), None))
    return dict(layer_id=layer_id, existing=False, **stats)


__all__ = ['INDEX_VERSION', 'Matcher', 'load_config', 'TAXA_RULES', 'Taxonomy', 'build_index', 'clean_value', 'fetch', 'find_judged_name', 'fold', 'import_taxa_layer', 'index_path',
           'lineage_info', 'match_claims', 'match_run', 'paper_context', 'store_claims', 'store_layers', 'norm', 'read_taxon_packet', 'spelling_rows', 'summarize']


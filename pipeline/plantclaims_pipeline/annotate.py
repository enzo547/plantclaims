"""Additive annotation layers over accepted claims in the extraction store (organism and gene normalisation results).
A layer is write-once and never changes answers or derivations."""
import json
from collections import Counter
from pathlib import Path

import yaml

from .records import digest, now

SCHEMA = '''
CREATE TABLE IF NOT EXISTS annotation_layers (
  layer_id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL, created_at TEXT NOT NULL, source_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL, vocabulary_json TEXT NOT NULL, stats_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS annotation_claims (
  layer_id TEXT NOT NULL REFERENCES annotation_layers(layer_id), claim_id TEXT NOT NULL, revision_id TEXT NOT NULL, paper_id TEXT NOT NULL,
  value_json TEXT NOT NULL, note TEXT, PRIMARY KEY (layer_id, claim_id));
CREATE TABLE IF NOT EXISTS annotation_papers (
  layer_id TEXT NOT NULL REFERENCES annotation_layers(layer_id), paper_id TEXT NOT NULL, value_json TEXT NOT NULL, PRIMARY KEY (layer_id, paper_id));
'''
KIND_AREAS = 'research_areas'
KIND_TAXA = 'taxa'                     # organism fields normalised to a taxonomy backbone (plantclaims_pipeline/taxonomy.py import_taxa_layer)
KIND_GENES = 'genes'                   # gene field placed on (species, symbol) nodes and NCBI Gene ids by program rules (plantclaims_pipeline/genes.py import_genes_layer)
PROVENANCE_KEYS = {'judge_model_as_stated', 'interface', 'channel', 'input_tokens', 'output_tokens', 'cost_usd', 'started_at', 'finished_at',
                   'packet_manifest_sha256', 'instructions_sha256', 'saw_previous_areas', 'external_material_used', 'heuristic_classifier_used',
                   'human_review', 'notes'}
DECLARATIONS = ('saw_previous_areas', 'external_material_used', 'heuristic_classifier_used')
JUDGE_DECLARATIONS = ('saw_program_results', 'external_material_used', 'heuristic_resolver_used')      # the taxon judgement packet's declarations
JUDGE_PROVENANCE_KEYS = (PROVENANCE_KEYS - set(DECLARATIONS)) | set(JUDGE_DECLARATIONS)


def _sha(path):
    return digest(Path(path).read_bytes())


def area_vocabulary(scope_path):
    """What a research-area layer may contain: the scope's bucket ids minus the program-set ones (B8), plus every label."""
    scope = yaml.safe_load(Path(scope_path).read_text(encoding='utf-8'))
    rules = scope.get('bucket_rules') or {}
    program = sorted(b for b, r in (rules.get('objective_areas') or {}).items() if isinstance(r, dict) and r.get('set_by') == 'program')
    return dict(scope_version=scope['version'], labels=dict(scope['buckets']), allowed=[b for b in scope['buckets'] if b not in program], program_set=program,
                paper_level=rules.get('paper_level'), paper_level_fallback=rules.get('paper_level_fallback'))


def _areas_ok(value, allowed):
    return isinstance(value, list) and all(v in allowed for v in value) and len(set(value)) == len(value)


def seen_ids(resolved):
    return {c['claim_id'] for c, _, _ in resolved}


def read_area_packet(directory, vocabulary):
    """Frozen input + judge output of an area re-assignment packet → (claim rows, paper rows, source, provenance). Structural
    validation only (the same rules as the packet's own check_output.py): every input paper and claim answered once, values
    from the allowed vocabulary, an empty value explained, paper values only where the packet asked for them."""
    d = Path(directory)
    manifest = json.loads((d / 'packet-manifest.json').read_text(encoding='utf-8'))
    if _sha(d / 'INSTRUCTIONS.md') != manifest['instructions_sha256']:
        raise ValueError('INSTRUCTIONS.md changed since the packet was frozen')
    prov_path = d / 'output' / 'provenance.json'
    if not prov_path.exists():
        raise ValueError('output/provenance.json is missing: the packet is not finished')
    prov = json.loads(prov_path.read_text(encoding='utf-8'))
    if not isinstance(prov, dict) or set(prov) != PROVENANCE_KEYS:
        raise ValueError('provenance.json keys differ from the packet contract')
    if prov['packet_manifest_sha256'] != _sha(d / 'packet-manifest.json') or prov['instructions_sha256'] != manifest['instructions_sha256']:
        raise ValueError('provenance.json does not name the frozen packet')
    if prov['human_review'] != 'not_performed' or not (isinstance(prov['judge_model_as_stated'], str) and prov['judge_model_as_stated'].strip()):
        raise ValueError('provenance.json: judge_model_as_stated must be stated and human_review must be not_performed')
    if any(not isinstance(prov[k], bool) for k in DECLARATIONS):
        raise ValueError('provenance.json: declarations must be true or false')
    allowed, claims, papers, files = set(vocabulary['allowed']), [], [], {}
    for name, info in sorted(manifest['files'].items()):
        if _sha(d / 'input' / name) != info['sha256']:
            raise ValueError(f'input/{name} changed since the packet was frozen')
        out = d / 'output' / name
        if not out.exists():
            raise ValueError(f'output/{name} is missing: the packet is not finished')
        files[name] = _sha(out)
        expected = [json.loads(l) for l in (d / 'input' / name).read_text(encoding='utf-8').splitlines() if l.strip()]
        got = {}
        for line in out.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {'paper_id', 'claims', 'paper_buckets'} or row['paper_id'] in got:
                raise ValueError(f'output/{name}: malformed or repeated paper row')
            got[row['paper_id']] = row
        for paper in expected:
            row = got.pop(paper['paper_id'], None)
            if row is None:
                raise ValueError(f"output/{name}: paper {paper['paper_id']} not answered")
            want = [c['claim_id'] for c in paper['claims']]
            if not isinstance(row['claims'], list) or [c.get('claim_id') if isinstance(c, dict) else None for c in row['claims']] != want:
                raise ValueError(f"output/{name}: {paper['paper_id']}: claims differ from the packet")
            for c in row['claims']:
                if set(c) != {'claim_id', 'bk', 'note'} or not _areas_ok(c['bk'], allowed) or not (c['note'] is None or (isinstance(c['note'], str) and c['note'].strip())):
                    raise ValueError(f"output/{name}: {c.get('claim_id')}: bk must be distinct ids from {sorted(allowed)}, note null or text")
                if not c['bk'] and c['note'] is None:
                    raise ValueError(f"output/{name}: {c['claim_id']}: an empty value needs a note")
                claims.append(dict(claim_id=c['claim_id'], paper_id=paper['paper_id'], areas=list(c['bk']), note=c['note']))
            if paper['needs_paper_level']:
                if not _areas_ok(row['paper_buckets'], allowed):
                    raise ValueError(f"output/{name}: {paper['paper_id']}: paper_buckets missing or invalid")
                papers.append(dict(paper_id=paper['paper_id'], areas=list(row['paper_buckets'])))
            elif row['paper_buckets'] is not None:
                raise ValueError(f"output/{name}: {paper['paper_id']}: paper_buckets must be null for a paper with claims")
        if got:
            raise ValueError(f'output/{name}: papers outside the packet: ' + ', '.join(sorted(got)))
    if len(claims) != manifest['claims']:
        raise ValueError('Claim count differs from the packet manifest')
    source = dict(directory=str(d), packet_manifest_sha256=_sha(d / 'packet-manifest.json'), instructions_sha256=manifest['instructions_sha256'],
                  output_files=files, provenance_sha256=_sha(prov_path), source_site_export_data_sha256=manifest.get('source_site_export_data_sha256'),
                  source_layers=manifest.get('source_layers'), blinded=manifest.get('blinded'))
    return claims, papers, source, prov


class Annotations:
    def __init__(self, store):
        self.store, self.db = store, store.db
        if not store.readonly:
            self.db.executescript(SCHEMA)

    def exists_any(self):
        return self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='annotation_layers'").fetchone() is not None

    def list(self):
        if not self.exists_any():
            return []
        return [self._row(r) for r in self.db.execute('SELECT * FROM annotation_layers ORDER BY created_at')]

    def layer(self, layer_id):
        row = self.db.execute('SELECT * FROM annotation_layers WHERE layer_id=?', (layer_id,)).fetchone() if self.exists_any() else None
        if row is None:
            raise KeyError('No annotation layer ' + layer_id)
        return self._row(row)

    @staticmethod
    def _row(r):
        return dict(layer_id=r['layer_id'], kind=r['kind'], name=r['name'], created_at=r['created_at'], source=json.loads(r['source_json']),
                    provenance=json.loads(r['provenance_json']), vocabulary=json.loads(r['vocabulary_json']), stats=json.loads(r['stats_json']))

    def claim_values(self, layer_id):
        return {r['claim_id']: dict(revision_id=r['revision_id'], paper_id=r['paper_id'], areas=json.loads(r['value_json']), note=r['note'])
                for r in self.db.execute('SELECT * FROM annotation_claims WHERE layer_id=?', (layer_id,))}

    def claim_rows(self, layer_id):
        """Kind-agnostic claim rows: claim id → revision, paper, the stored value as JSON, note."""
        return {r['claim_id']: dict(revision_id=r['revision_id'], paper_id=r['paper_id'], value=json.loads(r['value_json']), note=r['note'])
                for r in self.db.execute('SELECT * FROM annotation_claims WHERE layer_id=?', (layer_id,))}

    def paper_values(self, layer_id):
        return {r['paper_id']: json.loads(r['value_json']) for r in self.db.execute('SELECT * FROM annotation_papers WHERE layer_id=?', (layer_id,))}

    def resolve_claim(self, claim_id, derivations=None):
        """Site claim id (claim-<16 hex>) → the one accepted derived claim revision it abbreviates (rev-<64 hex>) and its paper.
        `derivations` (the layers' derivation ids) turns the prefix match into a range on the derived_claims_rev index: without it
        the lookup scans every derived claim of the store (seconds each; fine for a pilot, not for millions of claims)."""
        if not (isinstance(claim_id, str) and claim_id.startswith('claim-') and len(claim_id) == 22):
            raise ValueError('Not a claim id: ' + str(claim_id))
        if derivations:
            prefix = 'rev-' + claim_id[6:]
            marks = ','.join('?' * len(derivations))
            rows = self.db.execute("SELECT DISTINCT revision_id, paper_id FROM derived_claims WHERE derivation_id IN (" + marks + ") AND revision_id >= ? AND revision_id < ? "
                                   "AND disposition = 'accepted'", (*sorted(set(derivations)), prefix, prefix + '~')).fetchall()
        else:
            rows = self.db.execute("SELECT DISTINCT revision_id, paper_id FROM derived_claims WHERE disposition = 'accepted' AND revision_id LIKE ?",
                                   ('rev-' + claim_id[6:] + '%',)).fetchall()
        if len(rows) != 1:
            raise ValueError(f'{claim_id} resolves to {len(rows)} accepted claim revisions in the store')
        return rows[0]['revision_id'], rows[0]['paper_id']

    def import_research_areas(self, directory, scope_path, name=None, base_layer=None):
        """One area re-assignment packet → one write-once layer. Deterministic id from the packet's output; importing the same
        packet again is a no-op, a packet with different output is another layer. With `base_layer` the packet is a PATCH: it
        may answer only claims the base layer holds (values replaced) and give paper values for papers of the base (added or
        replaced); every other value is copied from the base, so the new layer is complete on its own."""
        from .layers import site_id
        if self.store.readonly:
            raise ValueError('import writes the store')
        vocabulary = area_vocabulary(scope_path)
        claims, papers, source, prov = read_area_packet(directory, vocabulary)
        base = self.layer(base_layer) if base_layer else None
        if base is not None:
            if base['kind'] != KIND_AREAS:
                raise ValueError(f"Base layer {base_layer} is a {base['kind']} layer, not {KIND_AREAS}")
            if (base['vocabulary']['allowed'], base['vocabulary']['labels']) != (vocabulary['allowed'], vocabulary['labels']):
                raise ValueError('The patch vocabulary (ids, labels) differs from the base layer; a patch cannot redefine the areas')
        layer_id = 'ann-' + digest([KIND_AREAS, source['output_files'], source['provenance_sha256'], vocabulary] + ([base_layer] if base else []))[:16]
        if self.exists_any() and self.db.execute('SELECT 1 FROM annotation_layers WHERE layer_id=?', (layer_id,)).fetchone():
            return dict(layer_id=layer_id, existing=True, **self.layer(layer_id)['stats'])
        base_claims = self.claim_values(base_layer) if base else {}
        resolved, seen = [], set()
        for c in claims:
            rev, paper = self.resolve_claim(c['claim_id'])
            if site_id(paper) != c['paper_id']:
                raise ValueError(f"{c['claim_id']} belongs to {paper}, not to {c['paper_id']}")
            if rev in seen:
                raise ValueError('Two claim ids resolve to one revision: ' + rev)
            if base is not None and (c['claim_id'] not in base_claims or base_claims[c['claim_id']]['revision_id'] != rev):
                raise ValueError(f"Patch names {c['claim_id']}, which the base layer {base_layer} does not hold at this revision")
            seen.add(rev)
            resolved.append((c, rev, paper))
        if base is not None:
            base_paper_ids = {site_id(v['paper_id']) for v in base_claims.values()} | set(self.paper_values(base_layer))
            outside = sorted(p['paper_id'] for p in papers if p['paper_id'] not in base_paper_ids)
            if outside:
                raise ValueError('Patch gives paper values for papers outside the base layer: ' + ', '.join(outside))
            patched = dict(layer_id=base_layer, claims_replaced=sorted(c['claim_id'] for c in claims), papers_valued=sorted(p['paper_id'] for p in papers),
                           base_judge_model_as_stated=base['provenance']['judge_model_as_stated'], base_scope_version=base['vocabulary']['scope_version'],
                           base_source_directory=base['source']['directory'])
            kept = [(dict(claim_id=cid, paper_id=site_id(v['paper_id']), areas=v['areas'], note=v['note']), v['revision_id'], v['paper_id'])
                    for cid, v in sorted(base_claims.items()) if cid not in seen_ids(resolved)]
            resolved = kept + resolved
            merged_papers = dict(self.paper_values(base_layer)); merged_papers.update({p['paper_id']: p['areas'] for p in papers})
            papers = [dict(paper_id=pid, areas=areas) for pid, areas in sorted(merged_papers.items())]
            source = dict(source, patch_of=patched)
        all_claims = [c for c, _, _ in resolved]
        stats = dict(claims=len(all_claims), papers_with_claims=len({c['paper_id'] for c in all_claims}), papers_direct=len(papers), notes=sum(1 for c in all_claims if c['note']),
                     empty=sum(1 for c in all_claims if not c['areas']), area_counts=dict(sorted(Counter(a for c in all_claims for a in c['areas']).items())),
                     areas_per_claim=dict(sorted(Counter(str(len(c['areas'])) for c in all_claims).items())),
                     **({'patched': dict(claims=len(claims), paper_values=len(source['patch_of']['papers_valued']))} if base else {}))
        with self.db:
            self.db.execute('INSERT INTO annotation_layers VALUES (?,?,?,?,?,?,?,?)',
                            (layer_id, KIND_AREAS, name or ('research areas re-judged under ' + vocabulary['scope_version'] + (f' (patch over {base_layer})' if base else '')), now(),
                             json.dumps(source, sort_keys=True), json.dumps(prov, sort_keys=True), json.dumps(vocabulary, sort_keys=True), json.dumps(stats, sort_keys=True)))
            for c, rev, paper in resolved:
                self.db.execute('INSERT INTO annotation_claims VALUES (?,?,?,?,?,?)', (layer_id, c['claim_id'], rev, paper, json.dumps(c['areas']), c['note']))
            for p in papers:
                self.db.execute('INSERT INTO annotation_papers VALUES (?,?,?)', (layer_id, p['paper_id'], json.dumps(p['areas'])))
        return dict(layer_id=layer_id, existing=False, **stats)

    @staticmethod
    def _judge(p):
        return dict(model_as_stated=p['judge_model_as_stated'], interface=p['interface'], channel=p['channel'], started_at=p['started_at'], finished_at=p['finished_at'],
                    input_tokens=p['input_tokens'], output_tokens=p['output_tokens'], cost_usd=p['cost_usd'])

    def summary(self, layer_id):
        """What a site export or report shows about a layer: identity, judge as stated, declarations — never the values. For a
        patch layer, `patch_of` names the base layer, its judge and how many claims / paper values the patch round supplied;
        `judge` and `declarations` are then those of the patch round."""
        L = self.layer(layer_id)
        if L['kind'] == KIND_GENES:
            V, S = L['vocabulary'], L['stats']
            return dict(layer_id=layer_id, kind=L['kind'], name=L['name'], created_at=L['created_at'],
                        backbone=dict(source=V['backbone'], gene_run=V['gene_run'], retrieved=V['backbone_retrieved'], index_version=V['index_version'], index_sha256=V['index_sha256'],
                                      taxonomy_run=V['taxonomy_run'], nodes=len(V['nodes']),
                                      files=sorted(V.get('backbone_files') or {})),     # the gene_info files (Plants, Fungi, Viruses) behind the index
                        rules=V['rules'], match_rules=V['match_rules'], match_run=L['source']['match_run'], taxa_layer=L['source']['taxa_layer'], chem=V.get('chem'),
                        judge=None, human_review='not_performed', claims_annotated=S['claims'], values=S['values'], parts=S['parts'], by_status=S['by_status'],
                        role='development_candidates_not_gold', patch_of=None)
        if L['kind'] == KIND_TAXA:
            V, S = L['vocabulary'], L['stats']
            base = dict(layer_id=layer_id, kind=L['kind'], name=L['name'], created_at=L['created_at'],
                        backbone=dict(source=V['backbone'], taxonomy_run=V['taxonomy_run'], retrieved=V['backbone_retrieved'], index_version=V['index_version'], index_sha256=V['index_sha256'], taxa=len(V['taxa'])),
                        rules=V['rules'], match_run=L['source']['match_run'])
            if 'packets' not in L['source']:
                p = L['provenance']
                return dict(base, judge=self._judge(p), declarations={k: p[k] for k in JUDGE_DECLARATIONS}, human_review=p['human_review'], source_packet=L['source']['packet_manifest_sha256'],
                            claims_annotated=S['claims'], values=S['values'], by_source=S['by_source'], role='development_candidates_not_gold', patch_of=None)
            # several packets (a ledger of judged spellings): `judge` and `declarations` sum them up — models joined, a declaration true
            # when any packet declares it, tokens and cost only when every packet has them — and `judges` lists each packet as recorded
            ps = L['provenance']['packets']
            judges = [dict(self._judge(p), declarations={k: p[k] for k in JUDGE_DECLARATIONS}, human_review=p['human_review'], source_packet=src['packet_manifest_sha256'])
                      for p, src in zip(ps, L['source']['packets'])]

            def total(key):
                return sum(j[key] for j in judges) if judges and all(j[key] is not None for j in judges) else None
            judge = dict(model_as_stated=' + '.join(dict.fromkeys(j['model_as_stated'] for j in judges)) or None,
                         interface=' + '.join(dict.fromkeys(j['interface'] for j in judges)) or None, channel=' + '.join(dict.fromkeys(j['channel'] for j in judges)) or None,
                         started_at=min((j['started_at'] for j in judges), default=None), finished_at=max((j['finished_at'] for j in judges), default=None),
                         input_tokens=total('input_tokens'), output_tokens=total('output_tokens'), cost_usd=total('cost_usd'))
            reviews = {j['human_review'] for j in judges}
            return dict(base, judge=judge, declarations={k: any(j['declarations'][k] for j in judges) for k in JUDGE_DECLARATIONS},
                        human_review=reviews.pop() if len(reviews) == 1 else 'mixed', source_packet=None, judges=judges,
                        claims_annotated=S['claims'], values=S['values'], by_source=S['by_source'], unjudged=S['unjudged'], role='development_candidates_not_gold', patch_of=None)
        p, patch = L['provenance'], L['source'].get('patch_of')
        return dict(layer_id=layer_id, kind=L['kind'], name=L['name'], scope_version=L['vocabulary']['scope_version'], created_at=L['created_at'],
                    judge=dict(model_as_stated=p['judge_model_as_stated'], interface=p['interface'], channel=p['channel'], started_at=p['started_at'], finished_at=p['finished_at'],
                               input_tokens=p['input_tokens'], output_tokens=p['output_tokens'], cost_usd=p['cost_usd']),
                    declarations={k: p[k] for k in DECLARATIONS}, human_review=p['human_review'], source_packet=L['source']['packet_manifest_sha256'],
                    claims_annotated=L['stats']['claims'], papers_direct=L['stats']['papers_direct'], role='development_candidates_not_gold',
                    patch_of=(dict(layer_id=patch['layer_id'], judge_model_as_stated=patch['base_judge_model_as_stated'], scope_version=patch['base_scope_version'],
                                   claims_replaced=len(patch['claims_replaced']), paper_values_given=len(patch['papers_valued'])) if patch else None))


__all__ = ['Annotations', 'DECLARATIONS', 'JUDGE_DECLARATIONS', 'JUDGE_PROVENANCE_KEYS', 'KIND_AREAS', 'KIND_GENES', 'KIND_TAXA', 'SCHEMA', 'area_vocabulary', 'read_area_packet']

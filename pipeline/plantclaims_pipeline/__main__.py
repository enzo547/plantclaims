import argparse
import json
import sqlite3
from pathlib import Path

import yaml

from .records import Run


def main():
    parser = argparse.ArgumentParser(description='PlantClaims extraction and release pipeline')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent.parent)
    sub = parser.add_subparsers(dest='command', required=True)
    snap = sub.add_parser('snapshot', help='Candidate pool from the public OpenAlex snapshot (configs/scope-v0.6.yaml); retained raw, offline filter')
    snap.add_argument('action', choices=('fetch-manifest', 'download', 'filter', 'index', 'replay', 'status'))
    snap.add_argument('--run-id')
    snap.add_argument('--parallel', type=int, help='download streams (default from config)')
    snap.add_argument('--workers', type=int, help='filter worker processes (default from config)')
    snap.add_argument('--keys', nargs='*', help='restrict download/filter/replay to these file keys (works/updated_date=.../part_N.gz)')
    snap.add_argument('--key-prefix', help='restrict to file keys starting with this prefix, e.g. works/updated_date=2016')
    snap.add_argument('--force', action='store_true', help='re-filter files already filtered')
    cand = sub.add_parser('candidates', help='Versioned candidate-pool selection over a completed snapshot run (XPAC records excluded)')
    cand.add_argument('action', choices=('build', 'status', 'ids', 'verify'))
    cand.add_argument('--run-id', help='pool run (pool-...) for status/ids/verify')
    cand.add_argument('--source-run', help='completed snapshot_filter run to select from (build only)')
    chk = sub.add_parser('chunks', help='Chunk plan of the full run: one random order of the pilot-sample frame cut into chunks (configs/chunking-v0.yaml); no model call')
    chk.add_argument('action', choices=('build', 'status', 'verify'))
    chk.add_argument('--run-id', help='status / verify: the chunks- run')
    samp = sub.add_parser('sample', help='Stratified seeded pilot sample from a built candidate pool (configs/pilot-sample-v0.yaml); unlabeled')
    samp.add_argument('action', choices=('build', 'status', 'verify'))
    samp.add_argument('--run-id', help='sample run (sample-...) for status/verify')
    samp.add_argument('--pool', help='built candidate_pool run (pool-...) to sample from (build only)')
    ext = sub.add_parser('extract', help='Report whether the packet builder, request builder, validator and store of this code match the versions in plantclaims_pipeline/components.lock.json')
    ext.add_argument('action', choices=('components',))
    ext.add_argument('--write-lock', action='store_true', help='pin the current component fingerprints in plantclaims_pipeline/components.lock.json')
    sto = sub.add_parser('store', help='Content-addressed extraction store (data/extraction/): inputs, specs and model answers addressed by what they are')
    sto.add_argument('action', choices=('init', 'verify', 'stats'))
    job = sub.add_parser('job', help='Extraction job over the store: a paper selection + a spec; fills in only the answers that are missing (no implicit model calls)')
    job.add_argument('action', choices=('create', 'submit', 'collect', 'run', 'derive', 'status', 'report', 'set-policy'))
    job.add_argument('--job-id', help='existing job (xjob-...)')
    job.add_argument('--selection', help='create: sample:<pilot_sample run> | pool:<candidate_pool run>[:N] | ids:<file> (with --pool)')
    job.add_argument('--pool', help='create with ids:<file>: the candidate_pool run whose snapshot index locates the ids')
    job.add_argument('--workers', type=int, help='create from pool:/ids: — processes reading match files (default: pilot-sample scan.workers)')
    job.add_argument('--limit', type=int, help='create: at most N papers, stratified by cell')
    job.add_argument('--cell', nargs='*', help='create: restrict to these sample cells')
    job.add_argument('--paper-ids', type=Path, help='create: text file with one OpenAlex work id per line')
    job.add_argument('--model', help='create: override the configured model id (frozen into the spec)')
    job.add_argument('--extract-config', type=Path, help='create: extraction config file instead of configs/extract-v0.12.yaml (e.g. configs/extract-v0.12.yaml = prompt v0.15); snapshotted into the job')
    job.add_argument('--effort', choices=['low', 'medium', 'high', 'xhigh', 'max', 'none'],
                     help='create: override the effort level (frozen into the spec); none = no effort parameter (Claude Haiku 4.5)')
    job.add_argument('--thinking', help='create: adaptive (default) | none | budget:N (Claude Haiku 4.5: thinking off or a fixed budget, 1024 <= N < max_tokens)')
    job.add_argument('--min-core-claims', type=int, choices=(0, 1), help='create / set-policy: minimum accepted claims for core papers (explicit opt-in; 0 restores the original policy)')
    job.add_argument('--audit-reason', help='create / set-policy: why the job validation policy is being changed')
    job.add_argument('--spend-cap', type=float, help='create: override the spend cap in USD (frozen into the job)')
    job.add_argument('--max-batches', type=int, default=1, help='submit: create at most N Message Batches now (default 1; each is cut at ~200 MB / 50,000 requests)')
    job.add_argument('--max-requests', type=int, help='submit / run: send at most N requests now (smoke test: --max-requests 1)')
    job.add_argument('--wait', type=int, default=0, help='collect: poll up to this many seconds for batches that have not ended')
    der = sub.add_parser('derive', help='Validate stored answers of a spec under the current validator (optionally another validator policy); never calls a model')
    der.add_argument('action', choices=('run', 'list', 'report', 'drop'))
    der.add_argument('--spec', help='spec key (or a unique prefix)')
    der.add_argument('--job-id', help='take the spec from this job / restrict the report to its papers (xjob-... or import-extract-...)')
    der.add_argument('--policy', type=Path, help='YAML with validator-policy keys (field_literal_match, required_by_kind, bucket_rules); a claims-vX.yaml or a scope-vX.yaml (its bucket_rules) works')
    der.add_argument('--derivation', help='report / drop: derivation id (report default: current validator + given policy)')
    rel = sub.add_parser('release', help='Named release = a pin on (job selection, spec, derivation, answer watermark); files only on export')
    rel.add_argument('action', choices=('create', 'list', 'verify', 'export'))
    rel.add_argument('--name', help='release name (lowercase letters, digits, . _ -)')
    rel.add_argument('--job-id', help='create: job whose papers the release covers')
    rel.add_argument('--derivation', help='create: derivation id (default: current validator, no policy)')
    rel.add_argument('--note', help='create: free-text note kept with the pin')
    rel.add_argument('--out', type=Path, help='export: target directory')
    ann = sub.add_parser('annotate', help='Additive annotation layers over accepted claims (step 3): import a judged packet as a write-once layer; answers and derivations never change')
    ann.add_argument('action', choices=('import-taxa', 'import-genes', 'list', 'report'))
    ann.add_argument('--dir', type=Path, help='import-taxa: a directory of answered judgements for the spellings the program left unresolved')
    ann.add_argument('--name', help='import-taxa / import-genes: layer name')
    ann.add_argument('--layer', help='report: layer id (ann-...)')
    ann.add_argument('--packet', action='append', type=Path, help='import-taxa: a further answered judgement packet; repeat in order (a later answer to the same field + spelling replaces an earlier one)')
    ann.add_argument('--allow-unjudged', action='store_true', help='import-taxa: store values the program left unresolved and no packet answers as unresolved (program) instead of refusing')
    ann.add_argument('--match-run', help='import-genes: the gene_match run (genematch-...); import-taxa: the taxon_match run (taxmatch-...) whose program results and judgement packet (--dir) become one taxa layer')
    tax = sub.add_parser('taxon', help='Taxon normalisation (steps 1-2): fetch the NCBI taxdump as retained raw (2 HTTPS requests), build the offline index, match the exported claims\' literal organism fields with program rules; no model call')
    tax.add_argument('action', choices=('fetch', 'build', 'match', 'report', 'lookup'))
    tax.add_argument('--taxonomy-run', help='build/match/lookup: the taxonomy_fetch run (taxdump-...) whose dump is used')
    tax.add_argument('--run-id', help='report: a taxon_match run (taxmatch-...)')
    tax.add_argument('--config', type=Path, help='match: a YAML file whose `layers` (job, derivation pairs) name the claims to match, instead of --layer')
    tax.add_argument('--layer', action='append', metavar='JOB:DERIVATION', help='match: a (job, derivation) layer read straight from the extraction store; repeat in merge order (a later layer replaces whole papers)')
    tax.add_argument('--force', action='store_true', help='build: rebuild the index even if one exists')
    tax.add_argument('--field', default='taxon', help='lookup: field name the rules are applied for (taxon or interactor)')
    tax.add_argument('names', nargs='*', help='lookup: literal values to resolve offline (no run, no record)')
    vern = sub.add_parser('vernacular', help='Ready-made common-name lists: fetch a source as retained raw (1 HTTPS request); propose common-name mappings for the leftover spellings of a match run for review; no model call')
    vern.add_argument('action', choices=('fetch', 'candidates'))
    vern.add_argument('--source', default='grin', help='fetch: source id in configs/vernacular-v0.yaml')
    vern.add_argument('--fetch-run', help='candidates: the vernacular_fetch run (vernac-...)')
    vern.add_argument('--taxonomy-run', help='candidates: the taxonomy_fetch run (taxdump-...) whose index places the names')
    vern.add_argument('--match-run', help='candidates: the taxon_match run (taxmatch-...) whose leftover spellings are looked up')
    gene = sub.add_parser('gene', help='Gene normalisation (phase 1): fetch NCBI Gene gene_info as retained raw, build the offline index, match the claims\' literal gene values with program rules conditioned on the taxa layer; no model call')
    gene.add_argument('action', choices=('fetch', 'build', 'match', 'report', 'lookup'))
    gene.add_argument('--gene-run', action='append', help='build/match/lookup: a gene_fetch run (genedump-...); repeat to build or use one index over several runs (plants + fungi + viruses)')
    gene.add_argument('--file', action='append', help='fetch: a gene_info file under base_url (default: source.files of configs/genes-v0.yaml); must be in source.available_files')
    gene.add_argument('--listing', action='append', help='fetch: a directory under base_url whose listing is saved to size its files (source.listings)')
    gene.add_argument('--taxonomy-run', help='build/match/lookup: the taxonomy_fetch run (taxdump-...) that ties genes to species')
    gene.add_argument('--chem-run', help='match: a chem_fetch run (chemdump-...) whose ChEBI index names the parts the gene rules leave untyped (genes-v0.7)')
    gene.add_argument('--taxa-layer', help='match: the taxa annotation layer (ann-...) that gives each claim its organism; or annotations.taxa of --config')
    gene.add_argument('--config', type=Path, help='match: a YAML file whose `layers` (and annotations.taxa) are used unless --layer / --taxa-layer are given')
    gene.add_argument('--layer', action='append', metavar='JOB:DERIVATION', help='match: a (job, derivation) layer read from the extraction store; repeat in merge order')
    gene.add_argument('--run-id', help='report: a gene_match run (genematch-...)')
    gene.add_argument('--force', action='store_true', help='build: rebuild the index even if one exists')
    gene.add_argument('--taxid', type=int, action='append', help='lookup: organism taxid(s) the value is read under')
    gene.add_argument('values', nargs='*', help='lookup: literal gene values to place offline (no run, no record)')
    chem = sub.add_parser('chem', help='Metabolites (phase 2): fetch ChEBI flat files as retained raw, build the offline name index; no model call')
    chem.add_argument('action', choices=('fetch', 'build', 'lookup'))
    chem.add_argument('--run-id', help='build/lookup: the chem_fetch run (chemdump-...)')
    chem.add_argument('--file', action='append', help='fetch: a file under base_url (default: source.files of configs/chem-v0.yaml)')
    chem.add_argument('--listing', action='append', help='fetch: a directory whose listing is saved')
    chem.add_argument('--force', action='store_true', help='build: rebuild the index even if one exists')
    chem.add_argument('names', nargs='*', help='lookup: names to look up offline')
    sub.add_parser('stats', help='Show run/request statistics, never a false corpus size')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == 'snapshot':
        from .snapshot import Snapshot, filter_part
        replay = args.action == 'replay'
        if args.action != 'fetch-manifest' and not args.run_id:
            parser.error('this snapshot action requires --run-id')
        if args.action == 'fetch-manifest' and args.run_id:
            parser.error('fetch-manifest starts a new run')
        run = Run(root, args.run_id, replay, kind='snapshot_filter')
        print(json.dumps({'run_id': run.id, 'manifest': str(run.path), 'replay': replay}), flush=True)
        code = 0
        try:
            snap = Snapshot(run)
            keys = set(args.keys or [])
            if args.key_prefix:
                keys |= {f['key'] for f in run.manifest.get('snapshot_input', {}).get('files', []) if f['key'].startswith(args.key_prefix)}
            if args.action == 'fetch-manifest':
                frozen = snap.fetch_manifest()
                print(json.dumps({k: v for k, v in frozen.items() if k != 'files'}, ensure_ascii=False, indent=2))
                run.finish('manifest_frozen')
            elif args.action == 'download':
                res = snap.download(args.parallel, keys or None)
                print(json.dumps(res, ensure_ascii=False, indent=2))
                code = 0 if res['failed'] == 0 else 2
                run.finish('download_complete' if res['ok'] == len(run.manifest['snapshot_input']['files']) else 'download_partial')
            elif args.action == 'filter':
                res = snap.filter(args.workers, keys or None, args.force)
                print(json.dumps(dict(files=len(res), lines_read=sum(r['lines_read'] for r in res.values()),
                                      matched=sum(r['matched'] for r in res.values())), indent=2))
                run.finish('filter_progress')
            elif args.action == 'index':
                res = snap.index()
                print(json.dumps(res, ensure_ascii=False, indent=2))
                run.artifact('summary.json', snap.summary())
                run.finish('indexed')
            elif args.action == 'replay':
                import tempfile
                with tempfile.TemporaryDirectory() as tmp:
                    res = snap.filter(args.workers, keys or None, force=True, out_root=Path(tmp))
                compared = {k: (res[k]['matches_sha256'] == snap.files.get(k, {}).get('matches_sha256')) for k in res}
                out = dict(files=len(compared), identical=sum(compared.values()), differing=[k for k, v in compared.items() if not v][:20])
                run.artifact('replay-filter.json', out)
                print(json.dumps(out, indent=2))
                code = 0 if out['identical'] == out['files'] else 2
            elif args.action == 'status':
                print(json.dumps(snap.summary(), ensure_ascii=False, indent=2))
            return code
        finally:
            run.close()
    if args.command == 'candidates':
        from .candidates import CandidatePool
        replay = args.action == 'verify'
        if args.action == 'build' and args.run_id:
            parser.error('build starts a new pool run; omit --run-id')
        if args.action != 'build' and not args.run_id:
            parser.error('this action requires --run-id')
        if args.action == 'build' and not args.source_run:
            parser.error('build requires --source-run')
        if args.action == 'build':
            overrides = dict(source_run=args.source_run)
        else:
            stored = root / 'data' / 'runs' / args.run_id / 'manifest.json'
            if not stored.exists():
                parser.error('Run does not exist')
            overrides = json.loads(stored.read_text()).get('config_snapshot', {}).get('candidate_pool', {})
        run = Run(root, args.run_id, replay, kind='candidate_pool', config_overrides=overrides)
        print(json.dumps({'run_id': run.id, 'manifest': str(run.path), 'replay': replay}), flush=True)
        try:
            pool = CandidatePool(run)
            if args.action == 'build':
                res = pool.build()
                print(json.dumps({k: v for k, v in res.items() if k not in ('by_decade',)}, ensure_ascii=False, indent=2))
                run.finish('pool_built')
            elif args.action == 'status':
                print(json.dumps(pool.status(), ensure_ascii=False, indent=2))
            elif args.action == 'ids':
                for oid in pool.ids():
                    print(oid)
            elif args.action == 'verify':
                out = pool.verify()
                print(json.dumps({k: v for k, v in out.items() if k != 'current'}, ensure_ascii=False, indent=2))
                return 0 if out['identical'] else 2
            return 0
        finally:
            run.close()
    if args.command == 'chunks':
        from . import chunks
        if args.action == 'build':
            if args.run_id:
                parser.error('build starts a new chunk plan; omit --run-id')
            run = Run(root, None, False, kind='chunk_plan')
            print(json.dumps({'run_id': run.id, 'manifest': str(run.path)}), flush=True)
            try:
                print(json.dumps(chunks.build(run), ensure_ascii=False, indent=2))
            finally:
                run.close()
        elif not args.run_id:
            parser.error('this action requires --run-id')
        elif args.action == 'verify':
            print(json.dumps(chunks.verify(root, args.run_id), ensure_ascii=False, indent=2))
        else:
            d = root / 'data' / 'runs' / args.run_id
            if not (d / 'summary.json').exists():
                parser.error('Chunk plan not built: ' + args.run_id)
            print((d / 'summary.json').read_text(encoding='utf-8'))
        return
    if args.command == 'sample':
        from .sample import PilotSample
        replay = args.action == 'verify'
        if args.action == 'build' and args.run_id:
            parser.error('build starts a new sample run; omit --run-id')
        if args.action != 'build' and not args.run_id:
            parser.error('this action requires --run-id')
        if args.action == 'build' and not args.pool:
            parser.error('build requires --pool')
        if args.action == 'build':
            overrides = dict(pool=args.pool)
        else:
            stored = root / 'data' / 'runs' / args.run_id / 'manifest.json'
            if not stored.exists():
                parser.error('Run does not exist')
            overrides = dict(pool=json.loads(stored.read_text()).get('config_snapshot', {}).get('pilot_sample', {}).get('pool'))
        run = Run(root, args.run_id, replay, kind='pilot_sample', config_overrides=overrides)
        print(json.dumps({'run_id': run.id, 'manifest': str(run.path), 'replay': replay}), flush=True)
        try:
            sample = PilotSample(run)
            if args.action == 'build':
                res = sample.build()
                print(json.dumps(res, ensure_ascii=False, indent=2))
                run.finish('sample_frozen')
            elif args.action == 'status':
                print(json.dumps(sample.status(), ensure_ascii=False, indent=2))
            elif args.action == 'verify':
                out = sample.verify()
                print(json.dumps({k: v for k, v in out.items() if k != 'current'}, ensure_ascii=False, indent=2))
                return 0 if out['identical'] else 2
            return 0
        finally:
            run.close()
    if args.command == 'store':
        from .store import Store
        store = Store(root, readonly=args.action in ('verify', 'stats'))
        try:
            if args.action == 'init':
                print(json.dumps(dict(store=str(store.dir), counts=store.verify()['counts']), indent=2))
            elif args.action == 'verify':
                out = store.verify()
                print(json.dumps(out, ensure_ascii=False, indent=2))
                return 0 if out['ok'] else 2
            else:
                print(json.dumps(store.stats(), ensure_ascii=False, indent=2))
            return 0
        finally:
            store.close()
    if args.command == 'derive':
        from .derive import Deriver
        from .extract import policy_from_yaml
        from .store import Store
        store = Store(root, readonly=args.action not in ('run', 'drop'))
        try:
            deriver = Deriver(store)
            skey = args.spec
            if args.job_id:
                skey = store.job(args.job_id)['spec_key']
            if args.action == 'list':
                print(json.dumps([{k: v for k, v in d.items() if k != 'policy_json'} for d in deriver.list(store.resolve_spec(skey) if skey else None)], ensure_ascii=False, indent=2))
                return 0
            if args.action == 'drop':
                if not args.derivation:
                    parser.error('drop requires --derivation')
                print(json.dumps(dict(dropped=args.derivation, claim_rows=deriver.drop(args.derivation), note='recomputable at any time with `derive run`'), indent=2))
                return 0
            if not skey:
                parser.error('derive needs --spec or --job-id')
            skey = store.resolve_spec(skey)
            policy = {}
            if args.policy:
                policy = policy_from_yaml(yaml.safe_load(args.policy.read_text()), store.get_spec(skey)['registry']['kinds'])
            if args.action == 'run':
                did = deriver.ensure(skey, policy)
                n = deriver.derive_missing(did)
                print(json.dumps(dict(derivation_id=did, spec_key=skey, policy_keys=sorted(policy), policy_source=str(args.policy) if args.policy else None,
                                      newly_derived=n, stats=deriver.stats(did, args.job_id)), ensure_ascii=False, indent=2))
            else:
                print(json.dumps(deriver.report(args.derivation or Deriver.current_id(skey, policy), args.job_id), ensure_ascii=False, indent=2))
            return 0
        finally:
            store.close()
    if args.command == 'release':
        from .release import Releases
        from .store import Store
        store = Store(root, readonly=args.action in ('list', 'verify', 'export'))
        try:
            rel = Releases(store)
            if args.action == 'list':
                print(json.dumps(rel.list(), ensure_ascii=False, indent=2))
                return 0
            if not args.name:
                parser.error('this action requires --name')
            if args.action == 'create':
                if not args.job_id:
                    parser.error('create requires --job-id')
                print(json.dumps(rel.create(args.name, args.job_id, args.derivation, args.note), ensure_ascii=False, indent=2))
            elif args.action == 'verify':
                out = rel.verify(args.name)
                print(json.dumps(out, ensure_ascii=False, indent=2))
                return 0 if out['identical'] else 2
            else:
                if not args.out:
                    parser.error('export requires --out')
                print(json.dumps(rel.export(args.name, args.out), ensure_ascii=False, indent=2))
            return 0
        finally:
            store.close()
    if args.command == 'annotate':
        from .annotate import Annotations
        from .store import Store
        store = Store(root, readonly=args.action not in ('import-taxa', 'import-genes'))
        try:
            ann = Annotations(store)
            if args.action == 'import-taxa':
                if not args.match_run or not (args.dir or args.packet or args.allow_unjudged):
                    parser.error('import-taxa requires --match-run and --dir (or --packet ..., or --allow-unjudged)')
                from .taxonomy import import_taxa_layer
                packets = ([args.dir] if args.dir else []) + list(args.packet or [])
                print(json.dumps(import_taxa_layer(store, root, args.match_run, packets[0] if len(packets) == 1 else packets, args.name, allow_unjudged=args.allow_unjudged),
                                 ensure_ascii=False, indent=2))
            elif args.action == 'import-genes':
                if not args.match_run:
                    parser.error('import-genes requires --match-run')
                from .genes import import_genes_layer
                print(json.dumps(import_genes_layer(store, root, args.match_run, args.name), ensure_ascii=False, indent=2))
            elif args.action == 'list':
                print(json.dumps([dict(layer_id=L['layer_id'], kind=L['kind'], name=L['name'], created_at=L['created_at'], scope_version=L['vocabulary'].get('scope_version'), **L['stats']) for L in ann.list()],
                                 ensure_ascii=False, indent=2))
            else:
                if not args.layer:
                    parser.error('report requires --layer')
                L = ann.layer(args.layer)
                print(json.dumps(dict(ann.summary(args.layer), stats=L['stats'], source=L['source'], vocabulary=L['vocabulary']), ensure_ascii=False, indent=2))
            return 0
        finally:
            store.close()
    if args.command == 'taxon':
        from .taxonomy import Matcher, Taxonomy, build_index, fetch, index_path, match_run
        if args.action == 'fetch':
            run = Run(root, None, False, kind='taxonomy_fetch')
            print(json.dumps({'run_id': run.id, 'manifest': str(run.path)}), flush=True)
            try:
                print(json.dumps(fetch(run), ensure_ascii=False, indent=2))
            finally:
                run.close()
            return 0
        if args.action in ('build', 'match', 'lookup') and not args.taxonomy_run:
            parser.error('this action requires --taxonomy-run')
        if args.action == 'build':
            print(json.dumps(build_index(root, args.taxonomy_run, args.force), ensure_ascii=False, indent=2))
            return 0
        if args.action == 'match':
            if args.layer:
                if any(':' not in l for l in args.layer):
                    parser.error('--layer takes JOB:DERIVATION')
                layers, layers_from = [dict(zip(('job', 'derivation'), l.split(':', 1))) for l in args.layer], None
            elif args.config:
                config = args.config if args.config.is_absolute() else root / args.config
                layers, layers_from = [dict(job=l['job'], derivation=l['derivation']) for l in yaml.safe_load(config.read_text(encoding='utf-8'))['layers']], str(args.config)
            else:
                parser.error('match needs --layer JOB:DERIVATION (repeatable) or --config')
            run = Run(root, None, False, kind='taxon_match', config_overrides=dict(taxonomy_run=args.taxonomy_run, layers=layers, layers_from=layers_from))
            print(json.dumps({'run_id': run.id, 'manifest': str(run.path)}), flush=True)
            try:
                print(json.dumps(match_run(run), ensure_ascii=False, indent=2))
            finally:
                run.close()
            return 0
        if args.action == 'report':
            if not args.run_id:
                parser.error('report requires --run-id')
            print((root / 'data' / 'runs' / args.run_id / 'summary.json').read_text(encoding='utf-8'))
            return 0
        from .taxonomy import load_config as load_taxonomy_config
        cfg = load_taxonomy_config(root)
        tax = Taxonomy(index_path(root, args.taxonomy_run))
        try:
            matcher = Matcher(tax, cfg['match'])
            for name in args.names:
                print(json.dumps(matcher.resolve(name, args.field, ''), ensure_ascii=False))
        finally:
            tax.close()
        return 0
    if args.command == 'gene':
        from .genes import GeneIndex, build_index as build_gene_index, fetch as gfetch, index_path as gene_index_path, lookup_values, match_run as gene_match_run
        if args.action == 'fetch':
            run = Run(root, None, False, kind='gene_fetch', config_overrides=({k: v for k, v in dict(files=args.file, listings=args.listing).items() if v} or None))
            print(json.dumps({'run_id': run.id, 'manifest': str(run.path)}), flush=True)
            try:
                print(json.dumps(gfetch(run), ensure_ascii=False, indent=2))
            finally:
                run.close()
            return 0
        if args.action == 'report':
            if not args.run_id:
                parser.error('report requires --run-id')
            print((root / 'data' / 'runs' / args.run_id / 'summary.json').read_text(encoding='utf-8'))
            return 0
        if not (args.gene_run and args.taxonomy_run):
            parser.error('this action requires --gene-run and --taxonomy-run')
        gene_runs = '+'.join(args.gene_run)
        if args.action == 'build':
            print(json.dumps(build_gene_index(root, gene_runs, args.taxonomy_run, args.force), ensure_ascii=False, indent=2))
            return 0
        if args.action == 'lookup':
            for row in lookup_values(root, gene_runs, args.taxonomy_run, args.values, args.taxid or []):
                print(json.dumps(row, ensure_ascii=False))
            return 0
        export_cfg = yaml.safe_load((args.config if args.config.is_absolute() else root / args.config).read_text(encoding='utf-8')) if args.config else {}
        if args.layer:
            if any(':' not in l for l in args.layer):
                parser.error('--layer takes JOB:DERIVATION')
            layers, layers_from = [dict(zip(('job', 'derivation'), l.split(':', 1))) for l in args.layer], None
        elif export_cfg.get('layers'):
            layers, layers_from = [dict(job=l['job'], derivation=l['derivation']) for l in export_cfg['layers']], str(args.config)
        else:
            parser.error('match needs --layer JOB:DERIVATION (repeatable) or --config')
        taxa_layer = args.taxa_layer or (export_cfg.get('annotations') or {}).get('taxa')
        if not taxa_layer:
            parser.error('match needs a taxa layer (--taxa-layer or annotations.taxa in --config)')
        run = Run(root, None, False, kind='gene_match', config_overrides=dict(gene_run=gene_runs, taxonomy_run=args.taxonomy_run, taxa_layer=taxa_layer, **({'chem_run': args.chem_run} if args.chem_run else {}),
                                                                              layers=layers, layers_from=layers_from))
        print(json.dumps({'run_id': run.id, 'manifest': str(run.path)}), flush=True)
        try:
            print(json.dumps(gene_match_run(run), ensure_ascii=False, indent=2))
        finally:
            run.close()
        return 0
    if args.command == 'chem':
        from .chem import ChemIndex, build_index as build_chem_index, fetch as cfetch, index_path as chem_index_path
        if args.action == 'fetch':
            run = Run(root, None, False, kind='chem_fetch', config_overrides=dict(files=args.file or [], listings=args.listing or []) if (args.file or args.listing) else dict(listings=[]))
            print(json.dumps({'run_id': run.id, 'manifest': str(run.path)}), flush=True)
            try:
                print(json.dumps(cfetch(run), ensure_ascii=False, indent=2))
            finally:
                run.close()
            return 0
        if not args.run_id:
            parser.error('this action requires --run-id')
        if args.action == 'build':
            print(json.dumps(build_chem_index(root, args.run_id, args.force), ensure_ascii=False, indent=2))
            return 0
        index = ChemIndex(chem_index_path(root, args.run_id))
        try:
            for name in args.names:
                print(json.dumps(dict(name=name, result=index.place(name)), ensure_ascii=False))
        finally:
            index.close()
        return 0
    if args.command == 'vernacular':
        from .vernacular import candidates_run, fetch as vfetch
        if args.action == 'fetch':
            run = Run(root, None, False, kind='vernacular_fetch')
            print(json.dumps({'run_id': run.id, 'manifest': str(run.path)}), flush=True)
            try:
                print(json.dumps(vfetch(run, args.source), ensure_ascii=False, indent=2))
            finally:
                run.close()
            return 0
        if not (args.fetch_run and args.taxonomy_run and args.match_run):
            parser.error('candidates requires --fetch-run, --taxonomy-run and --match-run')
        run = Run(root, None, False, kind='vernacular_candidates', config_overrides=dict(fetch_run=args.fetch_run, taxonomy_run=args.taxonomy_run, match_run=args.match_run))
        print(json.dumps({'run_id': run.id, 'manifest': str(run.path)}), flush=True)
        try:
            print(json.dumps(candidates_run(run), ensure_ascii=False, indent=2))
        finally:
            run.close()
        return 0
    if args.command == 'job':
        from .xjob import Job
        if args.action == 'set-policy' and args.min_core_claims is None:
            parser.error('set-policy requires --min-core-claims')
        if args.min_core_claims is not None:
            if args.action not in ('create', 'set-policy') or not args.audit_reason or not args.audit_reason.strip():
                parser.error('--min-core-claims requires create/set-policy and a nonempty --audit-reason')
        looking = args.action in ('status', 'report')
        if args.action == 'create':
            if args.job_id or not args.selection:
                parser.error('create starts a new job: --selection, no --job-id')
            overrides = dict(model=args.model) if args.model else {}
            if args.effort:
                overrides['effort'] = None if args.effort == 'none' else args.effort
            if args.thinking:
                if args.thinking in ('adaptive', 'none'):
                    overrides['thinking'] = args.thinking
                elif args.thinking.startswith('budget:') and args.thinking[7:].isdigit():
                    overrides['thinking'] = dict(type='enabled', budget_tokens=int(args.thinking[7:]))
                else:
                    parser.error('--thinking takes adaptive, none or budget:N')
            if args.spend_cap is not None:
                overrides['spend_cap_usd'] = args.spend_cap
        else:
            if not args.job_id:
                parser.error('this action requires --job-id')
            overrides = None
        if args.extract_config is not None and args.action != 'create':
            parser.error('--extract-config belongs to job create; an existing job keeps its snapshot')
        run = Run(root, args.job_id, False, kind='extraction_job', config_overrides=overrides, code_lock='components', readonly=looking,
                  extract_config=args.extract_config)
        print(json.dumps({'job_id': run.id, 'manifest': str(run.path)}), flush=True)
        job = None
        try:
            job = Job(run)
            if args.action == 'create':
                ids = [l.strip() for l in args.paper_ids.read_text().splitlines() if l.strip() and not l.startswith('#')] if args.paper_ids else None
                result = job.create(args.selection, 'api', limit=args.limit, cells=args.cell, paper_ids=ids, pool=args.pool, workers=args.workers)
                if args.min_core_claims is not None:
                    result.update(job.set_policy(args.min_core_claims, args.audit_reason))
                print(json.dumps(result, ensure_ascii=False, indent=2))
            elif args.action == 'set-policy':
                print(json.dumps(job.set_policy(args.min_core_claims, args.audit_reason), ensure_ascii=False, indent=2))
            elif args.action == 'submit':       # sends real requests: needs the anthropic SDK, ANTHROPIC_API_KEY and prices in the config
                print(json.dumps(job.submit(max_batches=args.max_batches, limit=args.max_requests), ensure_ascii=False, indent=2))
            elif args.action == 'collect':
                print(json.dumps(job.collect(wait_seconds=args.wait), ensure_ascii=False, indent=2))
            elif args.action == 'run':          # sync mode, same remark as submit
                print(json.dumps(job.run_sync(limit=args.max_requests), ensure_ascii=False, indent=2))
            elif args.action == 'derive':
                did = job.derivation()
                run.flush()
                print(json.dumps(dict(derivation_id=did, stats=job.deriver.stats(did, run.id)), ensure_ascii=False, indent=2))
            elif args.action == 'status':
                print(json.dumps(job.status(), ensure_ascii=False, indent=2))
            elif args.action == 'report':
                print(json.dumps(job.report(), ensure_ascii=False, indent=2))
            return 0
        finally:
            if job is not None:
                job.store.close()
            run.close()
    if args.command == 'extract' and args.action == 'components':
        from .extract import LOCK_FILE, components, components_lock_status
        if args.write_lock:
            LOCK_FILE.write_text(json.dumps(components(), indent=2, sort_keys=True) + '\n')
        print(json.dumps(components_lock_status(), ensure_ascii=False, indent=2))
        return 0 if components_lock_status()['in_sync'] else 2
    if args.command == 'stats':
        path = root / 'data' / 'plantclaims.sqlite'
        if not path.exists():
            print(json.dumps({'runs': [], 'requests': 0, 'corpus_papers': 0}))
        else:
            with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
                has_demo = db.execute("SELECT 1 FROM sqlite_master WHERE name='demo_papers'").fetchone()
                print(json.dumps({'runs': db.execute('SELECT run_id,status FROM runs').fetchall(),
                                  'requests': db.execute('SELECT count(*) FROM requests_log').fetchone()[0],
                                  'corpus_papers': 0,
                                  'demo_paper_memberships': db.execute('SELECT count(*) FROM demo_papers').fetchone()[0] if has_demo else 0,
                                  'note': 'P0 fixtures and unreviewed demos are not formal corpus membership'}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Run records: immutable raw responses, resumable run manifests with a config snapshot, and a SQLite request ledger."""

import hashlib
import json
import platform
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    data = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def code_ref(root):
    """Git commit of the working tree that created a run, or None when the root is not a git checkout. `dirty` says
    whether plantclaims_pipeline/ or configs/ differed from that commit, so a manifest never claims a commit it was not run from."""
    try:
        def git(*args):
            return subprocess.run(('git', '-C', str(root)) + args, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
        if git('rev-parse', '--show-toplevel') != str(Path(root).resolve()):
            return None     # the root only sits inside some other repository
        return dict(commit=git('rev-parse', 'HEAD'), dirty=bool(git('status', '--porcelain', '--', 'plantclaims_pipeline', 'configs')))
    except (OSError, subprocess.SubprocessError):
        return None


def configs(root):
    return {}


class Run:
    def __init__(self, root, run_id=None, replay=False, kind=None, config_overrides=None, variant=None, code_lock='full', readonly=False,
                 extract_config=None):
        """code_lock='full' (every kind, the default): an existing run opens only with byte-identical package code and config.
        code_lock='components' (abstract_extraction and extraction_job): an existing run opens with its own stored config snapshot and
        without the whole-package comparison; plantclaims_pipeline.extract then checks, per action, only the component the action
        depends on (request builder before sending, validator before ingesting). readonly=True never writes the manifest
        or the runs table, so looking at a run does not touch it.
        extract_config: another extraction config file than configs/extract-v0.yaml, e.g.
        configs/extract-v0.12.yaml (prompt v0.15); the run snapshots it like the default and names the file in `extract_config_file`."""
        if code_lock not in ('full', 'components') or (code_lock == 'components' and kind != 'extraction_job'):
            raise ValueError('code_lock=components is defined for extraction_job runs only')
        if extract_config is not None and kind != 'extraction_job':
            raise ValueError('extract_config is defined for extraction runs only')
        self.readonly = readonly
        self.root = Path(root).resolve()
        self.config = configs(self.root)
        if kind not in ('snapshot_filter', 'candidate_pool', 'pilot_sample', 'extraction_job', 'taxonomy_fetch', 'taxon_match',
                        'vernacular_fetch', 'vernacular_candidates', 'gene_fetch', 'gene_match', 'chem_fetch', 'chunk_plan'):
            raise ValueError('Unknown run kind')
        if kind == 'snapshot_filter':
            self.config['snapshot'] = yaml.safe_load((self.root / 'configs' / 'snapshot-v0.yaml').read_text())
            self.config['scope'] = yaml.safe_load((self.root / 'configs' / 'scope-v0.6.yaml').read_text())
            for key, value in (config_overrides or {}).items():
                self.config['snapshot'][key] = value
        if kind == 'candidate_pool':
            # Selection stage: binds the scope and the source snapshot run; both frozen in the manifest.
            self.config['scope'] = yaml.safe_load((self.root / 'configs' / 'scope-v0.6.yaml').read_text())
            self.config['candidate_pool'] = dict(config_overrides or {})
        if kind == 'pilot_sample':
            # Stratified seeded draw from a built candidate pool; the pool id and any sampling parameters are frozen.
            self.config['pilot_sample'] = yaml.safe_load((self.root / 'configs' / 'pilot-sample-v0.yaml').read_text())
            self.config['scope'] = yaml.safe_load((self.root / 'configs' / 'scope-v0.6.yaml').read_text())
            for key, value in (config_overrides or {}).items():
                self.config['pilot_sample'][key] = value
        if kind == 'extraction_job':
            # Merged abstract-level run: relevance-v0.3 rules (verbatim) + claims-v0.9 + paper profile, one paper per request.
            # extraction_job (prefix xjob-) binds the same configs; its inputs and answers live in the extraction store
            # (plantclaims_pipeline/store.py), so its run directory holds a small manifest only.
            if extract_config is not None and run_id is not None:
                raise ValueError('extract_config selects the config of a NEW run; an existing run keeps its snapshot')
            config_file = Path(extract_config) if extract_config is not None else Path('configs') / 'extract-v0.12.yaml'
            if config_file.is_absolute():
                config_file = config_file.resolve().relative_to(self.root)
            if not (self.root / config_file).is_file() or not config_file.name.startswith('extract-v0'):
                raise ValueError(f'Not an extraction config: {config_file}')
            cfg = yaml.safe_load((self.root / config_file).read_text())
            for key, value in (config_overrides or {}).items():
                cfg[key] = value
            self.config['extract'] = cfg
            if config_file != Path('configs') / 'extract-v0.12.yaml':
                self.config['extract_config_file'] = str(config_file)
            self.config['claims'] = yaml.safe_load((self.root / cfg['claims_schema']).read_text())
            self.config['scope'] = yaml.safe_load((self.root / 'configs' / 'scope-v0.6.yaml').read_text())
            if cfg.get('bucket_scope'):
                # Research areas (bucket labels + bucket_rules) come from the scope file named by extract.bucket_scope.
                self.config['bucket_scope'] = yaml.safe_load((self.root / cfg['bucket_scope']).read_text())
            self.config['relevance_v0_3'] = yaml.safe_load((self.root / cfg['relevance_config']).read_text())
            self.config['relevance_prompt_v0_3'] = (self.root / cfg['relevance_prompt']).read_text()
            self.config['extract_prompt'] = (self.root / cfg['prompt']).read_text()
        if kind in ('taxonomy_fetch', 'taxon_match'):
            # Taxon normalisation: the fetched NCBI dump is retained raw; a match run binds the fetch run and the site export config it reads.
            from .taxonomy import load_config as load_taxonomy_config        # inlines the common-name table (taxonomy-v0.3) into the snapshot
            self.config['taxonomy'] = load_taxonomy_config(self.root)
            if kind == 'taxon_match':
                self.config['taxon_match'] = dict(config_overrides or {})
        if kind in ('vernacular_fetch', 'vernacular_candidates'):
            self.config['vernacular'] = yaml.safe_load((self.root / 'configs' / 'vernacular-v0.yaml').read_text())
            if kind == 'vernacular_candidates':
                self.config['vernacular_candidates'] = dict(config_overrides or {})
        if kind in ('gene_fetch', 'gene_match'):
            # Gene normalisation: NCBI Gene's gene_info retained raw; a match run binds the gene fetch run, the taxonomy
            # backbone, the taxa layer that conditions the lookup and the (job, derivation) layers it reads.
            self.config['genes'] = yaml.safe_load((self.root / 'configs' / 'genes-v0.yaml').read_text())
            if kind == 'gene_match':
                self.config['gene_match'] = dict(config_overrides or {})
            elif config_overrides:                   # gene_fetch: the files (and directory listings) this run downloads
                self.config['gene_fetch'] = dict(config_overrides)
        if kind == 'chem_fetch':
            # Metabolites: ChEBI flat files retained raw; the run's own choice of files/listings is part of its config.
            self.config['chem'] = yaml.safe_load((self.root / 'configs' / 'chem-v0.yaml').read_text())
            if config_overrides:
                self.config['chem_fetch'] = dict(config_overrides)
        if kind == 'chunk_plan':
            self.config['chunking'] = yaml.safe_load((self.root / 'configs' / 'chunking-v0.yaml').read_text())
            self.config['scope'] = yaml.safe_load((self.root / 'configs' / 'scope-v0.6.yaml').read_text())
        self.code_hashes = {p.name: digest(p.read_bytes()) for p in sorted((self.root / 'plantclaims_pipeline').glob('*.py'))}
        self.config_hash = digest(self.config)
        prefix = {'snapshot_filter': 'snapshot-', 'candidate_pool': 'pool-', 'pilot_sample': 'sample-', 'extraction_job': 'xjob-',
                  'taxonomy_fetch': 'taxdump-', 'taxon_match': 'taxmatch-', 'vernacular_fetch': 'vernac-', 'vernacular_candidates': 'vcand-',
                  'gene_fetch': 'genedump-', 'gene_match': 'genematch-', 'chem_fetch': 'chemdump-', 'chunk_plan': 'chunks-'}[kind]
        self.id = run_id or (prefix + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        if not self.id.startswith(prefix) or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in self.id):
            raise ValueError('Invalid run ID')
        self.replay = replay
        self.directory = self.root / 'data' / 'runs' / self.id
        self.raw = self.root / 'data' / 'raw' / self.id
        self.path = self.directory / 'manifest.json'
        if run_id and not self.path.exists():
            raise ValueError('Run does not exist; omit --run-id to create one')
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text())
            if replay or code_lock == 'components':
                self.config = self.manifest['config_snapshot']     # a run's config is the one it was created with
                self.config_hash = digest(self.config)
                if self.config_hash != self.manifest['config_hash']:
                    raise ValueError('Stored config snapshot hash mismatch')
            elif self.manifest['config_hash'] != self.config_hash:
                raise ValueError('Config changed: start a new run, never resume with different config')
            if not replay and code_lock == 'full' and self.manifest['code_hashes'] != self.code_hashes:
                raise ValueError('Code changed: start a new run; original raw responses remain immutable')
        else:
            self.directory.mkdir(parents=True)
            self.raw.mkdir(parents=True)
            self.manifest = dict(run_id=self.id, kind=kind, created_at=now(), status='created',
                                 config_hash=self.config_hash, config_snapshot=self.config,
                                 code_hashes=self.code_hashes, code_ref=code_ref(self.root), python=platform.python_version(),
                                 dependencies={'requests': requests.__version__, 'PyYAML': yaml.__version__},
                                 attempts=[], checks=[], fixtures_are_corpus=False)
        dbpath = self.root / 'data' / 'plantclaims.sqlite'
        self.db = sqlite3.connect(dbpath, timeout=60)       # wait for a busy writer instead of failing at once
        try:
            self.db.execute('PRAGMA journal_mode=WAL')
        except sqlite3.OperationalError:
            pass                                             # another connection holds the file right now; mode stays as it is
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
              config_hash TEXT NOT NULL, created_at TEXT NOT NULL, manifest_path TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests_log (
              run_id TEXT NOT NULL REFERENCES runs(run_id), seq INTEGER NOT NULL,
              step TEXT NOT NULL, request_hash TEXT NOT NULL, details TEXT NOT NULL,
              PRIMARY KEY(run_id,seq));
        ''')
        if not replay:
            self.flush()

    def flush(self):
        if self.replay or self.readonly:
            return
        self.manifest['updated_at'] = now()
        save_json(self.path, self.manifest)
        with self.db:
            self.db.execute('INSERT INTO runs VALUES (?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET status=excluded.status',
                            (self.id, self.manifest['kind'], self.manifest['status'], self.config_hash,
                             self.manifest['created_at'], str(self.path.relative_to(self.root))))
            for a in self.manifest['attempts']:
                self.db.execute('INSERT OR REPLACE INTO requests_log VALUES (?,?,?,?,?)',
                                (self.id, a['seq'], a['step'], a['request_hash'], json.dumps(a)))

    def response(self, step, request_hash):
        for a in reversed(self.manifest['attempts']):
            if a['step'] == step and a['request_hash'] == request_hash and a.get('status') == 200 and not a.get('error'):
                data = (self.root / a['raw_path']).read_bytes()
                if digest(data) != a['sha256']:
                    raise ValueError('Raw response hash mismatch')
                return data
        return None

    def artifact(self, name, value):
        if self.replay:
            target = self.root / 'data' / 'replays' / self.id / digest(self.code_hashes)[:16]
            save_json(target / name, value)
            save_json(target / 'provenance.json', {'source_run_id': self.id, 'mode': 'offline_replay',
                      'source_code_hashes': self.manifest['code_hashes'], 'analysis_code_hashes': self.code_hashes,
                      'config_hash': self.config_hash, 'source_manifest_sha256': digest(self.path.read_bytes())})
        else:
            save_json(self.directory / name, value)

    def finish(self, status):
        if not self.replay:
            self.manifest['status'] = status
            self.manifest['finished_at'] = now()
            self.flush()

    def close(self):
        self.db.close()

"""Regenerate current source locks from an exact public VCS core installation.

Run in the qualification environment after installing the reviewed core commit.
Historical backend-s081.lock.json and s081-profile.json are never rewritten.
"""
import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path


def closure(package):
    module = importlib.import_module(package)
    root = Path(module.__file__).resolve().parent
    return {str(path.relative_to(root.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*')) if path.is_file() and '__pycache__' not in path.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core-commit', required=True)
    args = parser.parse_args()
    core = metadata.distribution('particle-sliders-core')
    direct = json.loads(core.read_text('direct_url.json') or '{}')
    if direct.get('vcs_info', {}).get('commit_id') != args.core_commit or core.version != '0.13.0':
        raise ValueError('install the exact reviewed public VCS core0.13.0 commit first')
    if metadata.version('particlegan') != '0.9.5':
        raise ValueError('install released ParticleGAN0.9.5 first')
    root = Path(__file__).resolve().parents[1]
    sources = closure('particle_sliders') | closure('concept_slider_core')
    pg_sources = closure('particlegan')
    for relative in ('src/iris_sliders/stable-profile.json', 'src/iris_sliders_hndl/profile.json'):
        path = root / relative
        document = json.loads(path.read_text())
        document.update(core_commit=args.core_commit, core_version=core.version, particlegan_version='0.9.5',
                        core_full_sources_sha256=sources, pg_full_sources_sha256=pg_sources)
        if 'core_sources_sha256' in document:
            document['core_sources_sha256'] = {name: sources['particle_sliders/' + name]
                                                for name in document['core_sources_sha256']}
        path.write_text(json.dumps(document, indent=2) + '\n')
    path = root / 'backend.lock.json'
    lock = json.loads(path.read_text())
    for key in ('wheel_sha256', 'canonical_receipt_sha256'):
        lock['core'].pop(key, None)
    lock['core'].update(commit=args.core_commit, version=core.version, source_sha256=sources,
                        source_map_scope='all installed non-cache core package source/data files')
    lock['particlegan'] = dict(repository='https://github.com/255BITS/ParticleGAN',
        commit='1b8355b184b21bb7d81104ee590813c262f6df0a', version='0.9.5',
        release='https://github.com/255BITS/ParticleGAN/releases/tag/v0.9.5',
        source_manifest='particle_sliders/pg095_release.json', source_sha256=pg_sources,
        role='core transitive dependency; exact installed release source closure verified')
    lock['network_profile'] = 'native-pg095-v1'
    lock['historical_environment'] = 'requirements-s081.txt; backend-s081.lock.json'
    path.write_text(json.dumps(lock, indent=2) + '\n')


if __name__ == '__main__':
    main()

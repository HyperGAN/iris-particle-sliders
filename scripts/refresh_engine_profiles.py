"""Regenerate current source locks from an exact public VCS core installation.

Run in the qualification environment after installing the reviewed core commit.
Historical runtime locks live at the archived source revision.
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
    if direct.get('vcs_info', {}).get('commit_id') != args.core_commit or core.version != '0.15.2':
        raise ValueError('install the exact reviewed public VCS core0.15.2 commit first')
    if metadata.version('particlegan') != '0.11.1':
        raise ValueError('install released ParticleGAN0.11.1 first')
    root = Path(__file__).resolve().parents[1]
    sources = closure('particle_sliders') | closure('concept_slider_core')
    pg_sources = closure('particlegan')
    for relative in ('src/iris_sliders/stable-profile.json', 'src/iris_sliders_hndl/profile.json'):
        path = root / relative
        document = json.loads(path.read_text())
        document.update(core_commit=args.core_commit, core_version=core.version, particlegan_version='0.11.1',
                        core_full_sources_sha256=sources, pg_full_sources_sha256=pg_sources)
        if "core_sources_sha256" in document:
            document["profile"] = "hypergan-hndl-engine-v1"
            document["core_sources_sha256"] = {
                name: sources["particle_sliders/" + name]
                for name in ("engine.py", "engine_release.json", "backends.py", "reference.py", "lifecycle.py", "inference.py")
            }
            manifest = root / "src/iris_sliders_hndl/fibo_hndl_010_sources.json"
            document["network_identity"]["runtime_sources_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        path.write_text(json.dumps(document, indent=2) + '\n')
    path = root / 'backend.lock.json'
    lock = json.loads(path.read_text())
    for key in ('wheel_sha256', 'canonical_receipt_sha256'):
        lock['core'].pop(key, None)
    lock["core"].update(entrypoint="engine.ConditionalParticleGAN", runner="ConditionalParticleGAN",
                        native_backend="iris_sliders.image_bindings.IrisResponseAdapter")
    lock['core'].update(commit=args.core_commit, version=core.version, source_sha256=sources,
                        source_map_scope='all installed non-cache core package source/data files')
    lock['particlegan'] = dict(repository='https://github.com/255BITS/ParticleGAN',
        commit='d5dec2fe005305082acf382f0939a4bdb51c7131', version='0.11.1',
        release='https://github.com/255BITS/ParticleGAN/releases/tag/v0.11.1',
        source_manifest='particle_sliders/engine_release.json', source_sha256=pg_sources,
        role='core transitive dependency; exact installed release source closure verified')
    lock['network_profile'] = 'native-engine-v1'
    if 'optional_observation' in lock and 'shared_api' in lock['optional_observation']:
        lock['optional_observation']['shared_api'] = 'iris_sliders.hypergan_diagnostics.response_diversity'
    lock['historical_environment'] = ('https://github.com/HyperGAN/iris-particle-sliders/tree/'
                                       '7236ede009983b9fdbe020c5eed7c65a466ba408')
    path.write_text(json.dumps(lock, indent=2) + '\n')


if __name__ == '__main__':
    main()

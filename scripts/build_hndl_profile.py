"""Build a separate opt-in Iris HyperGAN/HNDL distribution; never publish it.

The default pyproject installs the current native PG094 profile. The emitted sdist is complete
and contains its own exact profile pyproject, so wheel rebuilds retain the pins.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path


def stage(root, destination):
    profile = json.loads((root / 'src/iris_sliders_hndl/profile.json').read_text())
    text = (root / 'pyproject.toml').read_text()
    config = tomllib.loads(text)
    old = next(d for d in config['project']['dependencies'] if d.startswith('particle-sliders-core @'))
    core = profile['core_commit']
    new = ('particle-sliders-core[fibo-hndl] @ git+https://github.com/HyperGAN/particle-sliders.git@'
           + core + '#subdirectory=packages/particle-sliders-core')
    if text.count(old) != 1:
        raise ValueError('ambiguous core dependency')
    text = text.replace(old, new).replace('version = "0.3.0"', 'version = "0.4.0rc1"')
    text = text.replace('name = "iris-particle-sliders"', 'name = "iris-particle-sliders-hndl"')
    text = text.replace('readme = "README.md"', 'readme = "HYPERGAN-HNDL.md"')
    text = text.replace('iris-slider-train-legacy = "iris_sliders.legacy_train:main"\n', '')
    text = text.replace('exclude = ["iris_sliders_hndl*"]', 'exclude = []')
    text = text.replace('iris_sliders = ["*.json"]', 'iris_sliders = ["*.json"]\niris_sliders_hndl = ["*.json"]')
    text = text.replace('requires-python = ">=3.11"', 'requires-python = ">=3.11,<3.15"')
    text = text.replace('hndl = ["hndl==0.8.2"]', 'hndl = ["hndl==0.10.0"]')
    text = text.replace('a879c4fbff903ccefeba2194bcc7a13fe7b5abb7', profile['hypergan_commit'])
    for directory in ('src', 'scripts', 'tests', 'configs'):
        shutil.copytree(root / directory, destination / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', '*.pyc'))
    for path in root.iterdir():
        if path.is_file() and (path.suffix in ('.md', '.json', '.toml', '.txt')
                               or path.name in ('LICENSE', 'NOTICE', 'MANIFEST.in', 'setup.py')):
            shutil.copy2(path, destination / path.name)
    # Keep the historical lock verbatim; the profile archive's active lock must
    # describe its actual dependencies rather than repeating S081 provenance.
    shutil.copy2(root / 'backend-s081.lock.json', destination / 'backend-s081.lock.json')
    lock = json.loads((root / 'backend.lock.json').read_text())
    lock['core'] = dict(repository=lock['core']['repository'], commit=core,
                        subdirectory='packages/particle-sliders-core', version=profile['core_version'],
                        source_sha256=profile['core_full_sources_sha256'],
                        source_map_scope='all installed non-cache core package source/data files; shared builder closure verified separately')
    lock['particlegan'] = dict(repository='https://github.com/255BITS/ParticleGAN',
                              version=profile['particlegan_version'],
                              source_manifest='particle_sliders/pg094_release.json')
    lock['network_profile'] = profile
    lock.pop('optional_reuse', None)
    (destination / 'backend.lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    with (destination / 'MANIFEST.in').open('a') as manifest:
        manifest.write('\ninclude backend-s081.lock.json\n')
    (destination / 'pyproject.toml').write_text(text)
    metadata = tomllib.loads(text)['project']
    if (metadata['version'], metadata['requires-python']) != ('0.4.0rc1', '>=3.11,<3.15'):
        raise ValueError('profile version/Python metadata did not bind')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.outdir.resolve()
    with tempfile.TemporaryDirectory(prefix='iris-hndl-build-') as temporary:
        destination = Path(temporary)
        stage(root, destination)
        subprocess.run([sys.executable, '-m', 'build', '--no-isolation', '--outdir', str(output),
                        str(destination)], check=True)


if __name__ == '__main__':
    main()

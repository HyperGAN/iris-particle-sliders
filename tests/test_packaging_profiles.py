"""Two complete metadata cohorts; default source metadata must stay supported."""
import importlib.util
import json
import tomllib
from pathlib import Path


def test_profile_staging_preserves_default_and_binds_exact_dependencies(tmp_path):
    root = Path(__file__).resolve().parents[1]
    path = root / 'scripts/build_hndl_profile.py'
    spec = importlib.util.spec_from_file_location('iris_profile_build', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = (root / 'pyproject.toml').read_bytes()
    module.stage(root, tmp_path)
    assert (root / 'pyproject.toml').read_bytes() == before
    stable = tomllib.loads(before.decode())['project']
    modern = tomllib.loads((tmp_path / 'pyproject.toml').read_text())['project']
    assert stable['version'] == '0.2.0'
    assert '32f6830f6402e85aa77b6631db64fcbbb2131f6b' in next(
        d for d in stable['dependencies'] if d.startswith('particle-sliders-core '))
    assert modern['version'] == '0.3.0rc1' and modern['requires-python'] == '>=3.11,<3.15'
    profile = json.loads((tmp_path / 'src/iris_sliders/hndl-profile.json').read_text())
    assert profile['core_commit'] in next(d for d in modern['dependencies'] if d.startswith('particle-sliders-core[fibo-hndl] '))
    assert modern['optional-dependencies']['hndl'] == ['hndl==0.10.0']
    assert profile['hypergan_commit'] in modern['optional-dependencies']['diagnostics'][0]
    assert json.loads((tmp_path / 'src/iris_sliders/installed-profile.json').read_text()) == {
        'network_profile': 'hypergan-hndl-v2'}
    assert 'a191c999e931dc9bd7d1f9dcb8f2c200048d342d' in (tmp_path / 'requirements-s081.txt').read_text()
    assert (tmp_path / 'HYPERGAN-HNDL.md').is_file()

    assert (tmp_path / 'backend-s081.lock.json').read_bytes() == (root / 'backend.lock.json').read_bytes()
    lock = json.loads((tmp_path / 'backend.lock.json').read_text())
    assert lock['core']['commit'] == profile['core_commit']
    assert lock['particlegan']['version'] == '0.9.4'
    assert lock['network_profile']['hypergan_commit'] == profile['hypergan_commit']
    assert 'canonical_receipt_sha256' not in lock['core']

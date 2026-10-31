"""Installed distribution ownership and exact source admission, before model work.

Admission is deliberately uncached: metadata, file closure and actual imported
module origins are checked on each public entry. No mutable admission token is
shared with callers or retained across package/file changes.
"""
import hashlib
import importlib
import importlib.util
from importlib import metadata
from importlib.resources import files
import json
from pathlib import Path
import sys

STABLE = 'iris-particle-sliders'
MODERN = 'iris-particle-sliders-hndl'
PROFILE = 'hypergan-hndl-v2'


def installation():
    owners = []
    for name in (STABLE, MODERN):
        try:
            owners.append(metadata.distribution(name))
        except metadata.PackageNotFoundError:
            pass
    if len(owners) != 1:
        raise RuntimeError('install exactly one Iris distribution in a fresh environment; mixed profiles are refused')
    dist = owners[0]
    own_init = Path(dist.locate_file('iris_sliders/__init__.py')).resolve()
    if own_init != Path(__file__).resolve().with_name('__init__.py'):
        raise RuntimeError('Iris import origin differs from installed distribution ownership')
    modern = dist.metadata['Name'].lower() == MODERN
    prefixes = ('iris_sliders/', 'iris_sliders_hndl/') if modern else ('iris_sliders/',)
    # RECORD hashes verify the package actually imported, not just a version label.
    recorded = {str(p): p for p in (dist.files or ()) if str(p).startswith(prefixes)
                and Path(str(p)).suffix in ('.py', '.json')}
    actual = {}
    for prefix in prefixes:
        package_root = own_init.parent.parent / prefix.rstrip('/')
        for path in package_root.rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.json'):
                actual[str(path.relative_to(own_init.parent.parent))] = path
        spec = importlib.util.find_spec(prefix.rstrip('/'))
        if spec is None or Path(spec.origin).resolve() != package_root / '__init__.py':
            raise RuntimeError('Iris addon import origin differs from distribution ownership')
    if set(actual) != set(recorded):
        raise RuntimeError('Iris package files differ from installed profile ownership')
    import base64
    for name, path in actual.items():
        expected = recorded[name].hash
        digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b'=').decode()
        if expected is None or expected.mode != 'sha256' or digest != expected.value:
            raise RuntimeError('Iris installed package bytes differ from distribution RECORD')
    if not modern and importlib.util.find_spec('iris_sliders_hndl') is not None:
        raise RuntimeError('modern Iris modules coexist with stable distribution; use a fresh environment')
    return dist, PROFILE if modern else 'legacy'


def default_profile():
    return installation()[1]


def select_profile(requested=None):
    installed = default_profile()
    if requested is not None and requested != installed:
        raise ValueError('requested network profile differs from the installed Iris distribution')
    return installed


def profile_document():
    if default_profile() == 'legacy':
        return json.loads(files('iris_sliders').joinpath('stable-profile.json').read_text())
    return json.loads(files('iris_sliders_hndl').joinpath('profile.json').read_text())


def _origin(package, distribution):
    module = importlib.import_module(package)
    dist = metadata.distribution(distribution)
    root = Path(dist.locate_file(package)).resolve()
    if Path(module.__file__).resolve() != root / '__init__.py':
        raise RuntimeError(package + ' actual import origin differs from installed distribution')
    for name, loaded in tuple(sys.modules.items()):
        if name == package or name.startswith(package + '.'):
            path = getattr(loaded, '__file__', None)
            if path is not None and not Path(path).resolve().is_relative_to(root):
                raise RuntimeError(name + ' actual import origin is outside verified package')
    return dist, root


def verify_runtime():
    document = profile_document()
    core_dist, core_root = _origin('particle_sliders', 'particle-sliders-core')
    direct = json.loads(core_dist.read_text('direct_url.json') or '{}')
    if (core_dist.version != document['core_version'] or
            direct.get('vcs_info', {}).get('commit_id') != document['core_commit']):
        raise RuntimeError('Iris requires the exact public core VCS commit and version')
    _origin('concept_slider_core', 'particle-sliders-core')
    parent = core_root.parent
    actual = {}
    for package in ('particle_sliders', 'concept_slider_core'):
        for path in (parent / package).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:
                actual[str(path.relative_to(parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != document['core_full_sources_sha256']:
        raise RuntimeError('Iris full core source closure differs from exact public candidate')
    pg_dist, pg_root = _origin('particlegan', 'particlegan')
    actual_pg = {str(path.relative_to(pg_root.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in pg_root.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    if pg_dist.version != document['particlegan_version'] or actual_pg != document['pg_full_sources_sha256']:
        raise RuntimeError('Iris actual ParticleGAN version/source bytes differ from admitted release')
    # Execute the verified shared admission rather than advertising a hardcoded PG version.
    from particle_sliders.pg09 import validate_pg09_domain
    validate_pg09_domain((1, 3, 16, 16), response_domain='continuous_field')
    return document


def adapter_bank(model, stamp, profile=None):
    profile = select_profile(profile)
    verify_runtime()
    if profile == 'legacy':
        from .adapters import AdapterBank
        return AdapterBank(model, stamp)
    from iris_sliders_hndl import adapter_bank as modern_bank
    return modern_bank(model, stamp, profile)

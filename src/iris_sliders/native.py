"""Native-only Iris adapter construction and raw velocity observation.

No training factory, teacher collection, game, critic or optimizer is imported.
This is not a denoising sampler or an image rendering implementation.
"""
from contextlib import contextmanager
from dataclasses import replace
import math
from pathlib import Path

import torch
from particle_sliders import winning_formulation
from particle_sliders.inference import InferenceBackend, InferenceBinding, native_identity
from particle_sliders.lifecycle import Capabilities, UnsupportedOperation, native_cleanup
from particle_sliders.pg09_backends import ResponseRow
from particle_sliders.pg09_images import IrisResponseAdapter

from .adapters import AdapterBank, new_particles
from .card import IRIS_COMMIT, MODEL_REVISION, TEXT_REVISION
from .checkpoint import config_digest

RUNTIME_KEYS = {'checkpoint', 'text_dir', 'cache_dir', 'device', 'text_device'}
CACHE_DIR = '/ml2/iris-particle-sliders/cache'
PROTOCOL = 'iris-native-rgb-velocity-v1'
CAPABILITIES = Capabilities(
    'routed_particle_gan', render=False, base_precisions=('float32',), render_outputs=(),
    operations=('train', 'resume', 'evaluate', 'export', 'response'),
    unavailable_reasons=(('render', 'Iris native PG rendering requires FlowDPMSolver parity'),),
)


def validate_runtime(options):
    """Check local option shapes without loading weights or selecting a device."""
    if type(options.get('dummy', False)) is not bool:
        raise ValueError('Iris dummy must be a boolean')
    if options.get('base_precision', 'float32') != 'float32':
        raise ValueError('Iris base weights are float32; mixed base precision is unsupported')
    for key in ('device', 'text_device'):
        value = options.get(key, 'cpu')
        try:
            device = torch.device(value) if isinstance(value, str) else None
        except (RuntimeError, ValueError, TypeError) as exc:
            raise ValueError(key + ' must be cpu or a CUDA device') from exc
        if device is None or device.type not in ('cpu', 'cuda') or (device.type == 'cpu' and value != 'cpu'):
            raise ValueError(key + ' must be cpu or a CUDA device')
        if key == 'text_device' and device.type == 'cuda':
            raise UnsupportedOperation('separate CUDA text_device requires shared RNG device coverage')
    if options.get('dummy'):
        if (options.get('device', 'cpu') != 'cpu' or options.get('text_device', 'cpu') != 'cpu'
                or any(options.get(key) is not None for key in ('checkpoint', 'text_dir'))):
            raise ValueError('Iris dummy accepts only the CPU tiny fixture, without native model paths')
    else:
        cache_dir = options.get('cache_dir', CACHE_DIR)
        if not isinstance(cache_dir, str) or not cache_dir:
            raise ValueError('native Iris requires a local cache_dir under /ml2')
        cache_path = Path(cache_dir).resolve()
        if cache_path != Path('/ml2') and Path('/ml2') not in cache_path.parents:
            raise ValueError('native Iris cache_dir must be under /ml2')
        for key in ('checkpoint', 'text_dir'):
            value = options.get(key)
            if value is not None and (not isinstance(value, str) or not Path(value).is_dir()):
                raise ValueError(key + ' must be an existing local directory')


def load_runtime(options, seed):
    if options.get('dummy'):
        from .tiny import TinyRuntime
        return TinyRuntime(seed)
    from .runtime import IrisRuntime
    native = {key: value for key, value in options.items() if key in RUNTIME_KEYS}
    native.setdefault('cache_dir', CACHE_DIR)
    return IrisRuntime(allow_hub=False, **native)


def model_identity(runtime):
    return dict(iris_commit=IRIS_COMMIT, model_revision=MODEL_REVISION,
                text_revision=TEXT_REVISION, base_sha256=runtime.base_sha256,
                model_config_sha256=config_digest(runtime.raw['model']))


def admitted(native):
    """Pure ownership/precision decision over already-loaded native objects."""
    return (all(not p.requires_grad and p.dtype == torch.float32 for p in native.runtime.model.parameters())
            and all(p.requires_grad and p.dtype == torch.float32 for p in native.network.parameters())
            and {id(p) for p in native.adapter.parameters()} | {id(native.particles)}
            == {id(p) for p in native.network.parameters()})


@contextmanager
def observation(runtime):
    """Restore Iris model buffers/flags and its prompt cache without copying base weights."""
    restore = native_cleanup(runtime.model)
    modules = [(module, dict(module._buffers), set(module._non_persistent_buffers_set))
               for module in runtime.model.modules()]
    buffers = {id(value): (value, value.detach().clone())
               for _, registered, _ in modules for value in registered.values() if value is not None}
    cache = getattr(runtime, 'encoding_cache', None)
    cached = dict(cache) if cache is not None else None
    try:
        runtime.model.eval()
        yield
    finally:
        try:
            with torch.no_grad():
                for value, saved in buffers.values():
                    if (value.shape, value.dtype, value.device) == (saved.shape, saved.dtype, saved.device):
                        value.copy_(saved)
                    else:
                        value.data = saved.clone()
                for module, registered, nonpersistent in modules:
                    module._buffers.clear()
                    module._buffers.update(registered)
                    module._non_persistent_buffers_set.clear()
                    module._non_persistent_buffers_set.update(nonpersistent)
            if cache is not None:
                cache.clear()
                cache.update(cached)
                runtime.encoding_cache = cache
        finally:
            restore()


def validate_inference(options, spec):
    if set(options) - (RUNTIME_KEYS | {'base_precision'}):
        raise ValueError('native Iris inference accepts only local runtime options')
    architecture = spec.configuration
    if (set(architecture) != {'dummy', 'seed'} or type(architecture['dummy']) is not bool
            or type(architecture['seed']) is not int or not 0 <= architecture['seed'] < 2**63 - 100):
        raise ValueError('invalid Iris native adapter configuration')
    validate_runtime(dict(options, dummy=architecture['dummy']))


def preflight_inference(options, request, spec):
    native = request['native']
    if set(native) - {'prompt', 'height', 'width', 'timestep', 'cfg_scale'}:
        raise ValueError('unknown Iris raw velocity request argument')
    if not isinstance(native.get('prompt', ''), str):
        raise ValueError('Iris prompt must be text')
    height, width = native.get('height', 16 if spec.configuration['dummy'] else 512), native.get(
        'width', 16 if spec.configuration['dummy'] else 512)
    if type(height) is not int or type(width) is not int:
        raise ValueError('Iris height and width must be integers')
    from .runtime import validate_shape
    validate_shape(height, width)
    for key, default, maximum in (('timestep', 500., 1000.), ('cfg_scale', 3., 20.)):
        value = native.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(key + ' is outside the native Iris range')


def prepare_inference(options, spec):
    validate_inference(options, spec)
    architecture = spec.configuration
    runtime = load_runtime(dict(options, dummy=architecture['dummy']), architecture['seed'])
    restore = native_cleanup(runtime.model)
    bank = None
    try:
        if any(p.dtype != torch.float32 for p in runtime.model.parameters()):
            raise ValueError('the staged Iris base must use float32 weights')
        stamp = winning_formulation()
        bank = AdapterBank(runtime.model, stamp)
        bank.to(runtime.device)
        particles = new_particles(stamp, runtime.device)
        native = IrisResponseAdapter(runtime, bank, particles, model_identity=model_identity(runtime))
        identity = native_identity(native, model_identity=model_identity(runtime), protocol=PROTOCOL)

        def response(*, strength=1., seed=0, prompt='', height=None, width=None, timestep=500., cfg_scale=3.):
            size = 16 if architecture['dummy'] else 512
            height, width = size if height is None else height, size if width is None else width
            encoded, null = runtime.encode(prompt), runtime.encode_null()
            condition = dict(embeds=encoded.embeddings, mask=encoded.mask)
            negative = dict(embeds=null.embeddings, mask=null.mask)
            latent = torch.randn(1, 3, height, width, device=runtime.device,
                                 generator=torch.Generator(device=runtime.device).manual_seed(seed))
            time = torch.tensor(float(timestep), device=runtime.device)
            row = ResponseRow(neutral=condition, positive=condition, negative=negative,
                              latent=latent, timestep=time, guidance=float(cfg_scale))
            return native.cfg(row, latent, time, scale=strength).detach().clone()

        def close():
            try:
                bank.close()
            finally:
                restore()
        return InferenceBinding(native.network, identity, close, response=response,
                                observation=lambda: observation(runtime))
    except BaseException:
        try:
            if bank is not None:
                bank.close()
        finally:
            restore()
        raise


INFERENCE_BACKEND = InferenceBackend('iris', replace(CAPABILITIES, operations=('response',)), prepare_inference,
                                     validate_inference, preflight=preflight_inference)

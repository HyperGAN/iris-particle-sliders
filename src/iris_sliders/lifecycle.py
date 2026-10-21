"""Iris factory for the common image lifecycle; response-only until solver parity."""
from pathlib import Path

import torch
from particle_sliders.lifecycle import Backend, Binding, Capabilities, checkpoint_cleanup

_TRAIN = {'rows', 'steps', 'seed', 'batch', 'height', 'width', 'trajectory_steps',
          'records_per_pair', 'teacher_cfg', 'preset', 'response_scale', 'sigma',
          'critic_hidden', 'critic'}
_RUNTIME = {'checkpoint', 'text_dir', 'cache_dir', 'device', 'text_device'}
_CACHE_DIR = '/ml2/iris-particle-sliders/cache'


def validate(options):
    from .train import configuration
    if set(options) - (_TRAIN | _RUNTIME | {'dummy', 'base_precision', 'ema_decay'}):
        raise ValueError('unknown Iris session option')
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
    if options.get('dummy'):
        if (options.get('device', 'cpu') != 'cpu' or options.get('text_device', 'cpu') != 'cpu'
                or any(options.get(key) is not None for key in ('checkpoint', 'text_dir'))):
            raise ValueError('Iris dummy accepts only the CPU tiny fixture, without native model paths')
    else:
        cache_dir = options.get('cache_dir', _CACHE_DIR)
        if not isinstance(cache_dir, str) or not cache_dir:
            raise ValueError('native Iris requires a local cache_dir under /ml2')
        cache_path = Path(cache_dir).resolve()
        if cache_path != Path('/ml2') and Path('/ml2') not in cache_path.parents:
            raise ValueError('native Iris cache_dir must be under /ml2')
        for key in ('checkpoint', 'text_dir'):
            value = options.get(key)
            if value is not None and (not isinstance(value, str) or not Path(value).is_dir()):
                raise ValueError(key + ' must be an existing local directory')
    if any(type(options.get(key, 512)) is not int for key in ('height', 'width')):
        raise ValueError('Iris height and width must be integers')
    config = configuration(**{key: value for key, value in options.items() if key in _TRAIN})
    from .runtime import validate_shape
    validate_shape(config['height'], config['width'])


def prepare(options, *, runtime=None):
    # Direct staged callers receive the same allocation-free gates as Backend.open.
    validate(options)
    from .train import configuration, build_game
    config = configuration(**{key: value for key, value in options.items() if key in _TRAIN})
    if runtime is None:
        if options.get('dummy'):
            from .tiny import TinyRuntime
            runtime = TinyRuntime(config['seed'])
        else:
            from .runtime import IrisRuntime
            native = {key: value for key, value in options.items() if key in _RUNTIME}
            native.setdefault('cache_dir', _CACHE_DIR)
            runtime = IrisRuntime(allow_hub=False, **native)
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32
           for parameter in runtime.model.parameters()):
        raise ValueError('the staged Iris base must use float32 weights')
    restore = checkpoint_cleanup(runtime.model)
    try:
        game = build_game(runtime, config)
    except BaseException:
        restore()
        raise
    def close():
        try:
            game.backend.adapter.close()
        finally:
            restore()
    return Binding(game, close)


BACKEND = Backend('iris', Capabilities('routed_particle_gan', render=False,
                                     base_precisions=('float32',)), prepare, validate)

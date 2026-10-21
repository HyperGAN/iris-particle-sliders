"""Iris factory for the common image lifecycle; response-only until solver parity."""
import torch
from particle_sliders.lifecycle import Backend, Binding, native_cleanup

from .native import (CAPABILITIES, INFERENCE_BACKEND, PROTOCOL, RUNTIME_KEYS, admitted,
                     load_runtime, model_identity, observation, validate_runtime)

_TRAIN = {'rows', 'steps', 'seed', 'batch', 'height', 'width', 'trajectory_steps',
          'records_per_pair', 'teacher_cfg', 'preset', 'response_scale', 'sigma',
          'critic_hidden', 'critic'}
_RUNTIME = RUNTIME_KEYS


def validate(options):
    from .train import configuration
    if set(options) - (_TRAIN | _RUNTIME | {'dummy', 'base_precision', 'ema_decay'}):
        raise ValueError('unknown Iris session option')
    validate_runtime(options)
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
        runtime = load_runtime(options, config['seed'])
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32
           for parameter in runtime.model.parameters()):
        raise ValueError('the staged Iris base must use float32 weights')
    restore = native_cleanup(runtime.model)
    game = None
    try:
        game = build_game(runtime, config)
        spec = INFERENCE_BACKEND.specification(
            game.backend, model_identity=model_identity(runtime),
            protocol=PROTOCOL,
            configuration=dict(dummy=options.get('dummy', False), seed=config['seed']))
    except BaseException:
        try:
            if game is not None:
                game.backend.adapter.close()
        finally:
            restore()
        raise
    def close():
        try:
            game.backend.adapter.close()
        finally:
            restore()
    def before_update(request):
        return admitted(game.backend) and game.prior.z is game.backend.particles
    return Binding(game, close, inference=spec, before_update=before_update,
                   observation=lambda: observation(runtime))


def preflight(options, request, staged):
    if request['native']:
        raise ValueError('Iris training sessions evaluate bound rows; use native inference for new prompts')
    if set(staged) - {'runtime'}:
        raise ValueError('unknown Iris staged runtime argument')
    runtime = staged.get('runtime')
    if runtime is not None and any(p.dtype != torch.float32 for p in runtime.model.parameters()):
        raise ValueError('the staged Iris base must use float32 weights')


BACKEND = Backend('iris', CAPABILITIES, prepare, validate, preflight=preflight)

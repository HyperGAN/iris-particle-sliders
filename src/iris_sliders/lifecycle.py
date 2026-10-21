"""Iris factory for the common image lifecycle; response-only until solver parity."""
from particle_sliders.lifecycle import Backend, Binding, Capabilities, checkpoint_cleanup

_TRAIN = {'rows', 'steps', 'seed', 'batch', 'height', 'width', 'trajectory_steps',
          'records_per_pair', 'teacher_cfg', 'preset', 'response_scale', 'sigma',
          'critic_hidden', 'critic'}
_RUNTIME = {'checkpoint', 'text_dir', 'cache_dir', 'device', 'text_device'}


def validate(options):
    from .train import configuration
    if set(options) - (_TRAIN | _RUNTIME | {'dummy', 'base_precision', 'ema_decay'}):
        raise ValueError('unknown Iris session option')
    if options.get('dummy') and (options.get('device', 'cpu') != 'cpu' or
                                 options.get('base_precision', 'float32') != 'float32'):
        raise ValueError('Iris dummy is CPU float32 only')
    config = configuration(**{key: value for key, value in options.items() if key in _TRAIN})
    from .runtime import validate_shape
    validate_shape(config['height'], config['width'])


def prepare(options, *, runtime=None):
    from .train import configuration, build_game
    config = configuration(**{key: value for key, value in options.items() if key in _TRAIN})
    if runtime is None:
        if options.get('dummy'):
            from .tiny import TinyRuntime
            runtime = TinyRuntime(config['seed'])
        else:
            from .runtime import IrisRuntime
            runtime = IrisRuntime(allow_hub=False, **{key: value for key, value in options.items() if key in _RUNTIME})
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


BACKEND = Backend('iris', Capabilities('routed_particle_gan', render=False), prepare, validate)

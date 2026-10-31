"""Iris host-shape binding to the shared, verified HyperGAN/HNDL graphs.

All network operations, GAN objectives and update laws belong to shared packages.
This module only bridges Iris hook tensors to the existing shared graph inputs.
"""
from copy import deepcopy

import torch
from torch import nn

from .adapters import AdapterBank

PROFILE = 'hypergan-hndl-v2'


def default_profile():
    import json
    from importlib.resources import files
    marker = files('iris_sliders').joinpath('installed-profile.json')
    if not marker.is_file():
        return 'legacy'
    if json.loads(marker.read_text()) != {'network_profile': PROFILE}:
        raise ValueError('invalid Iris installed profile marker')
    return PROFILE


def verified_profile():
    from importlib.metadata import version
    if version('particle-sliders-core') != '0.12.0':
        raise RuntimeError('Iris hypergan-hndl-v2 requires the exact core 0.12 profile build')
    import hashlib
    import json
    from importlib.resources import files
    expected = json.loads(files('iris_sliders').joinpath('hndl-profile.json').read_text())
    core = files('particle_sliders')
    if any(hashlib.sha256(core.joinpath(name).read_bytes()).hexdigest() != digest
           for name, digest in expected['core_sources_sha256'].items()):
        raise ValueError('Iris HNDL core sources differ from the exact candidate')
    from particle_sliders.fibo_hndl import verified_builder
    builder, identity = verified_builder()
    if identity != expected['network_identity']:
        raise ValueError('Iris HNDL network provenance differs')
    return builder, identity


class HNDLRoutedProjection(nn.Module):
    """Apply the shared routed graph as an Iris projection-hook residual."""
    def __init__(self, base, stamp):
        super().__init__()
        if (stamp.spec['adapter_rank'], stamp.spec['parts'], stamp.spec['particle_dim']) != (8, 128, 4):
            raise ValueError('Iris HNDL requires the shared rank8/128x4 architecture')
        from particle_sliders.fibo_hndl import HNDLParticleProjection
        builder, identity = verified_profile()
        # The shared wrapper constructs and initializes the common graph. It does
        # not attach to the host, and its placeholder bank is never registered.
        shared = HNDLParticleProjection('iris', base, torch.empty(128, 4), 8, 8.,
                                       builder=builder, identity=identity, attach=False)
        self.graph = shared.graph
        self.input_width, self.output_width = base.in_features, base.out_features
        self.identity = deepcopy(identity)

    @property
    def down(self):
        return self.graph['down']

    @property
    def up(self):
        return self.graph['up']

    def extra_repr(self):
        import json
        return json.dumps(self.identity, sort_keys=True)

    def forward(self, x, particles):
        shape = x.shape
        value = x.to(self.down.weight).reshape(-1, 1, self.input_width)
        bank = particles.unsqueeze(0).expand(value.shape[0], -1, -1)
        return self.graph(value=value, particles=bank).reshape(*shape[:-1], self.output_width)


def adapter_bank(model, stamp, profile='legacy'):
    if profile == 'legacy':
        return AdapterBank(model, stamp)
    if profile != PROFILE:
        raise ValueError('unknown Iris network profile')
    verified_profile()
    return AdapterBank(model, stamp, projection=HNDLRoutedProjection)


def spatial_critic(channels, condition_width, spatial_shape, width):
    verified_profile()
    from particle_sliders.fibo_hndl import HNDLFieldCritic
    return HNDLFieldCritic(channels, condition_width, spatial_shape, width=width)

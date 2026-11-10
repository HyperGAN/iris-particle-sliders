"""RAW-only native velocity owner; no critic, game, teachers or fresh initializer."""
from copy import deepcopy
import math
import torch
from particle_sliders.lifecycle import _network_observation
from particle_sliders.pg09_backends import ResponseRow
from iris_sliders.native import model_identity,observation
from iris_sliders.train import configuration
from . import PROFILE,ownership
from .training import Owner,ReceiptBackend,RAW_SCHEMA,attach,receipt_valid
from .util import canonical,equal,read_packet

class Inference(Owner):
    def __init__(self,runtime,*,raw,owner_admission=None):
        super().__init__()
        architecture,_=ownership.require();sources=ownership.sources()
        packet=read_packet(raw)
        keys={'schema','profile','cohort_id','config','sources','initialization','cache_identity','model_identity','backend_metadata','network','completed_steps','selection','ema_decay','rendering'}
        if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=RAW_SCHEMA
                or packet['profile']!=PROFILE or packet['sources']!=sources
                or packet['selection']!='raw' or packet['ema_decay']is not None
                or packet['model_identity']!=model_identity(runtime)
                or type(packet['completed_steps'])is not int or packet['completed_steps']<0):
            raise ValueError('exact new-profile RAW metadata required; no old migration')
        config=packet['config']
        if configuration(**config)!=config or not receipt_valid(packet['initialization'],packet['cohort_id'],config,sources,architecture):
            raise ValueError('RAW original config/receipt differs')
        if any(p.is_floating_point()and p.dtype!=torch.float32 for p in runtime.model.parameters()):
            raise ValueError('original Iris base precision is FP32')
        runtime.device=next(runtime.model.parameters()).device
        if runtime.device.type!='cpu'and(not callable(owner_admission)or owner_admission(dict(
                operation='RAW-response',profile=PROFILE,model_identity=model_identity(runtime)))is not True):
            raise ValueError('external owner did not admit native RAW inference')
        self.runtime,self.packet,self.config=runtime,packet,config
        devices=[runtime.device.index or 0]if runtime.device.type=='cuda'else[]
        try:
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(torch.Generator().manual_seed(config['seed']).get_state())
                if devices:torch.cuda.set_rng_state(torch.Generator(device=runtime.device).manual_seed(config['seed']).get_state(),runtime.device)
                self.bank,particles,self.native=attach(runtime,config,architecture)
            self.backend=ReceiptBackend(self.native,packet['initialization'],sources,packet['cache_identity'])
            if canonical(self.backend.metadata())!=canonical(packet['backend_metadata']):raise ValueError('RAW native backend metadata differs')
            expected=self.native.network.state_dict();state=packet['network']
            if not isinstance(state,dict)or set(state)!=set(expected):raise ValueError('RAW tensor keys differ')
            for key,value in state.items():
                if (not isinstance(value,torch.Tensor)or value.shape!=expected[key].shape or value.dtype!=expected[key].dtype
                        or not torch.isfinite(value).all()):raise ValueError('RAW tensor schema/finite state differs')
            self.native.network.load_state_dict(state,strict=True)
            if not equal(self.native.network.state_dict(),state):raise RuntimeError('RAW load equality failed')
            # No game/critic/cache load or scoped initializer on this route.
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Iris RAW entry/cleanup failed',[primary,cleanup])
            raise

    @torch.no_grad()
    def response(self,row,*,strength=1.,seed=0):
        self.require_open()
        if (type(row)is not ResponseRow or row.schedule is not None or row.timestep is None
                or tuple(row.latent.shape)!=(1,3,self.config['height'],self.config['width'])):
            raise ValueError('original declared Iris RGB velocity row required')
        if type(seed)is not int or not 0<=seed<2**63:raise ValueError('explicit response seed required')
        if type(strength)not in(int,float)or not math.isfinite(strength)or not 0<=strength<=1:
            raise ValueError('Iris native scale is in [0,1]')
        device=self.runtime.device;devices=[device.index or 0]if device.type=='cuda'else[]
        with _network_observation(self.native.network,seed,devices=devices,observation=lambda:observation(self.runtime)):
            return self.native.cfg(row,row.latent,row.timestep,scale=float(strength)).detach().clone()

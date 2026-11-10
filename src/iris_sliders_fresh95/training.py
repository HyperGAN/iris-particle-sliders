"""Explicit fresh/resume cached entry, reusing original Iris/core components."""
from copy import deepcopy
import hashlib
from pathlib import Path
import torch
from particle_sliders import winning_formulation
from particle_sliders.field_critic import FieldCritic
from particle_sliders.pg09 import ConditionalPG09
from particle_sliders.pg09_images import IrisResponseAdapter
from iris_sliders.adapters import new_particles
from iris_sliders.profiles import adapter_bank,select_profile
from iris_sliders.native import model_identity,admitted
from iris_sliders.train import configuration
from . import PROFILE,ownership,policy
from .cache import load_cache,to_device
from .util import canonical,equal,new_directory,read_packet,save_packet,write_json

FULL_SCHEMA='Iris-explicit-cached-fresh95-v1-FULL'
RAW_SCHEMA='Iris-explicit-cached-fresh95-v1-RAW'

class ReceiptBackend:
    def __init__(self,backend,initialization,sources,cache_identity):
        self.backend=backend
        self.network,self.particles=backend.network,backend.particles
        self._receipt,self._sources,self._cache=deepcopy(initialization),deepcopy(sources),deepcopy(cache_identity)
    def __getattr__(self,name):return getattr(self.backend,name)
    def metadata(self):
        return dict(self.backend.metadata(),fresh_initialization=dict(profile=PROFILE,
            receipt=deepcopy(self._receipt),sources=deepcopy(self._sources)),completed_native_cache=deepcopy(self._cache))

class Owner:
    def __init__(self):self.bank=None;self.closed=False;self.closing=False;self.cleanup_pending=False
    def close(self):
        if self.closed:return
        self.closing=True
        try:
            if self.bank is not None:self.bank.close()  # Original retry-only-pending Iris Cleanup.
        except BaseException:
            self.cleanup_pending=True
            raise
        self.closed,self.closing,self.cleanup_pending=True,False,False
    def require_open(self):
        if self.closed or self.closing:raise RuntimeError('Iris owner is closed/cleanup pending')

def receipt_valid(receipt,cohort_id,config,sources,architecture):
    return (isinstance(receipt,dict)and receipt.get('schema')==policy.SCHEMA
        and receipt.get('policy')==policy.POLICIES['iris']and receipt.get('product')=='iris'
        and receipt.get('cohort_id')==cohort_id and receipt.get('seed')==config['seed']
        and receipt.get('source_identity')==dict(sources,network_profile=architecture)
        and receipt.get('completed_steps')==0 and receipt.get('restored_from')is None
        and receipt.get('selection')=='raw'and receipt.get('ema_decay')is None)

def attach(runtime,config,architecture):
    stamp=winning_formulation()
    bank=adapter_bank(runtime.model,stamp,architecture)
    try:
        bank.to(runtime.device)
        particles=new_particles(stamp,runtime.device)
        native=IrisResponseAdapter(runtime,bank,particles,model_identity=model_identity(runtime))
        return bank,particles,native
    except BaseException as primary:
        try:bank.close()
        except BaseException as cleanup:raise BaseExceptionGroup('Iris attachment/cleanup failed',[primary,cleanup])
        raise

class Training(Owner):
    def __init__(self,runtime,cache,directory,*,mode,cohort_id,config,resume=None,ema_decay=None,owner_admission=None):
        super().__init__()
        # Resolve explicit intent and read packet/cache before any added component.
        if mode not in('fresh','resume')or not isinstance(cohort_id,str)or not cohort_id.strip():
            raise ValueError('explicit fresh|resume and profile cohort ID required')
        if (mode=='fresh'and resume is not None)or(mode=='resume'and resume is None):
            raise ValueError('fresh refuses saved state; resume requires exact new-profile FULL')
        if ema_decay is not None:raise ValueError('no weight EMA; current RAW only')
        self.directory=Path(directory).absolute()
        if self.directory.exists():raise FileExistsError('new output directory required')
        architecture,_=ownership.require()
        if configuration(**config)!=config:raise ValueError('original exact Iris configuration required')
        config=deepcopy(config)
        if select_profile(config.get('network_profile'))!=architecture:raise ValueError('installed architecture differs')
        if any(p.is_floating_point()and p.dtype!=torch.float32 for p in runtime.model.parameters()):
            raise ValueError('original Iris base precision is FP32')
        runtime.device=next(runtime.model.parameters()).device
        self.runtime,self.config,self.cohort_id,self.architecture=runtime,config,cohort_id,architecture
        self.sources=ownership.sources();self._owner_admission=owner_admission
        if runtime.device.type!='cpu'and not callable(owner_admission):raise ValueError('native runtime requires external owner admission')
        packet=read_packet(resume)if mode=='resume'else None
        if packet is not None:
            keys={'schema','profile','cohort_id','config','sources','initialization','cache_identity','game','ema_decay'}
            if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=FULL_SCHEMA
                    or packet['profile']!=PROFILE or packet['cohort_id']!=cohort_id or packet['config']!=config
                    or packet['sources']!=self.sources or packet['ema_decay']is not None
                    or not receipt_valid(packet['initialization'],cohort_id,config,self.sources,architecture)):
                raise ValueError('exact new-profile FULL/receipt required; no old packet migration')
        prepared,conditions,self.cache_identity=load_cache(cache,runtime=runtime,config=config,device='cpu')
        if packet is not None and packet['cache_identity']!=self.cache_identity:raise ValueError('FULL native cache differs')
        if runtime.device.type!='cpu'and owner_admission(dict(operation='construct-train',profile=PROFILE,
                model_identity=model_identity(runtime),completed_cache=deepcopy(self.cache_identity)))is not True:
            raise ValueError('external owner did not admit native construction/training')
        prepared,conditions=to_device(prepared,runtime.device),conditions.to(runtime.device)
        devices=[runtime.device.index or 0]if runtime.device.type=='cuda'else[]
        try:
            # New cached cohort starts its declared constructor stream at seed;
            # no claim that old trajectory teacher RNG consumption is replayed.
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(torch.Generator().manual_seed(config['seed']).get_state())
                if devices:torch.cuda.set_rng_state(torch.Generator(device=runtime.device).manual_seed(config['seed']).get_state(),runtime.device)
                self.bank,particles,native=attach(runtime,config,architecture)
                if architecture=='hypergan-hndl-pg095-v1':
                    from iris_sliders_hndl import spatial_critic
                    critic=spatial_critic(prepared[0].target.shape[1],conditions.shape[1],tuple(prepared[0].target.shape[2:]),config['critic_hidden']).to(runtime.device)
                elif config['critic']=='hndl':
                    from particle_sliders.pg09_hndl import HNDLConditionalReadoutCritic
                    critic=HNDLConditionalReadoutCritic(prepared[0].target.numel(),conditions.shape[1],hidden=config['critic_hidden']).to(runtime.device)
                else:
                    critic=FieldCritic(prepared[0].target.shape[1],conditions.shape[1],width=config['critic_hidden']).to(runtime.device)
            if mode=='fresh':
                self.initialization=policy.initialize('iris',self.bank,critic,
                    construction=policy.Construction('fresh',cohort_id,hashlib.sha256(policy.canonical(config).encode()).hexdigest()),
                    options=config,seed=config['seed'],source_identity=dict(self.sources,network_profile=architecture),
                    particles=particles,iris_profile=architecture,optimizers=())
            else:self.initialization=deepcopy(packet['initialization'])
            self.backend=ReceiptBackend(native,self.initialization,self.sources,self.cache_identity)
            # First optimizer construction is after the fresh scoped initializer.
            self.game=ConditionalPG09(self.backend,prepared,critic,conditions=conditions,particles=particles,
                response_domain='continuous_field',response_scale=config['response_scale'],sigma=config['sigma'],
                preset=config['preset'],total_steps=config['steps'],batch_size=config['batch'],seed=config['seed'],
                sampling='balanced',recipe_overrides={k:config[k]for k in('lr','d_lr_mult','prior_lr_mult')if k in config})
            if mode=='resume':
                self.game.load_state_dict(packet['game'])
                if not equal(self.game.state_dict(),packet['game']):raise RuntimeError('complete typed game/optimizer/RNG restore differs')
            new_directory(self.directory)
            write_json(self.directory/'initialization.json',self.initialization)
            self.entry_FULL=self.save()
            self.entry_RAW=self.export(self.directory/f'RAW{self.step:06d}.pt')
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Iris entry/cleanup failed',[primary,cleanup])
            raise
    @property
    def step(self):return self.game.completed_steps
    def update(self):
        self.require_open()
        if not admitted(self.backend.backend):raise RuntimeError('original native ownership/precision law differs')
        return self.game.step()
    def save(self):
        self.require_open()
        return save_packet(self.directory/f'FULL{self.step:06d}.pt',dict(schema=FULL_SCHEMA,profile=PROFILE,
            cohort_id=self.cohort_id,config=deepcopy(self.config),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),game=self.game.state_dict(),ema_decay=None))
    def export(self,path):
        self.require_open()
        state=self.game.export_state_dict()
        return save_packet(path,dict(schema=RAW_SCHEMA,profile=PROFILE,cohort_id=self.cohort_id,
            config=deepcopy(self.config),sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            cache_identity=deepcopy(self.cache_identity),model_identity=model_identity(self.runtime),
            backend_metadata=self.backend.metadata(),network=state['network'],completed_steps=self.step,
            selection='raw',ema_decay=None,rendering='raw RGB velocity only; solver rendering remains unqualified'))

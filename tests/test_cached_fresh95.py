"""FOUR NEW UNRUN Iris entry tests; random tiny model/synthetic cache, no pretrained teachers."""
from copy import deepcopy
from pathlib import Path
import pytest
import torch
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow
from particle_sliders.pg09_images import IrisResponseAdapter
from iris_sliders.tiny import TinyRuntime
from iris_sliders.train import configuration
from iris_sliders.profiles import default_profile
from iris_sliders_fresh95 import open_training,open_inference,save_prepared_cache
from iris_sliders_fresh95 import policy,training
from iris_sliders_fresh95.util import Checkpoint,equal,file_pin


def runtime_fixture():
    with torch.random.fork_rng():return TinyRuntime(seed=9)

def config_fixture():
    return configuration([dict(neutral='software neutral one',positive='software target one'),
        dict(neutral='software neutral two',positive='software target two')],steps=4,seed=7,batch=2,
        height=16,width=16,trajectory_steps=4,records_per_pair=1,teacher_cfg=3.,critic_hidden=8,
        network_profile=default_profile())

def prepared_fixture(config):
    result=[]
    for index in range(2):
        mask=torch.ones(1,8,dtype=torch.long);mask[:,-2:]=0
        embeds=torch.full((1,8,2,32),.2+index*.1)
        condition=dict(embeds=embeds,mask=mask)
        positive=dict(embeds=embeds+.1,mask=mask.clone())
        negative=dict(embeds=torch.zeros_like(embeds),mask=mask.clone())
        latent=torch.randn(1,3,16,16,generator=torch.Generator().manual_seed(index+11))
        row=ResponseRow(neutral=condition,positive=positive,negative=negative,latent=latent,
            timestep=torch.tensor(500.-100*index),guidance=config['teacher_cfg'])
        result.append(PreparedResponse(row,torch.full_like(latent,.1),torch.zeros_like(latent)))
    return result

def cache_fixture(tmp_path,runtime,config):
    rows=prepared_fixture(config)
    return save_prepared_cache(tmp_path/'completed-cache.pt',runtime=runtime,config=config,prepared=rows,
        teacher_origins=[dict(kind='NEW synthetic software data; not native teacher outputs',row=i)for i in range(2)],
        cache_id='NEW-software-completed-cache'),rows

def forbid_teachers(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('cached entry attempted teacher collection/preparation')
    import iris_sliders.teacher
    monkeypatch.setattr(iris_sliders.teacher,'collect_response_rows',forbidden)
    monkeypatch.setattr(IrisResponseAdapter,'prepare',forbidden)


def test_new_cached_fresh_receipt_precedes_optimizers_and_preserves_parameters(tmp_path,monkeypatch):
    runtime,config=runtime_fixture(),config_fixture();cache,rows=cache_fixture(tmp_path,runtime,config)
    forbid_teachers(monkeypatch)
    initialize,game=policy.initialize,training.ConditionalPG09;seen={}
    def initialize_spy(*args,**kwargs):
        bank,critic=args[1:3];particles=kwargs['particles']
        seen['ids']=[id(p)for p in bank.parameters()]+[id(particles)]+[id(p)for p in critic.parameters()]
        seen['critic_before']=deepcopy(critic.state_dict())
        receipt=initialize(*args,**kwargs)
        assert seen['ids']==[id(p)for p in bank.parameters()]+[id(particles)]+[id(p)for p in critic.parameters()]
        assert equal(seen['critic_before'],critic.state_dict())
        seen['receipt']=receipt;return receipt
    def game_spy(backend,prepared,critic,**kwargs):
        assert backend.metadata()['fresh_initialization']['receipt']==seen['receipt']
        assert backend.network.particles is backend.particles is kwargs['particles']
        seen['before_optimizer']=True;return game(backend,prepared,critic,**kwargs)
    monkeypatch.setattr(policy,'initialize',initialize_spy);monkeypatch.setattr(training,'ConditionalPG09',game_spy)
    owner=open_training(runtime,cache,tmp_path/'fresh',mode='fresh',cohort_id='NEW-Iris-software',config=config)
    try:
        assert seen['before_optimizer']and owner.step==0
        assert owner.entry_FULL.bytes>0 and owner.entry_RAW.bytes>0
        assert all(torch.count_nonzero(layer.up.weight).item()==0 for layer in owner.bank.layers)
        bankrow=next(r for r in owner.initialization['parameter_policies']if r['name']=='bank.particles')
        assert bankrow['role']=='declared-bank-R2std002'and '0.02'in bankrow['distribution']
        for row in owner.initialization['parameter_policies']:
            if row['role']in('constructor-law-KEEP','protected-zero-UP'):
                assert row['before_sha256']==row['after_sha256']
        assert owner.game.prior.z is owner.backend.network.particles
    finally:owner.close()
    assert all(not m._forward_hooks for m in runtime.model.modules())


def test_new_policy_determinism_host_scale_and_rng_receipt(tmp_path):
    runtime,config=runtime_fixture(),config_fixture();cache,_=cache_fixture(tmp_path,runtime,config)
    before=torch.get_rng_state().clone()
    first=open_training(runtime,cache,tmp_path/'a',mode='fresh',cohort_id='fresh-a',config=config)
    second=None
    try:
        assert torch.equal(before,torch.get_rng_state())
        second=open_training(runtime_fixture(),cache,tmp_path/'b',mode='fresh',cohort_id='fresh-b',config=config)
        assert equal(first.backend.network.state_dict(),second.backend.network.state_dict())
        assert equal(first.game.critic.state_dict(),second.game.critic.state_dict())
        for layer in first.bank.layers:
            weight=layer.down.weight.detach();variance=1/(3*layer.down.in_features)
            gram=weight@weight.T
            assert torch.allclose(gram,torch.eye(gram.shape[0])*max(weight.shape)*variance,atol=2e-6,rtol=2e-5)
        assert first.initialization['seed']==7 and first.initialization['parameter_identity_preserved']is True
    finally:
        first.close()
        if second is not None:second.close()


def test_new_strict_FULL_resume_and_RAW_inference_skip_initialization_and_teachers(tmp_path,monkeypatch):
    runtime,config=runtime_fixture(),config_fixture();cache,rows=cache_fixture(tmp_path,runtime,config)
    forbid_teachers(monkeypatch)
    first=open_training(runtime,cache,tmp_path/'first',mode='fresh',cohort_id='resume-case',config=config)
    try:
        first.update()  # ONE genuinely new tiny software update, no old test replay.
        full=first.save();raw=first.export(tmp_path/'RAW1.pt')
        expected,receipt=first.game.state_dict(),deepcopy(first.initialization)
    finally:first.close()
    def forbidden(*args,**kwargs):raise AssertionError('restore/inference invoked fresh initializer')
    monkeypatch.setattr(policy,'initialize',forbidden)
    resumed=open_training(runtime_fixture(),cache,tmp_path/'resumed',mode='resume',cohort_id='resume-case',config=config,resume=full)
    try:
        assert resumed.step==1 and resumed.initialization==receipt
        assert equal(resumed.game.state_dict(),expected)
    finally:resumed.close()
    inference=open_inference(runtime_fixture(),raw=raw)
    try:
        assert not hasattr(inference,'game')and not hasattr(inference,'critic')
        response=inference.response(rows[0].row,strength=1.,seed=3)
        assert response.shape==(1,3,16,16)and torch.isfinite(response).all()
        assert inference.packet['initialization']==receipt
    finally:inference.close()


def test_new_missing_cache_and_legacy_intent_refuse_before_components_and_cleanup_retries(tmp_path,monkeypatch):
    runtime,config=runtime_fixture(),config_fixture();cache,_=cache_fixture(tmp_path,runtime,config)
    def forbidden(*args,**kwargs):raise AssertionError('refusal reached added component construction')
    broken=tmp_path/'corrupt-cache.pt'
    broken.write_bytes(Path(cache.path).read_bytes()+b'corrupt')
    corrupt=Checkpoint(str(broken),cache.bytes,cache.sha256)
    missing=Checkpoint(str(tmp_path/'missing-cache.pt'),cache.bytes,cache.sha256)
    with monkeypatch.context()as scoped:
        scoped.setattr(training,'attach',forbidden)
        for patch in (dict(cache=None),dict(cache=corrupt),dict(cache=missing),dict(mode='resume'),dict(mode='fresh',resume=cache),dict(ema_decay=.9)):
            args=dict(runtime=runtime,cache=cache,directory=tmp_path/'refused',mode='fresh',cohort_id='refused',config=config);args.update(patch)
            with pytest.raises((ValueError,FileNotFoundError)):
                open_training(**args)
        old=tmp_path/'old.pt';torch.save(dict(schema='iris-conditional-pg09-training-v1',config=config,game={}),old)
        with pytest.raises(ValueError):
            open_training(runtime,cache,tmp_path/'refused',mode='resume',cohort_id='refused',config=config,resume=file_pin(old))
    owner=open_training(runtime,cache,tmp_path/'cleanup',mode='fresh',cohort_id='cleanup',config=config)
    handle=owner.bank.handles[0];remove=handle.remove;calls=[]
    def fail_once():
        calls.append('remove')
        if len(calls)==1:raise RuntimeError('NEW synthetic hook removal fault')
        remove()
    monkeypatch.setattr(handle,'remove',fail_once)
    with pytest.raises(RuntimeError):owner.close()
    assert owner.closing and owner.cleanup_pending and not owner.closed and handle in owner.bank.handles
    with pytest.raises(RuntimeError):owner.update()
    owner.close();owner.close()
    assert owner.closed and not owner.cleanup_pending and calls==['remove','remove']
    assert all(not m._forward_hooks for m in runtime.model.modules())

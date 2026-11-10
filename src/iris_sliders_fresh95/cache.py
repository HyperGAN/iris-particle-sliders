"""Persist/reuse already-completed native prepared rows; zero teacher forwards."""
from dataclasses import fields
import hashlib,math
import torch
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow
from particle_sliders.pg09_images import image_condition_features
from iris_sliders.native import model_identity
from . import ownership
from .util import canonical,read_packet,save_packet,tree

SCHEMA='Iris-completed-native-prepared-cache-fresh95-v1'
TEACHER_KEYS=('rows','seed','height','width','trajectory_steps','records_per_pair','teacher_cfg')

def teacher_configuration(config):return {k:config[k]for k in TEACHER_KEYS}

def _finite(value):
    if isinstance(value,torch.Tensor):
        if value.requires_grad or value.is_complex()or value.layout!=torch.strided or not torch.isfinite(value).all():
            raise ValueError('completed cache requires frozen finite dense real tensors')
    elif isinstance(value,dict):
        if any(not isinstance(k,str)for k in value):raise ValueError('string native condition keys required')
        for v in value.values():_finite(v)
    elif isinstance(value,(tuple,list)):
        for v in value:_finite(v)
    elif value is not None and type(value)not in(str,int,float,bool):
        raise ValueError('cache contains an unsupported object')
    elif type(value)is float and not math.isfinite(value):raise ValueError('finite cache scalar required')

def validate_prepared(prepared,config):
    expected=(1,3,config['height'],config['width'])
    if len(prepared)!=len(config['rows'])*config['records_per_pair'] or len(prepared)<2:
        raise ValueError('complete original declared fixed-row bank required')
    for item in prepared:
        if type(item)is not PreparedResponse or type(item.row)is not ResponseRow:
            raise ValueError('actual original PreparedResponse/ResponseRow objects required')
        row=item.row
        if (row.schedule is not None or row.timestep is None or row.timestep.numel()!=1
                or not 0<=float(row.timestep)<=1000 or row.scale!=1 or row.strength!=1
                or row.guidance!=config['teacher_cfg']):
            raise ValueError('original Iris single velocity/CFG row contract differs')
        for value in (row.latent,item.target,item.neutral):
            if tuple(value.shape)!=expected or value.dtype!=torch.float32:
                raise ValueError('original complete FP32 RGB fields required')
        for condition in (row.neutral,row.positive,row.negative):
            if not isinstance(condition,dict)or set(condition)!={'embeds','mask'}:
                raise ValueError('all native Iris layer embeddings and token masks required')
            embeds,mask=condition['embeds'],condition['mask']
            if (embeds.ndim not in(3,4)or embeds.shape[0]!=1 or mask.shape!=embeds.shape[:2]
                    or embeds.dtype!=torch.float32):raise ValueError('native condition geometry differs')
        for value in (row.neutral,row.positive,row.negative,row.latent,row.timestep,item.target,item.neutral):_finite(value)
    return prepared

def _dump(item):
    return dict(row={f.name:tree(getattr(item.row,f.name),'cpu')for f in fields(ResponseRow)},target=tree(item.target,'cpu'),neutral=tree(item.neutral,'cpu'))

def _restore(row,device):
    if not isinstance(row,dict)or set(row)!={'row','target','neutral'}or set(row['row'])!={f.name for f in fields(ResponseRow)}:
        raise ValueError('complete serialized native row required')
    return PreparedResponse(ResponseRow(**tree(row['row'],device)),tree(row['target'],device),tree(row['neutral'],device))

def save_prepared_cache(path,*,runtime,config,prepared,teacher_origins,cache_id):
    """Serialize finished inputs only; cannot collect or repeat a teacher call.

    The caller supplies truthful per-row completed-output provenance. These
    records are input provenance, not a new pretrained/native qualification.
    """
    ownership.require()
    from iris_sliders.train import configuration
    if configuration(**config)!=config:raise ValueError('exact original config required')
    if (not isinstance(cache_id,str)or not cache_id.strip()or not isinstance(teacher_origins,list)
            or len(teacher_origins)!=len(prepared)or any(not isinstance(x,dict)or not x for x in teacher_origins)):
        raise ValueError('explicit completed cache ID and each row origin required')
    canonical(teacher_origins)  # Finite portable metadata only; no automatic origin claims.
    validate_prepared(prepared,config)
    # Exact original condition function; no runtime.encode/predict/model calls.
    conditions=image_condition_features(prepared).detach().cpu().clone()
    data=dict(schema=SCHEMA,cache_id=cache_id,teacher_configuration=teacher_configuration(config),
        model_identity=model_identity(runtime),original_sources=ownership.sources()['original'],
        teacher_origins=teacher_origins,rows=[_dump(x)for x in prepared],conditions=conditions,
        completed=True,model_or_teacher_calls_by_serializer=0)
    return save_packet(path,data)

def load_cache(pin,*,runtime,config,device=None):
    packet=read_packet(pin)
    keys={'schema','cache_id','teacher_configuration','model_identity','original_sources','teacher_origins','rows','conditions','completed','model_or_teacher_calls_by_serializer'}
    if (not isinstance(packet,dict)or set(packet)!=keys or packet['schema']!=SCHEMA
            or packet['completed']is not True or packet['model_or_teacher_calls_by_serializer']!=0
            or packet['teacher_configuration']!=teacher_configuration(config)
            or packet['model_identity']!=model_identity(runtime)
            or packet['original_sources']!=ownership.sources()['original']):
        raise ValueError('completed native cache identity/configuration/Source differs')
    prepared=[_restore(x,'cpu')for x in packet['rows']];validate_prepared(prepared,config)
    conditions=packet['conditions'];_finite(conditions)
    if (conditions.dtype!=torch.float32 or conditions.ndim!=2 or conditions.shape[0]!=len(prepared)
            or not torch.equal(image_condition_features(prepared).cpu(),conditions)):
        raise ValueError('persisted original complete critic conditions differ')
    if (not isinstance(packet['teacher_origins'],list)or len(packet['teacher_origins'])!=len(prepared)
            or any(not isinstance(x,dict)or not x for x in packet['teacher_origins'])):
        raise ValueError('per-row completed teacher origins missing')
    canonical(packet['teacher_origins'])
    identity=dict(schema=SCHEMA,bytes=pin.bytes,sha256=pin.sha256,cache_id=packet['cache_id'],
        teacher_configuration=packet['teacher_configuration'],model_identity=packet['model_identity'],
        original_sources=packet['original_sources'],teacher_origins=packet['teacher_origins'],row_count=len(prepared))
    # Original build_game holds the complete prepared bank on this same device.
    destination=runtime.device if device is None else device
    return [_restore(x,destination)for x in packet['rows']],conditions.to(destination),identity

def to_device(prepared,device):
    return [_restore(_dump(x),device)for x in prepared]

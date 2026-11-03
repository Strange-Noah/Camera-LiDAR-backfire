"""Real Torch snapshots, alias-preserving replay and explicit runtime state."""
import contextlib,hashlib,io,json,os,pickle,random,copy,enum
import numpy as np
import torch


def tensors(value,path='root'):
    if isinstance(value,torch.Tensor):yield path,value
    elif isinstance(value,dict):
        for k in sorted(value):yield from tensors(value[k],path+'/'+str(k))
    elif isinstance(value,(list,tuple)):
        for i,v in enumerate(value):yield from tensors(v,path+'/'+str(i))
    elif hasattr(value,'tensor') and isinstance(value.tensor,torch.Tensor):yield path+'/tensor',value.tensor


def immutable_cpu(t):
    if t.layout!=torch.strided:raise ValueError('Only dense strided snapshot supported')
    # Blocking device transfer + independent clone; never keep a live view.
    if t.is_cuda:torch.cuda.synchronize(t.device)
    return t.detach().cpu().contiguous().clone()


def raw_bytes(t):
    cpu=immutable_cpu(t)
    # Torch 1.10 cannot reinterpret FP32/FP64 as differently sized uint8.
    # NumPy exports exact contiguous bytes without changing values or dtype.
    try:return cpu.numpy().tobytes()
    except TypeError:
        if cpu.dtype==torch.bfloat16:return cpu.view(torch.int16).numpy().tobytes()
        raise
def tensor_hash(t):return hashlib.sha256(raw_bytes(t)).hexdigest()
def storage_key(t):return (str(t.device),t.storage().data_ptr(),t.storage().size(),str(t.dtype))


def describe(value):
    leaves=list(tensors(value));groups={};records={}
    for path,t in leaves:
        key=storage_key(t);group=groups.setdefault(key,len(groups))
        records[path]=dict(shape=list(t.shape),dtype=str(t.dtype),device=str(t.device),stride=list(t.stride()),storage_offset=t.storage_offset(),storage_elements=t.storage().size(),storage_group=group,contiguous=t.is_contiguous(),requires_grad=t.requires_grad,sha256=tensor_hash(t))
    # Same-storage grouping is conservative: disjoint views may share storage.
    # Actual reachable offset sets are not enumerated for enormous image tensors.
    return records


def value_signature(value):
    return {p:{k:r[k] for k in ('shape','dtype','sha256')} for p,r in describe(value).items()}


def replay(value,device='cpu',cpu_only_keys=()):
    """Copy complete typed storages once; restore original strides/offsets/aliases.

    Mixed-dtype views on one untyped storage are rejected, never silently split.
    Plain tensor/dict/list/tuple/scalar payload only, prepared after MMCV scatter.
    """
    memo={};bases={};typed={};destinations={};storage_destinations={}
    def visit(v,target):
        if isinstance(v,torch.Tensor):
            target=str(target)
            if id(v) in memo:
                if destinations[id(v)]!=target:raise ValueError('Same tensor requested on different devices')
                return memo[id(v)]
            if v.layout!=torch.strided or any(s<0 for s in v.stride()):raise ValueError('Unsupported tensor layout')
            if id(v) in destinations and destinations[id(v)]!=target:raise ValueError('Same tensor requested on different devices')
            destinations[id(v)]=target
            untyped=(str(v.device),v.storage().data_ptr())
            if untyped in storage_destinations and storage_destinations[untyped]!=target:raise ValueError('Shared storage requested across device roles')
            storage_destinations[untyped]=target
            if untyped in typed and typed[untyped]!=v.dtype:raise ValueError('Mixed dtype shared storage')
            typed[untyped]=v.dtype;key=(storage_key(v),str(target))
            if key not in bases:
                flat=torch.empty(0,dtype=v.dtype,device=v.device).set_(v.storage(),0,(v.storage().size(),),(1,))
                bases[key]=flat.detach().to(device=target,copy=True)
            out=torch.as_strided(bases[key],v.size(),v.stride(),v.storage_offset());memo[id(v)]=out;return out
        if isinstance(v,dict):return {k:visit(x,target) for k,x in v.items()}
        if isinstance(v,list):return [visit(x,target) for x in v]
        if isinstance(v,tuple):return tuple(visit(x,target) for x in v)
        if isinstance(v,type):return v
        if hasattr(v,'tensor') and type(v).__module__.startswith('mmdet3d.core.bbox.structures'):
            result=copy.copy(v);result.tensor=visit(v.tensor,target);return result
        if v is None or isinstance(v,(str,int,float,bool)):return v
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,np.ndarray):return v.copy()
        raise TypeError('Unprepared payload object: '+type(v).__name__)
    if not isinstance(value,dict) and cpu_only_keys:raise ValueError('CPU-only key policy requires mapping')
    return {k:visit(v,'cpu' if k in cpu_only_keys else device) for k,v in value.items()} if isinstance(value,dict) else visit(value,device)


def seed_all(seed,cuda=False):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if cuda:torch.cuda.manual_seed_all(seed)


def rng_state(cuda=False):
    out={'python':hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),'numpy':hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),'torch_cpu':tensor_hash(torch.get_rng_state())}
    if cuda:out['torch_cuda']=[tensor_hash(x) for x in torch.cuda.get_rng_state_all()]
    return out


def runtime():
    return dict(torch_config=torch.__config__.show(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cudnn_allow_tf32=getattr(torch.backends.cudnn,'allow_tf32',None),matmul_allow_tf32=getattr(torch.backends.cuda.matmul,'allow_tf32',None),autocast_enabled=torch.is_autocast_enabled(),CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),PYTHONHASHSEED=os.environ.get('PYTHONHASHSEED'))


def model_state(model):
    registered={**{'parameter/'+n:p for n,p in model.named_parameters()},**{'buffer/'+n:b for n,b in model.named_buffers()}}
    mutable={}
    for n,m in model.named_modules():
        for attr in sorted(set(('query_labels','tasks')) | {k for k,v in vars(m).items() if isinstance(v,torch.Tensor) and k not in m._parameters and k not in m._buffers}):
            if hasattr(m,attr):mutable[n+'/'+attr]=state_value(getattr(m,attr))
        coder=getattr(m,'bbox_coder',None)
        if coder is not None:mutable[n+'/bbox_coder']={k:state_value(v) for k,v in sorted(vars(coder).items())}
    return dict(registered=describe(registered),unregistered=mutable,training={n:m.training for n,m in model.named_modules()})


def state_value(v):
    if isinstance(v,torch.Tensor):return describe({'value':v})
    if isinstance(v,dict):return {str(k):state_value(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [state_value(x) for x in v]
    if v is None or isinstance(v,(str,int,float,bool)):return v
    return {'unsupported_type':type(v).__module__+'.'+type(v).__name__}


def numerical_difference(a,b):
    if list(a.shape)!=list(b.shape) or a.dtype!=b.dtype:return dict(comparable=False)
    if a.numel()==0:return dict(comparable=True,elements=0,max_abs=0.,mean_abs=0.)
    x=a.double();y=b.double();finite=torch.isfinite(x)&torch.isfinite(y);d=(x-y).abs()
    if not finite.all():return dict(comparable=True,finite=False,equal=torch.equal(a,b))
    return dict(comparable=True,finite=True,elements=a.numel(),max_abs=d.max().item(),mean_abs=d.mean().item(),rmse=((x-y).square().mean().sqrt()).item(),allclose_frozen=torch.allclose(x,y,atol=1e-4,rtol=1e-5))


def metadata(value):
    """Content/structure identity of non-tensor kwargs, including img_metas."""
    if isinstance(value,torch.Tensor):return {'tensor':True}
    if isinstance(value,type):return {'class':value.__module__+'.'+value.__qualname__}
    if isinstance(value,enum.Enum):return {'enum':type(value).__module__+'.'+type(value).__qualname__,'value':value.value}
    if hasattr(value,'tensor') and type(value).__module__.startswith('mmdet3d.core.bbox.structures'):
        return {'box_container':type(value).__module__+'.'+type(value).__qualname__,'attributes':{k:metadata(v) for k,v in vars(value).items()}}
    if isinstance(value,dict):return {str(k):metadata(v) for k,v in sorted(value.items())}
    if isinstance(value,(list,tuple)):return {'sequence_type':type(value).__name__,'items':[metadata(x) for x in value]}
    if isinstance(value,np.ndarray):return dict(numpy_sha256=hashlib.sha256(value.tobytes()).hexdigest(),dtype=str(value.dtype),shape=list(value.shape))
    if isinstance(value,np.generic):return value.item()
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError('Unprepared metadata type '+type(value).__name__)


def collated_cpu_payload(batch,container_type=None):
    """Single-device DataContainer unwrap with GPU-branch contiguous semantics.

    MMCV 1.4 CPU scatter adds a singleton axis to EVERY non-CPU tensor. It is
    unsuitable as a direct-module payload. No device transfer occurs here.
    """
    if container_type is None:
        from mmcv.parallel import DataContainer
        container_type=DataContainer
    def contiguous(value):
        if isinstance(value,torch.Tensor):return value.contiguous()
        if isinstance(value,list):return [contiguous(x) for x in value]
        if isinstance(value,tuple):return tuple(contiguous(x) for x in value)
        if isinstance(value,dict):return {k:contiguous(v) for k,v in value.items()}
        return value
    result={}
    for key,value in batch.items():
        if isinstance(value,container_type):
            if len(value.data)!=1:raise ValueError('Expected one-device DataContainer: '+key)
            result[key]=value.data[0] if value.cpu_only else contiguous(value.data[0])
        elif isinstance(value,torch.Tensor):result[key]=value.contiguous()
        else:raise ValueError('Unsupported collated field '+key+': '+type(value).__name__)
    return result


def validate_model_input(payload):
    image=payload['img'];points=payload['points'];metas=payload['metas']
    if image.ndim!=5 or image.shape[:3]!=(1,6,3):raise ValueError('Expected image B,N,C,H,W = 1,6,3,H,W')
    if len(points)!=1 or points[0].ndim!=2 or points[0].shape[1]!=5:raise ValueError('Expected one Nx5 point tensor')
    for key in ['camera2ego','lidar2camera','lidar2image','camera_intrinsics','camera2lidar','img_aug_matrix']:
        if tuple(payload[key].shape)!=(1,6,4,4):raise ValueError('Expected B,N,4,4 calibration: '+key)
    if tuple(payload['lidar2ego'].shape)!=(1,4,4) or tuple(payload['lidar_aug_matrix'].shape)!=(1,4,4):raise ValueError('Expected B,4,4 ego/LiDAR augmentation')
    if len(metas)!=1 or not isinstance(metas[0],dict):raise ValueError('Expected one metadata dictionary')
    return dict(status='PASS',img_shape=list(image.shape),points_shape=list(points[0].shape),coordinate_schema='LiDAR Nx5; six camera padded 4x4; single batch')

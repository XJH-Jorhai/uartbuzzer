"""Measured M2 electrical-current surrogate. Fixed existing 55+1 ohm platform."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from .uart_codec import ON

MODEL_VERSION='measured-M2-fixed55-sense1-ZOH-v1'

@dataclass(frozen=True)
class BeamModel:
    baud:int
    parameters:tuple
    model_hash:str
    A:np.ndarray
    bit_F:np.ndarray
    bit_g:np.ndarray
    half_F:np.ndarray
    half_g:np.ndarray
    F:np.ndarray
    g:np.ndarray
    H:np.ndarray
    q:np.ndarray

def model_identity(fit_path):
    content=Path(fit_path).read_bytes();fit=json.loads(content)
    if fit.get('selected')!='M2':raise ValueError('Beam需要已测M2电气模型')
    p=np.asarray(fit['selected_parameters'],float)
    if p.shape!=(10,) or not np.all(np.isfinite(p)) or np.any(p<=0):raise ValueError('无效M2参数')
    record=dict(version=MODEL_VERSION,fit_sha256=hashlib.sha256(content).hexdigest(),parameters=p.tolist(),Rlimit=55.,Rsense=1.,supply_V=5.)
    return hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest(),tuple(p)

def load_model(fit_path,baud):
    if baud not in (460800,921600):raise ValueError('Beam波特率需要460800或921600')
    digest,p=model_identity(fit_path)
    return _build(int(baud),p,digest)

@lru_cache(maxsize=8)
def _build(baud,p,digest):
    r,l,re,le,rm,lm,cm,r2,l2,c2=p;rt=r+56
    a=np.zeros((6,6));b=np.zeros(6)
    a[0]=[-(rt+re)/l,re/l,-rt/l,0,-rt/l,0];b[0]=rt/l
    a[1,:2]=[re/le,-re/le];a[2,[0,2,3]]=[1/(rt*cm),-1/(rm*cm),-1/(rt*cm)]
    a[3,2]=rt/lm;a[4,[0,4,5]]=[1/(rt*c2),-1/(r2*c2),-1/(rt*c2)];a[5,4]=rt/l2
    aug=np.zeros((7,7));aug[:6,:6]=a;aug[:6,6]=b
    e=expm(aug/baud);half=expm(aug/(2*baud));ad=e[:6,:6];bd=e[:6,6]
    states=np.zeros((256,6));h=[];q=[];power=np.eye(6)
    for k in range(10):
        h.append((half[:6,:6]@power)[0]);q.append((states@half[:6,:6].T)[:,0]+ON[:,k]*half[0,6])
        states=states@ad.T+ON[:,k,None]*bd;power=ad@power
    result=BeamModel(baud,p,digest,a,ad,bd,half[:6,:6],half[:6,6],power,states,np.array(h),np.array(q).T)
    for value in result.__dict__.values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
    return result

def finite_gram(F,H,frames):
    """Exact finite output observability Gramian, including explicit OFF gaps."""
    frames=int(frames);total=np.zeros((6,6));a=F.copy();q=H.T@H
    # Binary doubling: accumulate contiguous powers without O(frames) work.
    power=np.eye(6)
    while frames:
        if frames&1:total+=power.T@q@power;power=a@power
        q=q+a.T@q@a;a=a@a;frames>>=1
    return (total+total.T)/2

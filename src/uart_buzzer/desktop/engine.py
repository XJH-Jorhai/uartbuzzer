"""GUI-independent conversion, digital UART reconstruction and live-volume encoder."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfilt
from ..music import FS,KNOTS,read_audio,filter_audio,encode_pcm,fit_acoustic_response,electrical_equalizer
from ..nonlinear import validate_predistortion,predistort

COUNTS=np.array([9-int(v).bit_count() for v in range(256)],dtype=float)


@dataclass
class AudioDocument:
    name:str
    path:str
    source:np.ndarray
    processed:np.ndarray
    decoded:np.ndarray
    segments:list
    profile:dict
    metadata:dict
    base_processed:np.ndarray|None=None
    quadratic_correction:np.ndarray|None=None

    @property
    def duration(self):return len(self.processed)/FS


def validate_profile(profile):
    p=dict(profile)
    validate_predistortion(p.get('predistortion'))
    validate_predistortion(p.get('available_predistortion'))
    gains=np.asarray(p.get('gains_db'),float)
    if gains.shape!=(len(KNOTS),) or not np.all(np.isfinite(gains)) or np.max(abs(gains))>12:
        raise ValueError('配置需要13个有限增益点，范围±12dB')
    if p.get('knots_Hz')!=KNOTS.tolist():raise ValueError('配置的频率结点不匹配')
    if p.get('baud') not in (460800,921600):raise ValueError('波特率需要460800或921600')
    loudness=float(p.get('loudness_target_rms',.32))
    if not np.isfinite(loudness) or not .1<=loudness<=.7:raise ValueError('音乐响度目标需要0.1到0.7')
    bias=float(p.get('bias',.3));amplitude=float(p.get('amplitude',.18))
    if not (amplitude>0 and .1<=bias-amplitude and bias+amplitude<=.9):raise ValueError('配置超出UART密度范围')
    p['bias']=bias;p['amplitude']=amplitude;p['gains_db']=gains.tolist();p.setdefault('encoder','bit_feedback_uart_v1')
    if p.get('encoder','bit_feedback_uart_v1') not in ('bit_feedback_uart_v1','frame_density_feedback','frame_aware_sigma_delta','edge_feedback_uart_v1'):
        raise ValueError('未知编码器')
    edge=p.get('edge_model') or {}
    if any(k not in ('on_loss','off_tail','bias_offset') or not np.isfinite(v) or abs(v)>(.15 if k=='bias_offset' else .4) or (k!='bias_offset' and v<0) for k,v in edge.items()):
        raise ValueError('电路边沿模型超出有界范围')
    return p


def decode_segments(segments,length,baud,bias,amplitude):
    """Digital frame-density reconstruction. Not a coil-current measurement."""
    result=np.zeros(length,dtype=np.float32)
    sos=butter(4,[400,5000],btype='bandpass',fs=baud/10,output='sos')
    for seg in segments:
        values=(COUNTS[np.frombuffer(seg['data'],np.uint8)]/10-bias)/amplitude
        values=sosfilt(sos,values)
        first=max(0,int(seg['start_s']*FS));last=min(length,int(seg['end_s']*FS))
        if last<=first or not len(values):continue
        t=seg['start_s']+(np.arange(len(values))+.5)*10/baud
        result[first:last]=np.interp(np.arange(first,last)/FS,t,values)
    return result


def convert(path,profile,mode='music',fit_path=None,progress=None,cancel_check=None,source=None,name=None):
    p=validate_profile(profile)
    if source is None:x,_=read_audio(path)
    else:x=np.asarray(source,float)
    if len(x)<FS*.1:raise ValueError('音频至少需要0.1秒')
    if len(x)>FS*600:raise ValueError('此版本支持最长10分钟WAV，请先裁剪更长文件')
    if progress:progress(.05)
    if mode=='electrical':
        gains,model=electrical_equalizer(fit_path)
        p['gains_db']=gains.tolist();p['name']='电气模型补偿'
    elif mode=='bypass':p['gains_db']=[0.]*len(KNOTS)
    else:model=None
    y,conditioning=filter_audio(x,np.array(p['gains_db']),loudness=p.get('conditioning')=='music_loudness_lookahead_v2',
                               loudness_target=p.get('loudness_target_rms',.32))
    base=None;correction=None;pd_meta=dict(enabled=False)
    if p.get('predistortion'):
        base=y.astype(np.float32);y,correction,pd_meta=predistort(y,p['predistortion'])
        correction=correction.astype(np.float32)
    if progress:progress(.15)
    segments,codec=encode_pcm(y,baud=p['baud'],bias=p['bias'],amplitude=p['amplitude'],
             method=p.get('encoder','bit_feedback_uart_v1'),
             continuous=p.get('continuous_carrier',False),soft_edges=p.get('soft_carrier_edges',False),
             edge_model=p.get('edge_model'),
             progress=(lambda v:progress(.15+.80*v)) if progress else None,cancel_check=cancel_check)
    if cancel_check is not None and cancel_check():raise InterruptedError('Conversion cancelled')
    decoded=decode_segments(segments,len(y),p['baud'],p['bias'],p['amplitude'])
    digest=hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest()
    metadata=dict(conditioning=conditioning,predistortion=pd_meta,encoding=codec,mode=mode,source_PCM_sha256=digest,
                  total_bytes=sum(len(s['data']) for s in segments),sample_rate_Hz=FS,
                  reconstruction='frame density + 400–5000Hz digital bandpass; not measured coil current')
    if progress:progress(1.)
    return AudioDocument(name or Path(path).name,str(path),x.astype(np.float32),y.astype(np.float32),decoded,segments,p,metadata,base,correction)


class LiveEncoder:
    """Preserves feedback and ramps volume across blocks. Zero volume emits no bytes."""
    def __init__(self,profile):
        self.p=validate_profile(profile);self.state=0.;self.gain=0.;self.last_idle=True;self.carrier=0.
        self._source_id=None;self._source_time=None

    def block(self,source,start_frame,frames,volume=1.,active=True,correction=None):
        baud=self.p['baud'];target_gain=float(np.clip(volume,0,1))
        bit_times=(np.arange(start_frame*10,(start_frame+frames)*10)+.5)/baud
        if self._source_id!=id(source):
            self._source_time=np.arange(len(source))/FS;self._source_id=id(source)
        raw=np.interp(bit_times,self._source_time,source)
        fade=min(len(raw),max(1,int(baud*.01)))
        ramp=np.r_[np.linspace(self.gain,target_gain,fade),np.full(len(raw)-fade,target_gain)]
        self.gain=target_gain
        pcm=raw*ramp
        if correction is not None:pcm+=np.interp(bit_times,self._source_time,correction)*ramp**2
        if not active or (target_gain==0 and np.max(abs(pcm))<1e-12):
            self.state=0.;self.last_idle=True;self.carrier=0.
            return b'',np.zeros(len(pcm),np.float32)
        target=self.p['bias']+self.p['amplitude']*pcm
        if self.p.get('soft_carrier_edges',False):
            carrier_target=1. if target_gain>0 else 0.
            nf=min(len(raw),max(1,int(baud*.016)))
            edge=np.r_[np.linspace(self.carrier,carrier_target,nf),np.full(len(raw)-nf,carrier_target)]
            edge=np.minimum(edge,np.clip((len(source)/FS-bit_times)/.016,0,1))
            target=.1+(target-.1)*edge;self.carrier=carrier_target
        if self.last_idle:self.state=0.
        self.last_idle=False
        method=self.p.get('encoder','bit_feedback_uart_v1')
        if method=='edge_feedback_uart_v1':
            from ..uart_codec import edge_feedback_block
            data,self.state=edge_feedback_block(target,self.state,**self.p.get('edge_model',{}))
            return data,pcm
        if method=='frame_aware_sigma_delta':
            from ..uart_codec import encode
            data,self.state,_=encode(target,'frame_aware_sigma_delta',self.state)
            return data,pcm
        data=bytearray(frames)
        if method=='frame_density_feedback':
            for frame,demand in enumerate(target.reshape(-1,10).sum(axis=1)):
                self.state+=demand;n=int(np.clip(np.floor(self.state-.5),0,8))
                self.state-=1+n;data[frame]=255^((1<<n)-1)
        else:
            from ..uart_codec import bit_feedback_block
            encoded,self.state=bit_feedback_block(target,self.state)
            return encoded,pcm
        return bytes(data),pcm


def fit_recording(reference,capture,prior_profile,fit_path):
    prior,_=electrical_equalizer(fit_path)
    result=fit_acoustic_response(reference,capture,prior_response_db=-prior)
    p=dict(prior_profile);p.update(name='本机音乐录音拟合',gains_db=result['inverse_correction_db'],
                                  clock_scale_estimate=result['clock_scale'],status='user_recording_fit_needs_other_music_validation')
    return validate_profile(p),result


def export_document(doc,folder,volume=1.):
    """The JSON schedule is necessary: concatenated BIN alone loses actual rests."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    pcm=(doc.base_processed*volume+doc.quadratic_correction*volume**2) if doc.quadratic_correction is not None else doc.processed*volume
    segments,info=encode_pcm(pcm,baud=doc.profile['baud'],bias=doc.profile['bias'],
                            amplitude=doc.profile['amplitude'],method=doc.profile.get('encoder','bit_feedback_uart_v1'),
                            continuous=doc.profile.get('continuous_carrier',False),soft_edges=doc.profile.get('soft_carrier_edges',False),
                            edge_model=doc.profile.get('edge_model'))
    records=[]
    for j,s in enumerate(segments):
        name=f'segment_{j:04d}.bin';(folder/name).write_bytes(s['data'])
        records.append(dict(file=name,start_s=s['start_s'],end_s=s['end_s'],bytes=len(s['data']),
                            sha256=hashlib.sha256(s['data']).hexdigest()))
    metadata=dict(source=doc.name,volume=volume,profile=doc.profile,encoding=info,segments=records,
                  note='Stop TX between scheduled segments; 0xff is not silence')
    (folder/'uart_audio.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    return folder/'uart_audio.json'

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
    baseline_segments:list|None=None
    beam_pcm:np.ndarray|None=None

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
    if p.get('encoder','bit_feedback_uart_v1') not in ('bit_feedback_uart_v1','frame_density_feedback','frame_aware_sigma_delta','edge_feedback_uart_v1','beam_fast','beam_quality','beam_research','discrete_score'):
        raise ValueError('未知编码器')
    if p['encoder']=='discrete_score':
        from ..score.pipeline import settings
        p['score_config']=settings(p.get('score_config'));p['prepared_volume']=float(p.get('prepared_volume',.8))
        if not np.isfinite(p['prepared_volume']) or not 0<=p['prepared_volume']<=1:raise ValueError('曲谱音量无效')
    if p['encoder'].startswith('beam_'):
        from ..beam import settings_for,LEGACY_MODES
        from dataclasses import asdict
        p['beam_settings']=asdict(settings_for(p['encoder'],p.get('beam_settings')))
        p.setdefault('beam_baseline_encoder','bit_feedback_uart_v1')
        if p['beam_baseline_encoder'] not in LEGACY_MODES:raise ValueError('Beam基线必须为原有编码器')
        volume=float(p.get('beam_volume',1.))
        if not np.isfinite(volume) or not 0<=volume<=1:raise ValueError('Beam预计算音量需要0到100%')
        p['beam_volume']=volume
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
    if p['encoder']=='discrete_score':
        from ..score.pipeline import convert as convert_score
        result=convert_score(x,p['score_config'],p['baud'],p['prepared_volume'],p['bias'],p['amplitude'],progress,cancel_check)
        metadata=dict(mode='discrete_score',source_PCM_sha256=hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest(),
            total_bytes=sum(len(s['data']) for s in result['segments']),sample_rate_Hz=FS,score=result['score'],transport_plan=result['transport_plan'],
            conditioning=dict(kind='raw_source_before_legacy_EQ'),predistortion=dict(enabled=False),
            reconstruction='UART frame density + low-pass; not physical SPL',encoding=dict(encoder='discrete_score',format='8N1',baud=p['baud']))
        return AudioDocument(name or Path(path).name,str(path),x.astype(np.float32),result['processed'],result['decoded'],result['segments'],p,metadata)
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
    is_beam=p['encoder'].startswith('beam_');beam_meta=None
    prepared=y
    if is_beam:
        # Preserve the SAME high precision front-end output used by the old encoder.
        volume=p['beam_volume']
        prepared=y
    segments,codec=encode_pcm(prepared,baud=p['baud'],bias=p['bias'],amplitude=p['amplitude'],
             method=p.get('beam_baseline_encoder','bit_feedback_uart_v1') if is_beam else p.get('encoder','bit_feedback_uart_v1'),
             continuous=p.get('continuous_carrier',False),soft_edges=p.get('soft_carrier_edges',False),
             edge_model=p.get('edge_model'),
             progress=(lambda v:progress(.15+(.05 if is_beam else .80)*v)) if progress else None,cancel_check=cancel_check)
    if is_beam:
        from ..beam import encode_segments
        if fit_path is None:raise ValueError('Beam需要固定M2模型路径')
        prepared=(base if base is not None else y.astype(np.float32)).astype(float)
        baseline_segments,plan=live_baseline_plan(prepared,correction,segments,p,cancel_check)
        result=encode_segments(prepared,baseline_segments,p,fit_path,progress=(lambda v:progress(.2+.75*v)) if progress else None,cancel_check=cancel_check,
                               command_factory=plan.command,prep_hash=plan.digest())
        segments=result.segments;beam_meta=result.diagnostics
        beam_meta['live_target_blocks']=plan.records
        beam_meta['baseline_path']='Actual LiveEncoder with 2048-frame Playback boundaries, fixed gain, original ramps and idle protocol'
        codec=dict(codec,encoder=p['encoder'],beam=beam_meta,active_wire_s=result.expected_duration,
                   real_idle_s=len(y)/FS-result.expected_duration)
        codec.pop('segment_final_integral_errors',None)
    if cancel_check is not None and cancel_check():raise InterruptedError('Conversion cancelled')
    decoded=decode_segments(segments,len(y),p['baud'],p['bias'],p['amplitude'])
    digest=hashlib.sha256(np.asarray(x,dtype=np.float32).tobytes()).hexdigest()
    metadata=dict(conditioning=conditioning,predistortion=pd_meta,encoding=codec,mode=mode,source_PCM_sha256=digest,
                  total_bytes=sum(len(s['data']) for s in segments),sample_rate_Hz=FS,
                  reconstruction='frame density + 400–5000Hz digital bandpass; not measured coil current')
    if beam_meta is not None:metadata['beam']=beam_meta
    if progress:progress(1.)
    return AudioDocument(name or Path(path).name,str(path),x.astype(np.float32),y.astype(np.float32),decoded,segments,p,metadata,base,correction,
                         baseline_segments if is_beam else None,prepared.copy() if is_beam else None)


class LiveEncoder:
    """Preserves feedback and ramps volume across blocks. Zero volume emits no bytes."""
    def __init__(self,profile):
        self.p=validate_profile(profile);self.state=0.;self.gain=0.;self.last_idle=True;self.carrier=0.
        if self.p['encoder'].startswith('beam_'):raise ValueError('Beam必须使用已完成的离线预计算字节')
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


class PrecomputedEncoder:
    """Thin adapter to the existing playback protocol, with fixed prepared volume."""
    def __init__(self,document):
        self.doc=document;self.p=document.profile;self.volume=float(self.p.get('prepared_volume',self.p.get('beam_volume',1.)))
        if not (document.metadata.get('beam') or document.metadata.get('score')):raise ValueError('离线字节尚未完成')
        self.bounds=[(round(s['start_s']*self.p['baud']/10),s) for s in document.segments]
        self.cursor=0

    def block(self,source,start_frame,frames,volume=1.,active=True,correction=None):
        if not active or volume==0:return b'',np.zeros(frames*10,np.float32)
        if abs(volume-self.volume)>1e-9:raise ValueError('离线固定音量已改变，请先停止并重新编码')
        while self.cursor<len(self.bounds) and start_frame>=self.bounds[self.cursor][0]+len(self.bounds[self.cursor][1]['data']):self.cursor+=1
        if self.cursor>=len(self.bounds):return b'',np.zeros(frames*10,np.float32)
        begin,seg=self.bounds[self.cursor];offset=start_frame-begin
        if offset<0 or offset+frames>len(seg['data']):raise ValueError('离线播放块跨越休止边界')
        return seg['data'][offset:offset+frames],np.zeros(frames*10,np.float32)


class LiveTargetPlan:
    """Replay the old LiveEncoder's exact target without storing a bit-rate full-song array."""
    def __init__(self,source,correction,profile,records):
        self.source=np.asarray(source,float);self.correction=correction;self.p=profile;self.records=records
        self.times=np.arange(len(source))/FS;self.starts=np.array([r['start'] for r in records],dtype=int)
        self.memo={}

    def command(self,start,frames):
        baud=self.p['baud'];out=np.zeros(frames*10);stop=start+frames
        first=max(0,int(np.searchsorted(self.starts,start,side='right'))-1)
        for index in range(first,len(self.records)):
            r=self.records[index];a=r['start'];b=a+r['frames']
            if a>=stop:break
            if b<=start:continue
            target=self.memo.get(index)
            if target is None:
                bt=(np.arange(a*10,b*10)+.5)/baud;raw=np.interp(bt,self.times,self.source)
                fade=min(len(raw),max(1,int(baud*.01)))
                ramp=np.r_[np.linspace(r['gain_start'],r['gain_end'],fade),np.full(len(raw)-fade,r['gain_end'])]
                pcm=raw*ramp
                if self.correction is not None:pcm+=np.interp(bt,self.times,self.correction)*ramp**2
                target=self.p['bias']+self.p['amplitude']*pcm
                if self.p.get('soft_carrier_edges',False):
                    nf=min(len(raw),max(1,int(baud*.016)))
                    edge=np.r_[np.linspace(r['carrier_start'],1.,nf),np.ones(len(raw)-nf)]
                    edge=np.minimum(edge,np.clip((len(self.source)/FS-bt)/.016,0,1));target=.1+(target-.1)*edge
                if len(self.memo)>6:self.memo.clear()
                self.memo[index]=target
            lo=max(start,a);hi=min(stop,b);out[(lo-start)*10:(hi-start)*10]=target[(lo-a)*10:(hi-a)*10]
        return out

    def digest(self):
        h=hashlib.sha256(self.source.astype(np.float64).tobytes())
        if self.correction is not None:h.update(np.asarray(self.correction).tobytes())
        h.update(json.dumps(dict(records=self.records,baud=self.p['baud'],bias=self.p['bias'],amplitude=self.p['amplitude'],soft_edges=self.p.get('soft_carrier_edges',False)),sort_keys=True).encode())
        return h.hexdigest()


def live_baseline_plan(source,correction,segments,profile,cancel_check=None):
    """Exactly the existing Playback.run frame boundaries, with no serial port."""
    p=dict(profile,encoder=profile.get('beam_baseline_encoder','bit_feedback_uart_v1'))
    enc=LiveEncoder(p);baud=p['baud'];n=int(len(source)/FS*baud/10);frame=0;index=0;records=[];parts=[bytearray() for _ in segments]
    bounds=[(round(s['start_s']*baud/10),round(s['end_s']*baud/10)) for s in segments]
    while frame<n:
        if cancel_check and cancel_check():raise InterruptedError('Beam编码已取消')
        while index<len(bounds) and frame>=bounds[index][1]:index+=1
        active=index<len(bounds) and frame>=bounds[index][0]
        boundary=(bounds[index][1] if active else bounds[index][0]) if index<len(bounds) else n
        count=min(2048,n-frame,max(1,boundary-frame));gain=enc.gain;carrier=enc.carrier
        data,_=enc.block(source,frame,count,profile['beam_volume'],active,correction)
        if data:
            parts[index].extend(data);records.append(dict(start=frame,frames=count,gain_start=gain,gain_end=profile['beam_volume'],carrier_start=carrier))
        frame+=count
    result=[dict(s,data=bytes(data)) for s,data in zip(segments,parts) if data]
    return result,LiveTargetPlan(source,correction,profile,records)


def beam_ab_preview(doc,fit_path,seconds=.25):
    """Same stored reference/baseline/bytes, same gain. Preview is explicitly time-limited."""
    from ..beam_model import load_model
    from ..beam import command_block
    from .._beam_kernels import trajectory,reference
    if doc.beam_pcm is None or doc.baseline_segments is None:raise ValueError('请先完成Beam离线编码')
    model=load_model(fit_path,doc.profile['baud']);baud=model.baud
    if model.model_hash!=doc.metadata['beam']['model_hash']:raise ValueError('模型已改变，请重新编码')
    n=min(int(seconds*baud/10),int(doc.duration*baud/10));active=np.zeros(n,bool)
    chosen=np.full(n,256,np.uint16);baseline=np.full(n,256,np.uint16);command=np.zeros(n*10)
    plan=LiveTargetPlan(doc.beam_pcm,doc.quadratic_correction,doc.profile,doc.metadata['beam']['live_target_blocks'])
    for own,old in zip(doc.segments,doc.baseline_segments):
        start=round(own['start_s']*baud/10);end=start+len(own['data']);count=max(0,min(end,n)-start)
        if not count:continue
        chosen[start:start+count]=np.frombuffer(own['data'],np.uint8)[:count]
        baseline[start:start+count]=np.frombuffer(old['data'],np.uint8)[:count];active[start:start+count]=True
        command[start*10:(start+count)*10]=plan.command(start,count)
    g=np.vstack((model.g,np.zeros((1,6))));q=np.vstack((model.q,np.zeros((1,10))))
    current,_=trajectory(chosen,model.F,g,model.H,q,np.zeros(6));old,_=trajectory(baseline,model.F,g,model.H,q,np.zeros(6))
    target,_=reference(command,model.bit_F,model.bit_g,model.half_F,model.half_g,np.zeros(6))
    ff=np.fft.rfftfreq(n*10,1/baud);win=np.hanning(n*10);spectra={}
    scale=5000/(model.parameters[0]+56)
    for name,values in (('Beam',current),('原有',old),('目标',target)):
        y=values.ravel()*scale;spectra[name]=abs(np.fft.rfft((y-y.mean())*win))*2/win.sum()
    t=(np.arange(n*10)+.5)/baud;use=t>max(0,t[-1]-.020);step=max(1,int(np.sum(use)/5000))
    return dict(time_s=t[use][::step],current_mA=current.ravel()[use][::step]*scale,
        baseline_mA=old.ravel()[use][::step]*scale,target_mA=target.ravel()[use][::step]*scale,
        frequency_Hz=ff[ff<=10000],spectra={k:v[ff<=10000] for k,v in spectra.items()},
        diagnostics=doc.metadata['beam'],preview_seconds=n*10/baud,
        note='Same stored PCM, gain, model and initial state; bit-center spectrum preview, not microphone data')


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
    if doc.profile['encoder']=='discrete_score':
        if doc.metadata['transport_plan']['backend']=='generic_score_only':raise ValueError('通用档案只导出曲谱JSON/MIDI，没有UART数据')
        if abs(volume-doc.profile['prepared_volume'])>1e-9:raise ValueError('曲谱音量与预计算不一致，请重新转换')
        segments=doc.segments;info=dict(doc.metadata['encoding'],transport_plan=doc.metadata['transport_plan'])
    elif doc.profile['encoder'].startswith('beam_'):
        if abs(volume-doc.profile['beam_volume'])>1e-9:raise ValueError('Beam导出音量与预计算不一致，请重新编码')
        segments=doc.segments;info=doc.metadata['encoding']
    else:segments,info=encode_pcm(pcm,baud=doc.profile['baud'],bias=doc.profile['bias'],
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

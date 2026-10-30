"""Explicit short-tone measurement. Imported safely; I/O starts only in run()."""
import hashlib,json,threading,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from ..score.contracts import flat_profile,device_profile
from ..score.model import midi_to_hz,digest

VERSION='relative-tone-characterization-v1'


def analyze_tone(samples,rate,command_hz,background_rms,statuses=()):
    x=np.asarray(samples,float)
    if x.ndim!=1 or len(x)<rate*.15 or not np.isfinite(x).all():raise ValueError('采集音频不足或无效')
    clip=float(np.mean(abs(x)>=.999));width=max(1,round(rate*.01))
    env=np.sqrt(np.convolve(x*x,np.ones(width)/width,mode='same'))
    threshold=max(background_rms*4,float(env.max())*.15,1e-6);active=np.flatnonzero(env>threshold)
    steady=x[round(rate*.04):max(round(rate*.05),len(x)-round(rate*.04))];steady=steady-np.mean(steady)
    nfft=max(8192,1<<(len(steady)-1).bit_length());mag=abs(np.fft.rfft(steady*np.hanning(len(steady)),n=nfft))*2/max(1,np.hanning(len(steady)).sum())
    freq=np.fft.rfftfreq(nfft,1/rate);mag[freq<30]=0
    def peak_at(f):
        ids=np.flatnonzero(abs(freq-f)<=max(4,rate/nfft*2,f*.025))
        k=int(ids[np.argmax(mag[ids])]) if len(ids) else 0
        return float(mag[k]),float(freq[k])
    fundamental,observed=peak_at(command_hz);sub,sub_hz=peak_at(command_hz/2);double,double_hz=peak_at(command_hz*2)
    dominant=int(np.argmax(mag));largest=float(mag[dominant]);snr=20*np.log10(max(fundamental/np.sqrt(2),1e-12)/max(background_rms,1e-7))
    cents=1200*np.log2(max(observed,1e-9)/command_hz);reasons=[]
    if clip>.001:reasons.append('ADC削波')
    if statuses:reasons.append('采集/串口状态错误')
    if snr<12:reasons.append('基音信噪比不足')
    if abs(cents)>50:reasons.append('命令基音偏差过大')
    if largest>fundamental*2 or double>fundamental*1.8 or sub>fundamental*1.8:reasons.append('谐波或次谐波主导，基音不可确认')
    confidence=float(np.clip((snr-6)/24,0,1)*min(1.,fundamental/max(largest,1e-12)))
    if confidence<.5:reasons.append('可靠度不足')
    valid=not reasons
    return dict(command_frequency_hz=float(command_hz),actual_fundamental_hz=observed if valid else None,
        candidate_near_command_hz=observed,pitch_error_cents=float(cents),fundamental_amplitude=fundamental,
        half_frequency_peak_hz=sub_hz,half_frequency_amplitude=sub,double_frequency_peak_hz=double_hz,double_frequency_amplitude=double,
        largest_spectral_peak_hz=float(freq[dominant]),largest_spectral_amplitude=largest,identifiable=valid,
        RMS=float(np.sqrt(np.mean(steady**2))),background_RMS=float(background_rms),fundamental_SNR_dB=float(snr),
        clipped_fraction=clip,confidence=confidence,valid=valid,reasons=reasons,
        onset_s_from_capture_start=float(active[0]/rate) if len(active) else None,
        tail_s_from_capture_start=float(active[-1]/rate) if len(active) else None,
        timing_scope='capture-relative threshold crossing; driver/acoustic latency not calibrated')


class HardwareIO:
    """Owns fresh streams; never borrows current playback/recording objects."""
    def __init__(self,config):self.config=config;self.lock=threading.Lock();self.parts=[];self.total=0;self.statuses=[];self.stream=None;self.link=None
    def __enter__(self):
        import sounddevice as sd
        from .devices import SerialLink
        self.link=SerialLink()
        try:
            self.link.open(self.config['port'],self.config['baud']);self.rate=self.config['rate']
            def capture(data,frames,timing,status):
                with self.lock:
                    self.parts.append(data[:,0].copy());self.total+=len(data)
                    if status:self.statuses.append(str(status))
            self.stream=sd.InputStream(device=self.config['microphone'],samplerate=self.rate,channels=1,dtype='float32',callback=capture)
            self.stream.start();return self
        except Exception:self.__exit__(None,None,None);raise
    def __exit__(self,*args):
        try:
            if self.stream:
                try:self.stream.stop()
                finally:self.stream.close()
        finally:
            if self.link:
                try:
                    if self.link.connected:self.link.port.reset_output_buffer()
                finally:self.link.close()
    def wait(self,seconds,cancel):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            if cancel and cancel():raise InterruptedError('特性测试已取消')
            time.sleep(min(.005,max(0,until-time.monotonic())))
    def position(self):
        with self.lock:return self.total
    def since(self,start):
        with self.lock:
            data=np.concatenate(self.parts) if self.parts else np.zeros(0,np.float32)
        return data[start:]
    def background(self,seconds,cancel):
        start=self.position();self.wait(seconds,cancel);return self.since(start)
    def note(self,hz,volume,duration,gap,cancel):
        from ..score.uart_renderer import render_tone,transport_signature
        baud=self.config['baud'];frames=round(duration*baud/10)
        pcm,_,segments,plan=render_tone(hz,duration,baud,volume,self.config['bias'],self.config['amplitude'],cancel)
        payload=b''.join(s['data'] for s in segments);start=self.position();clock=time.monotonic()
        for a in range(0,len(payload),2048):
            if cancel and cancel():raise InterruptedError('特性测试已取消')
            data=payload[a:a+2048]
            deadline=time.monotonic()+2
            while self.link.port.out_waiting>4096:
                if time.monotonic()>deadline:raise RuntimeError('串口发送阻塞')
                self.wait(.002,cancel)
            if self.link.port.write(data)!=len(data):raise RuntimeError('串口短写')
        deadline=time.monotonic()+2
        while self.link.port.out_waiting:
            if time.monotonic()>deadline:raise RuntimeError('串口排空超时')
            self.wait(.002,cancel)
        self.wait(gap,cancel)
        return self.since(start),dict(elapsed_s=time.monotonic()-clock,statuses=list(self.statuses),idle_strategy='stop_writes_after_tone',driver_latency_unknown=True,
            payload_sha256=hashlib.sha256(payload).hexdigest(),reference_PCM_sha256=hashlib.sha256(np.asarray(pcm,np.float32).tobytes()).hexdigest(),
            frames=frames,format='8N1',reference_transport=transport_signature(baud,volume,self.config['bias'],self.config['amplitude']))


def run(config,folder,cancel_check=None,progress=None,io_factory=HardwareIO):
    cfg=dict(config);folder=Path(folder)
    from ..score.uart_renderer import transport_signature,VERSION as renderer_version,render_tone
    if not isinstance(cfg.get('name'),str) or not 1<=len(cfg['name'].strip())<=120:raise ValueError('请输入1–120字符的新档案名称')
    low=int(cfg['midi_min']);high=int(cfg['midi_max']);volume=float(cfg['volume']);duration=float(cfg['duration_s']);gap=float(cfg.get('gap_s',.2))
    if (not cfg.get('port') or cfg.get('microphone') is None or not 36<=low<=high<=96 or high-low>=24 or
        not .1<=volume<=1 or not .25<=duration<=1 or not .15<=gap<=.5 or cfg['baud'] not in (460800,921600) or
        not 8000<=cfg['rate']<=96000 or not .1<=cfg['bias']-cfg['amplitude']<=cfg['bias']+cfg['amplitude']<=.9 or
        cfg.get('renderer')!=renderer_version):raise ValueError('特性测试参数无效或超过24音高范围')
    if folder.exists():raise FileExistsError('特性测试需要新的会话目录')
    folder.mkdir(parents=True);records=[];report=dict(schema='finite-note-characterization-v1',version=VERSION,config=cfg,
        started_UTC=datetime.now(timezone.utc).isoformat(),records=records,complete=False,cancelled=False,absolute_SPL='unknown',
        linearity_model='two command levels only; relative gain × velocity approximation',physical_measurement=io_factory is HardwareIO)
    def save():
        tmp=folder/'evidence.pending';tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(folder/'evidence.json')
    try:
        if cancel_check and cancel_check():raise InterruptedError('特性测试已取消')
        with io_factory(cfg) as io:
            bg=io.background(.5,cancel_check)
            if len(bg)<cfg['rate']*.3:raise ValueError('背景采集失败')
            noise=float(np.sqrt(np.mean(np.asarray(bg,float)**2)));wavfile.write(folder/'background.wav',cfg['rate'],bg)
            if np.mean(abs(np.asarray(bg))>=.999)>.001:raise ValueError('背景采集已削波')
            report['background_sha256']=hashlib.sha256((folder/'background.wav').read_bytes()).hexdigest()
            reliable=[]
            for midi in range(low,high+1):
                frequency=float(midi_to_hz(midi));levels=[]
                for index,gain in enumerate((volume/2,volume)):
                    if cancel_check and cancel_check():raise InterruptedError('特性测试已取消')
                    samples,meta=io.note(frequency,gain,duration,gap,cancel_check)
                    name=f'note_{midi}_{index}.wav';wavfile.write(folder/name,cfg['rate'],np.asarray(samples,np.float32))
                    # Analyze sustained command region, retain raw capture with gap/tail.
                    a=analyze_tone(samples[:round(cfg['rate']*duration)],cfg['rate'],frequency,noise,meta.get('statuses',[]))
                    if len(samples)<cfg['rate']*duration*.85:a.update(valid=False,reasons=a['reasons']+['采集长度不足'])
                    tail=np.asarray(samples[-round(cfg['rate']*.05):],float)
                    tail_rms=float(np.sqrt(np.mean(tail**2))) if len(tail) else 0.
                    a['post_tone_noise_RMS']=tail_rms
                    if tail_rms>max(noise*4,.003):a.update(valid=False,reasons=a['reasons']+['尾音未衰减或背景漂移'])
                    ref,_,_,_=render_tone(frequency,duration,cfg['baud'],gain,cfg['bias'],cfg['amplitude'],cancel_check);ref=ref.astype(np.float32)
                    ref_name=f'reference_{midi}_{index}.wav';wavfile.write(folder/ref_name,16000,ref)
                    a.update(reference_file=ref_name,reference_PCM_sha256=hashlib.sha256(ref.tobytes()).hexdigest())
                    a.update(midi=midi,command_volume=gain,capture=name,capture_sha256=hashlib.sha256((folder/name).read_bytes()).hexdigest(),transport=meta)
                    levels.append(a);records.append(a)
                    if progress:progress(len(records)/(2*(high-low+1)))
                    save()
                ratio=levels[1]['fundamental_amplitude']/max(levels[0]['fundamental_amplitude'],1e-12)
                if all(v['valid'] for v in levels) and abs(ratio/2-1)<=.35:
                    reliable.append(dict(id=f'MIDI{midi}',frequency_hz=frequency,gain=levels[1]['fundamental_amplitude'],confidence=min(v['confidence'] for v in levels)))
                else:
                    for v in levels:
                        if abs(ratio/2-1)>.35:v.update(valid=False,reasons=v['reasons']+['两档音量不符合线性近似'])
            if not reliable:raise ValueError('没有可靠音高点；失败记录已保留，不生成有效档案')
            maximum=max(n['gain'] for n in reliable)
            for note in reliable:note['gain']=note['gain']/maximum
            physical=io_factory is HardwareIO
            p=flat_profile(cfg['name'],low,high);p.update(response_kind='relative_acoustic' if physical else 'synthetic_relative',notes=reliable,velocity_levels=[.5,1.],
                provenance=dict(measured=physical,reference_volume=volume,reference_transport=transport_signature(cfg['baud'],volume,cfg['bias'],cfg['amplitude']),
                                reference_transport_sha256=digest(transport_signature(cfg['baud'],volume,cfg['bias'],cfg['amplitude'])),measurement_version=VERSION,evidence_sha256=None,
                                microphone=cfg['microphone'],port=cfg['port'],position_description=cfg.get('position',''),
                                measured_UTC=report['started_UTC'],heuristic_confidence=True,absolute_SPL='unknown',
                                scope='fixed microphone/gain/position, two levels; repeat after hardware or placement change'))
            report.update(complete=True,reliable_notes=len(reliable));save()
            p['provenance']['evidence_sha256']=hashlib.sha256((folder/'evidence.json').read_bytes()).hexdigest();p=device_profile(p)
            (folder/'pending_profile.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
            return p
    except Exception as e:
        report.update(error=str(e),cancelled=isinstance(e,InterruptedError));save();raise

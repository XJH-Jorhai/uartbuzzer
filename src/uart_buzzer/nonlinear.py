"""Bounded, volume-aware second-order memory predistortion for arbitrary music.

The kernel is identified from polarity-paired music; no frequency/tone detector.
WAV preparation is offline, so a short centered FIR may anticipate transients.
"""
import numpy as np
from scipy.signal import butter,sosfiltfilt,resample_poly,fftconvolve,correlate,csd,welch,coherence,stft
from scipy.ndimage import minimum_filter1d,uniform_filter1d,gaussian_filter1d,map_coordinates
from scipy.stats import theilslopes
from scipy.optimize import minimize_scalar


def quadratic_feature(x,rate=16000):
    x=np.asarray(x,float)
    high=resample_poly(x,4,1)
    squared=high*high
    squared=sosfiltfilt(butter(4,[400,4800],fs=rate*4,btype='bandpass',output='sos'),squared)
    return resample_poly(squared,1,4)[:len(x)]


def validate_predistortion(model):
    if model is None:return None
    if model.get('kind')!='quadratic_memory_v1':raise ValueError('未知预失真模型')
    if model.get('sample_rate_Hz',16000)!=16000:raise ValueError('预失真模型采样率需要16kHz')
    kernel=np.asarray(model.get('kernel2'),float)
    if kernel.ndim!=1 or len(kernel)<3 or len(kernel)>1025 or len(kernel)%2!=1 or not np.all(np.isfinite(kernel)):
        raise ValueError('预失真 FIR 需要3到1025个有限系数，长度为奇数')
    strength=float(model.get('strength',1))
    if not np.isfinite(strength) or abs(strength)>1.5 or np.sum(abs(kernel))>6:
        raise ValueError('预失真系数超出有界范围')
    return model


def predistort(x,model,rate=16000):
    validate_predistortion(model);x=np.asarray(x,float)
    if not model or model.get('strength',1)==0:return x.copy(),np.zeros_like(x),dict(enabled=False)
    correction=fftconvolve(quadratic_feature(x,rate),np.asarray(model['kernel2']),mode='same')*model.get('strength',1)
    # Keep the quadratic correction modest; never normalize down the clean path.
    limit=.25*max(np.sqrt(np.mean(x*x)),1e-12)
    correction*=min(1.,limit/max(np.sqrt(np.mean(correction**2)),1e-15))
    required=np.ones(len(x));positive=correction>1e-15;negative=correction<-1e-15
    required[positive]=np.clip((.985-x[positive])/correction[positive],0,1)
    required[negative]=np.clip((-.985-x[negative])/correction[negative],0,1)
    radius=max(1,round(.003*rate))
    envelope=uniform_filter1d(minimum_filter1d(required,2*radius+1,mode='nearest'),2*radius+1,mode='nearest')
    correction*=envelope;y=x+correction
    return y,correction,dict(enabled=True,kind=model['kind'],strength=model.get('strength',1),
        correction_RMS=float(np.std(correction)),clean_RMS=float(np.std(x)),peak=float(max(abs(y))),
        correction_limited_fraction=float(np.mean(envelope<.999)),volume_scaling='clean*v + quadratic_correction*v^2')


def align_repeat(reference,signal,polarity=1,rate=16000):
    """Subsample relative delay/drift, with independent windows and robust slope."""
    sos=butter(3,[1300,4500],fs=rate,btype='bandpass',output='sos')
    a=sosfiltfilt(sos,reference);b=sosfiltfilt(sos,signal)*polarity
    rows=[];length=min(len(a),len(b))/rate
    for center in np.arange(.8,length-.8,.45):
        begin=int((center-.25)*rate);ref=a[begin:begin+int(.5*rate)]
        lo=max(0,begin-int(.09*rate));hi=min(len(b),begin+len(ref)+int(.09*rate));candidate=b[lo:hi]
        values=correlate(candidate,ref,mode='valid',method='fft')
        total=np.r_[0.,np.cumsum(candidate*candidate)];energy=total[len(ref):]-total[:-len(ref)]
        values/=np.sqrt(np.maximum(energy*np.dot(ref,ref),1e-30))
        k=int(np.argmax(values));fraction=0.
        if 0<k<len(values)-1:
            denom=values[k-1]-2*values[k]+values[k+1]
            if abs(denom)>1e-12:fraction=float(np.clip(.5*(values[k-1]-values[k+1])/denom,-.6,.6))
        rows.append(dict(time_s=center,lag_s=(lo+k+fraction-begin)/rate,correlation=float(values[k])))
    good=[r for r in rows if r['correlation']>.55]
    if len(good)<4:raise ValueError('重复音乐的相关性不足，无法可靠辨识非线性')
    slope,intercept,_,_=theilslopes([r['lag_s'] for r in good],[r['time_s'] for r in good])
    # Reject cycle-slip windows instead of absorbing them as microphone clock drift.
    good=[r for r in good if abs(r['lag_s']-(intercept+slope*r['time_s']))<.0006]
    if len(good)>=4:
        slope,intercept=np.polyfit([r['time_s'] for r in good],[r['lag_s'] for r in good],1,
                                   w=[r['correlation']**2 for r in good])
    if len(good)<4:raise ValueError('重复音乐的时钟对齐存在周期跳变，拒绝拟合')
    t=np.arange(len(reference))/rate
    aligned=map_coordinates(np.asarray(signal,float),[(t+intercept+slope*t)*rate],order=3,mode='constant',cval=0.)
    return aligned,dict(offset_s=float(intercept),relative_clock_drift=float(slope),windows=rows,
                       valid_windows=len(good),median_correlation=float(np.median([r['correlation'] for r in good])))


def identify_quadratic(source,recordings,source_offset,rate=16000):
    """Separate even/odd paths using +/- music at full and reduced amplitudes."""
    reference=recordings['plus'];aligned={'plus':reference};alignment={}
    for name,sign in [('minus',-1),('half_plus',1),('half_minus',-1),('repeat',1)]:
        aligned[name],alignment[name]=align_repeat(reference,recordings[name],sign,rate)
    odd=(reference-aligned['minus'])/2;even=(reference+aligned['minus'])/2
    half_odd=(aligned['half_plus']-aligned['half_minus'])/2
    half_even=(aligned['half_plus']+aligned['half_minus'])/2
    g=.6;cubic=(odd-half_odd/g)/(1-g*g);linear=odd-cubic
    n=len(source);t=np.arange(n)/rate
    # UART and ADC clocks are independent. Maximize broadband coherence, not tone peaks.
    candidates=np.linspace(.9895,.994,46);scores=[]
    for scale in candidates:
        z=np.interp(source_offset+t*scale,np.arange(len(linear))/rate,linear)
        f,c=coherence(source,z,rate,nperseg=1024,noverlap=512)
        _,power=welch(source,rate,nperseg=1024,noverlap=512)
        use=(f>1000)&(f<4500)&(power>power.max()*1e-4)
        scores.append(float(np.mean(c[use])))
    seed=float(candidates[np.argmax(scores)])
    def scale_score(scale):
        z=np.interp(source_offset+t*scale,np.arange(len(linear))/rate,linear)
        f,c=coherence(source,z,rate,nperseg=1024,noverlap=512)
        return -float(np.mean(c[use]))
    optimized=minimize_scalar(scale_score,bounds=(seed-.00012,seed+.00012),method='bounded',options={'xatol':1e-8})
    scale=float(optimized.x)
    crop=lambda a:np.interp(source_offset+t*scale,np.arange(len(a))/rate,a)
    l,e,h,c=crop(linear),crop(even),crop(half_even)/(g*g),crop(cubic)
    q=quadratic_feature(source,rate)
    kwargs=dict(fs=rate,nperseg=2048,noverlap=1536)
    f,px=welch(source,**kwargs);_,pq=welch(q,**kwargs)
    _,s1=csd(source,l,**kwargs);_,s2=csd(q,e,**kwargs)
    _,coh1=coherence(source,l,**kwargs);_,coh2=coherence(q,e,**kwargs)
    h1=s1/np.maximum(px,1e-20);h2=s2/np.maximum(pq,1e-20)
    mask=(f>=600)&(f<=4500)&(coh1>.35)&(coh2>.15)&(px>px.max()*1e-4)&(pq>pq.max()*1e-5)
    coherent_h1=abs(h1[coh1>.4])**2
    regularization=max(float(np.median(coherent_h1))*.05 if len(coherent_h1) else 1e-12,1e-15)
    raw=-h2*np.conj(h1)/(abs(h1)**2+regularization)
    weights=gaussian_filter1d(mask.astype(float),2)
    smoothed=(gaussian_filter1d(np.where(mask,raw.real,0),2)+1j*gaussian_filter1d(np.where(mask,raw.imag,0),2))/np.maximum(weights,.05)
    smoothed*=np.clip(weights,0,1);smoothed*=np.minimum(1,.8/np.maximum(abs(smoothed),1e-15))
    smoothed[(f<450)|(f>4800)]=0
    impulse=np.fft.fftshift(np.fft.irfft(smoothed,n=2048));center=1024;size=257
    kernel=impulse[center-size//2:center+size//2+1]*np.hanning(size)
    kernel-=np.mean(kernel)
    # Scale L1 into the independently enforced bound before serializing.
    kernel*=min(1,5./max(np.sum(abs(kernel)),1e-12))
    repeat=crop(aligned['repeat']);base=crop(reference)
    stable_gain=float(np.std(repeat)/max(np.std(base),1e-30))
    repeat_correlation=float(np.corrcoef(base,repeat)[0,1])
    analysis=dict(clock_scale=scale,clock_coherence_max=-float(optimized.fun),alignment=alignment,
        repeat_level_drift_dB=float(20*np.log10(max(abs(stable_gain),1e-15))),
        repeat_full_correlation=repeat_correlation,
        even_RMS=float(np.std(e)),odd_RMS=float(np.std(crop(odd))),cubic_RMS=float(np.std(c)),
        even_half_consistency=float(np.corrcoef(e,h)[0,1]),
        identified_frequency_bins=int(mask.sum()),median_even_coherence=float(np.median(coh2[mask])) if mask.any() else 0,
        kernel_L1=float(np.sum(abs(kernel))),
        scope='Combined driver/transducer/microphone/room; quadratic music model, not calibrated displacement')
    model=dict(kind='quadratic_memory_v1',kernel2=kernel.tolist(),strength=1.,trained_on='polarity-paired real music',
               sample_rate_Hz=rate,status='experimental_needs_heldout_playback')
    reliable=bool(mask.sum()>=12 and -optimized.fun>.25 and repeat_correlation>.8 and abs(analysis['repeat_level_drift_dB'])<1)
    analysis['identification_reliable']=reliable
    if not reliable:model.update(strength=0.,status='rejected_unreliable_identification')
    arrays=dict(frequency=f,H1=h1,H2=h2,coherence1=coh1,coherence2=coh2,usable=mask,
                linear=l,even=e,cubic=c,source=source,half_even=h)
    return model,analysis,arrays


def temporal_spectral_metrics(reference,capture,baseline_capture,rate=16000,clock_scale=.9912):
    """Phase-insensitive musical articulation, with the same baseline band weights.

    Mean frequency response is divided out; this evaluates how audible bands
    track the source over time. It is not THD or a calibrated perceptual score.
    """
    t=np.arange(len(reference))/rate
    warp=lambda a:np.interp(t*clock_scale,np.arange(len(a))/rate,a)
    kw=dict(fs=rate,nperseg=1024,noverlap=896,boundary=None,padded=False)
    f,_,zx=stft(reference,**kw);_,_,zy=stft(warp(capture),**kw);_,_,zb=stft(warp(baseline_capture),**kw)
    px=abs(zx)**2;py=abs(zy)**2;pb=abs(zb)**2
    mx=px.mean(axis=1);my=py.mean(axis=1);mb=pb.mean(axis=1)
    use=(f>600)&(f<4800)&(mx>mx.max()*1e-3)&(mb>mb.max()*1e-3)
    weights=np.sqrt(mx[use]*mb[use]);weights/=max(weights.sum(),1e-30)
    lx=10*np.log10(px[use]/np.maximum(mx[use,None],1e-30)+1e-4)
    ly=10*np.log10(py[use]/np.maximum(my[use,None],1e-30)+1e-4)
    candidates=[]
    for lag in range(-8,9):
        a=lx[:,max(0,lag):min(lx.shape[1],lx.shape[1]+lag)]
        b=ly[:,max(0,-lag):min(ly.shape[1],ly.shape[1]-lag)]
        ac=a-a.mean(axis=1,keepdims=True);bc=b-b.mean(axis=1,keepdims=True)
        correlations=np.sum(ac*bc,axis=1)/np.sqrt(np.maximum(np.sum(ac*ac,axis=1)*np.sum(bc*bc,axis=1),1e-30))
        offset=np.median(b-a,axis=1,keepdims=True)
        error=float(np.sum(weights*np.mean(abs(b-a-offset),axis=1)))
        candidates.append((float(np.sum(weights*correlations)),error,lag))
    best=max(candidates,key=lambda r:r[0])
    return dict(audible_band_modulation_correlation=best[0],temporal_log_spectral_error_dB=best[1],
                spectral_alignment_lag_ms=best[2]*8,audible_frequency_bins=int(use.sum()),
                scope='Time-varying music magnitude after dividing mean response; not THD')

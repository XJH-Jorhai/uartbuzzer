"""General PCM -> constrained UART audio. No tone-specific harmonic correction."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, resample_poly, firwin2, fftconvolve, welch,savgol_filter
from scipy.optimize import least_squares
from .impedance import impedance

FS=16000
KNOTS=np.array([350,500,650,800,1000,1250,1600,2000,2500,3150,4000,5000,6500.])


def read_audio(path,rate=FS):
    sr,x=wavfile.read(path)
    if x.dtype.kind in 'iu':
        info=np.iinfo(x.dtype)
        x=(x.astype(float)-(info.max+1)/2)/((info.max+1)/2) if x.dtype.kind=='u' else x.astype(float)/max(abs(info.min),info.max)
    else:x=x.astype(float)
    if x.ndim==2:x=x.mean(axis=1)
    if not np.all(np.isfinite(x)):raise ValueError('Non-finite PCM')
    ratio=Fraction(rate,int(sr));x=resample_poly(x,ratio.numerator,ratio.denominator)
    return x,rate


def music_fixture(kind='train',rate=FS,register_semitones=0):
    """Original six-second polyphonic composition, including actual silent rests.

    Continuous phase within each note, harmonic instruments, three-note chords,
    bass and broadband percussion. Holdout changes notes, key, rhythm and timbre.
    """
    rng=np.random.default_rng(9821 if kind=='train' else 5731)
    n=6*rate;x=np.zeros(n);notes=[];beat=.375
    tune=([76,79,81,79,76,74,72,74,76,79,83,81,79,76,74,72] if kind=='train' else
          [78,73,75,80,82,78,75,73,70,75,78,82,80,75,73,70])
    def add(midi,start,duration,amp,instrument):
        midi+=register_semitones
        count=min(int(duration*rate),n-int(start*rate))
        if count<=0:return
        t=np.arange(count)/rate;f=440*2**((midi-69)/12)
        env=np.minimum(t/.012,1)*np.minimum(np.maximum(duration-t,0)/.04,1)
        env*=np.exp(-t/(.28 if instrument=='pluck' else .8))
        y=np.zeros(count)
        for h in range(1,7):
            if h*f>6500:break
            strength=(1/h**1.6) if instrument=='pluck' else ((1/h) if h%2 else .12/h)
            y+=strength*np.sin(2*np.pi*h*f*t+.15*h)
        x[int(start*rate):int(start*rate)+count]+=amp*env*y
        notes.append(dict(midi=midi,start_s=start,duration_s=duration,instrument=instrument))
    for j,m in enumerate(tune):add(m,j*beat,.29 if j%4<3 else .17,.42,'pluck' if kind=='train' else 'reed')
    chords=([ [60,64,67],[57,60,64],[65,69,72],[55,59,62]] if kind=='train' else
            [[58,61,65],[63,66,70],[56,60,63],[61,65,68]])
    for bar,chord in enumerate(chords):
        for m in chord:add(m,bar*1.5,.98,.105,'reed' if kind=='train' else 'pluck')
        for j in range(4):add(chord[0]-12,bar*1.5+j*beat,.20,.18,'pluck')
    for j in range(16):
        at=int(j*beat*rate);count=int(.055*rate);t=np.arange(count)/rate
        noise=rng.normal(0,1,count)*np.exp(-t/.009)*.035
        x[at:at+count]+=noise[:min(count,n-at)]
    # Entire arrangement rests, not merely the lead voice. Fade boundaries to avoid artificial clicks.
    for a,b in [(2.80,3.0),(5.7,6.0)]:
        lo,hi=int(a*rate),int(b*rate);fade=min(int(.012*rate),lo)
        x[lo-fade:lo]*=np.linspace(1,0,fade);x[lo:hi]=0
    x/=max(abs(x).max(),1e-12)
    return x,dict(kind=kind,rate_Hz=rate,duration_s=len(x)/rate,notes=notes,score_register_semitones=register_semitones,
                  composition='original polyphonic PCM; no downloaded copyrighted source',holdout=kind!='train')


def bandlimit(x,rate=FS):
    return sosfiltfilt(butter(4,[400,5000],fs=rate,btype='bandpass',output='sos'),x)


def filter_audio(x,gains_db=None,rate=FS,loudness=False,loudness_target=.32):
    y=bandlimit(np.asarray(x,float),rate)
    if gains_db is not None:
        freq=np.r_[0,KNOTS,rate/2];g=np.r_[gains_db[0],gains_db,gains_db[-1]]
        taps=firwin2(257,freq,10**(g/20),fs=rate)
        y=fftconvolve(y,taps,mode='same')
    if loudness:
        return music_loudness(y,rate,target_rms=loudness_target)
    # Same modulation RMS for historical profile comparisons, with a common peak ceiling.
    rms=np.sqrt(np.mean(y*y));gain=min(.25/max(rms,1e-12),.95/max(abs(y).max(),1e-12))
    y*=gain
    return y,dict(normalization_gain=float(gain),RMS=float(np.sqrt(np.mean(y*y))),peak=float(abs(y).max()),
                  normalized_clipped_samples=0,band_Hz=[400,5000],pitch_shift=False)


def music_loudness(y,rate=FS,target_rms=.32):
    """Whole-song gain plus an offline lookahead limiter, never sample clipping.

    Isolated transients no longer turn down an entire track. The minimum-filter
    dilation before smoothing guarantees the smoothed gain remains below every
    sample's required ceiling. Musical dynamics below the ceiling are preserved.
    """
    from scipy.ndimage import maximum_filter1d,minimum_filter1d,uniform_filter1d
    y=np.asarray(y,float);rms=float(np.sqrt(np.mean(y*y)));ceiling=.95
    if not np.isfinite(target_rms) or not .1<=target_rms<=.7:raise ValueError('Music loudness target must be 0.1..0.7')
    gain=min(target_rms/max(rms,1e-12),16.) if rms>1e-9 else 1.
    z=y*gain
    # Stronger music settings recover faster after isolated percussion peaks.
    # The envelope remains smoothed and peak-safe at every setting.
    smoothing=max(.003,.012*(.32/target_rms)**2)
    look=max(1,round(rate*.003));smooth=max(1,round(rate*smoothing))
    peaks=maximum_filter1d(abs(z),size=2*look+1,mode='nearest')
    required=np.minimum(1.,ceiling/np.maximum(peaks,1e-15))
    envelope=uniform_filter1d(minimum_filter1d(required,size=2*smooth+1,mode='nearest'),
                             size=2*smooth+1,mode='nearest')
    result=z*envelope
    # Equal-power fades suppress file-boundary transients without changing pitch.
    fade=min(round(.012*rate),len(result)//2)
    if fade:
        ramp=np.sin(np.linspace(0,np.pi/2,fade))**2
        result[:fade]*=ramp;result[-fade:]*=ramp[::-1]
    return result,dict(conditioning='music_loudness_lookahead_v2',normalization_gain=gain,target_RMS=target_rms,
          envelope_smoothing_ms=smoothing*1000,
          RMS=float(np.sqrt(np.mean(result**2))),peak=float(max(abs(result),default=0)),
          limiter_fraction=float(np.mean(envelope<.999)),limiter_min_gain=float(min(envelope,default=1)),
          normalized_clipped_samples=0,band_Hz=[400,5000],pitch_shift=False)


def electrical_equalizer(fit_path,rlimit=55.):
    fit=json.loads(Path(fit_path).read_text(encoding='utf-8'));p=np.array(fit['selected_parameters'])
    z=impedance(KNOTS,p,fit['selected'])+rlimit+1
    # Inverse current transfer. No assertion that current is calibrated acoustic output.
    gain=20*np.log10(abs(z));gain-=np.interp(2000,KNOTS,gain)
    gain=np.clip(gain,-6,6)
    return gain,dict(load_fit_file=str(fit_path),load_fit_sha256=hashlib.sha256(Path(fit_path).read_bytes()).hexdigest(),
                     model=fit['selected'],Rlimit_assumed_from_existing_hardware_records_ohm=rlimit,
                     filter_role='bounded inverse electrical current transfer, not absolute acoustic equalization',
                     gains_db=gain.tolist(),knots_Hz=KNOTS.tolist())


def encode_pcm(y,rate=FS,baud=460800,bias=.30,amplitude=.18,silence_threshold=.0003,method='bit_feedback_uart_v1',progress=None,cancel_check=None,continuous=False,soft_edges=False,edge_model=None):
    """First-order bit-error feedback with forced start ON / stop OFF.

    Encodes arbitrary PCM, carries state across active frames, emits actual bytes.
    Real rests are represented by stopped transmission, never fake 0xff silence.
    """
    if baud not in (460800,921600):raise ValueError('Use verified candidate bauds')
    if method not in ('bit_feedback_uart_v1','frame_density_feedback','frame_aware_sigma_delta','edge_feedback_uart_v1'):raise ValueError('Unknown PCM encoder')
    if not (.1<=bias-amplitude and bias+amplitude<=.9):raise ValueError('Density budget violated')
    n=int(len(y)/rate*baud/10);centers=(np.arange(n)+.5)*10/baud
    env=np.sqrt(np.convolve(np.asarray(y)**2,np.ones(161)/161,mode='same'))
    active=np.interp(centers,np.arange(len(y))/rate,env)>silence_threshold
    if continuous:active[:]=True
    if not np.any(abs(np.asarray(y))>1e-10):active[:]=False
    # Keep short internal low-energy passages continuous; only long true rests stop the UART.
    changes=np.flatnonzero(np.diff(np.r_[True,active,True]))
    for a,b in changes.reshape(-1,2):
        if (b-a)*10/baud<.050:active[a:b]=True
    starts=np.flatnonzero(np.diff(np.r_[False,active].astype(int))==1)
    ends=np.flatnonzero(np.diff(np.r_[active,False].astype(int))==-1)+1
    segments=[];state=0.;errors=[]
    for begin,end in zip(starts,ends):
        data=bytearray(end-begin);state=0.
        source_time=np.arange(len(y))/rate
        # Bounded temporary arrays also support whole songs; error survives block boundaries.
        for block_start in range(begin,end,4096):
            if cancel_check is not None and cancel_check():raise InterruptedError('Conversion cancelled')
            block_end=min(block_start+4096,end)
            bt=(np.arange(block_start*10,block_end*10)+.5)/baud
            target=bias+amplitude*np.interp(bt,source_time,y)
            if soft_edges:
                edge=np.minimum(np.clip((bt-begin*10/baud)/.016,0,1),np.clip((end*10/baud-bt)/.016,0,1))
                target=.1+(target-.1)*edge
            if method=='edge_feedback_uart_v1':
                from .uart_codec import edge_feedback_block
                encoded,state=edge_feedback_block(target,state,**(edge_model or {}))
                data[block_start-begin:block_end-begin]=encoded
                if progress is not None:progress(block_end/max(n,1))
                continue
            if method=='frame_aware_sigma_delta':
                from .uart_codec import encode
                encoded,state,_=encode(target,'frame_aware_sigma_delta',state)
                data[block_start-begin:block_end-begin]=encoded
                continue
            if method=='frame_density_feedback':
                for frame,demand in enumerate(target.reshape(-1,10).sum(axis=1)):
                    state+=demand
                    controlled=int(np.clip(np.floor(state-.5),0,8))
                    state-=1+controlled
                    data[block_start-begin+frame]=255^((1<<controlled)-1)
                continue
            from .uart_codec import bit_feedback_block
            encoded,state=bit_feedback_block(target,state)
            data[block_start-begin:block_end-begin]=encoded
            if progress is not None:progress(block_end/max(n,1))
        errors.append(state)
        segments.append(dict(start_s=begin*10/baud,end_s=end*10/baud,data=bytes(data)))
    return segments,dict(encoder=method,baud=baud,format='8N1',bias=bias,amplitude=amplitude,
                         audio_duration_s=len(y)/rate,active_wire_s=sum(len(s['data'])*10/baud for s in segments),
                         real_idle_s=len(y)/rate-sum(len(s['data'])*10/baud for s in segments),
                         segment_final_integral_errors=errors,max_target_density=float(bias+amplitude*max(abs(y))),
                         state_reset='only at explicit silent idle segments',tone_specific_harmonic_correction=False)


def fit_acoustic_response(source,capture,rate=FS,prior_response_db=None):
    """Regularized log-magnitude system fit from an entire music passage.

    A fitted clock scale prevents spectral peaks from being mistaken for EQ.
    Microphone/room/driver/load are jointly identified; no SPL/current inference.
    """
    f,px=welch(source,rate,nperseg=2048,noverlap=1536)
    _,py=welch(capture,rate,nperseg=2048,noverlap=1536)
    use=(f>=450)&(f<=4800)&(px>px.max()*1e-4)
    lf=np.log(f[use]);xx=10*np.log10(np.maximum(px,1e-20));yy=10*np.log10(np.maximum(py[use],1e-20))
    grid=np.log(KNOTS)
    prior=np.zeros(len(KNOTS)) if prior_response_db is None else np.asarray(prior_response_db)
    # Estimate nuisance clock scale from many musical spectral peaks, after removing
    # smooth response. A smooth gain fit alone can absorb a wrong frequency scale.
    lx=10*np.log10(np.maximum(px,1e-20));ly=10*np.log10(np.maximum(py,1e-20))
    wx=np.clip(lx-savgol_filter(lx,31,3),-15,15)
    wy=np.clip(ly-savgol_filter(ly,31,3),-15,15)
    cmask=(f>=600)&(f<=4800)
    scales=np.linspace(.975,1.025,1001)
    scores=[float(np.corrcoef(np.interp(f[cmask]*s,f,wx),wy[cmask])[0,1]) for s in scales]
    clock_seed=float(scales[int(np.argmax(scores))])
    def calc(v):
        scale=v[0];g=v[1:]
        ref=np.interp(f[use]*scale,f,xx)
        model=ref+np.interp(lf,grid,g)
        return np.r_[(model-yy)/2, np.diff(g,2)*.7, (g-g.mean()-prior)*.06]
    initial=np.r_[1.,np.full(len(KNOTS),np.median(yy-np.interp(f[use],f,xx)))];solutions=[]
    for scale in (clock_seed-.00015,clock_seed,clock_seed+.00015):
        v=initial.copy();v[0]=scale
        lower=max(.974,clock_seed-.0003);upper=min(1.026,clock_seed+.0003)
        s=least_squares(calc,v,bounds=(np.r_[lower,np.full(len(KNOTS),-110)],np.r_[upper,np.full(len(KNOTS),20)]),
                        loss='soft_l1',f_scale=1.5,max_nfev=300)
        solutions.append(s)
    best=min(solutions,key=lambda v:v.cost);gain=best.x[1:].copy();gain-=np.interp(2000,KNOTS,gain)
    return dict(knots_Hz=KNOTS.tolist(),relative_chain_response_db=gain.tolist(),
                inverse_correction_db=np.clip(-gain,-6,6).tolist(),clock_scale=float(best.x[0]),
                fit_log_spectral_residual_dB=float(np.median(abs(calc(best.x)[:use.sum()])*2)),
                music_peak_clock_scale_seed=clock_seed,music_peak_correlation=float(max(scores)),
                excited_frequency_bins=int(use.sum()),cost=float(best.cost),
                scope='music-excited magnitude of combined microphone/room/electrical chain; uncalibrated',
                absolute_SPL_dB='unknown')


def music_metrics(reference,capture,rate=FS,clock_scale=1.):
    # Frequency-domain comparison uses the fitted frequency-scale consistently.
    f,px=welch(reference,rate,nperseg=2048,noverlap=1536)
    _,py=welch(capture,rate,nperseg=2048,noverlap=1536)
    target=np.interp(f*clock_scale,f,px)
    bands=np.geomspace(450,4800,17);rx=[];ry=[]
    for lo,hi in zip(bands[:-1],bands[1:]):
        m=(f>=lo)&(f<hi);rx.append(np.sum(target[m]));ry.append(np.sum(py[m]))
    rx=np.array(rx);ry=np.array(ry);valid=rx>max(rx.max()*1e-3,1e-20)
    level=10*np.log10(np.maximum(ry[valid],1e-20)/np.maximum(rx[valid],1e-20));level-=np.median(level)
    # Scale-independent spectral shape metrics; never optimize microphone RMS alone.
    a=rx/np.linalg.norm(rx);b=ry/max(np.linalg.norm(ry),1e-20)
    # Whole-passage temporal envelope, allowing capture latency and fitted clock drift.
    warped=np.interp(np.arange(len(reference))*clock_scale,np.arange(len(capture)),capture)
    smooth=np.ones(int(.02*rate))/int(.02*rate)
    er=np.sqrt(np.maximum(np.convolve(reference**2,smooth,'same'),0))[::int(.01*rate)]
    ec=np.sqrt(np.maximum(np.convolve(warped**2,smooth,'same'),0))[::int(.01*rate)]
    correlations=[]
    for lag in range(-20,21):
        aa=er[max(0,lag):min(len(er),len(er)+lag)]
        bb=ec[max(0,-lag):min(len(ec),len(ec)-lag)]
        corr=float(np.corrcoef(aa,bb)[0,1]) if np.std(aa)>1e-12 and np.std(bb)>1e-12 else 0.
        correlations.append(corr)
    best=int(np.argmax(correlations))
    return dict(log_band_error_RMS_dB=float(np.sqrt(np.mean(level*level))),
                log_band_error_median_abs_dB=float(np.median(abs(level))),
                spectral_cosine_similarity=float(np.dot(a,b)),
                captured_RMS_dBFS=float(20*np.log10(max(np.sqrt(np.mean(capture**2)),1e-15))),
                captured_peak_dBFS=float(20*np.log10(max(abs(capture).max(),1e-15))),
                clipped_fraction=float(np.mean(abs(capture)>=.999)),
                envelope_correlation=float(correlations[best]),envelope_alignment_lag_ms=(best-20)*10,
                spectrum_frequency_scale=clock_scale,absolute_SPL_dB='unknown',
                valid_bands=int(valid.sum()),target_band_Hz=[450,4800],
                evaluation='full-passage polyphonic music; band-shape fidelity, not single-tone THD')

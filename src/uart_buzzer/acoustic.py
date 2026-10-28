"""Relative acoustic metrics from raw microphone samples, never absolute SPL."""
import numpy as np
from scipy.signal import butter,sosfiltfilt
from scipy.optimize import minimize_scalar


def db(x):return float(20*np.log10(max(float(x),1e-15)))


def propose_input_gain_db(current,peak_db,clipped,minimum,maximum,step):
    """Decrease-only gain staging; quantize downward to the endpoint step."""
    if clipped:target=current-12
    elif peak_db>-6:target=current+(-12-peak_db)
    else:return float(current)
    target=np.floor((target-minimum)/step)*step+minimum
    return float(np.clip(target,minimum,maximum))


def reference_gate(first,last,controls_unchanged=True,audio_status=''):
    rms=last['total_window_RMS_dBFS']-first['total_window_RMS_dBFS']
    fundamental=last['fundamental_peak_dBFS']-first['fundamental_peak_dBFS']
    second=last['second_harmonic_peak_dBFS']-first['second_harmonic_peak_dBFS']
    return dict(reference_total_RMS_drift_dB=rms,reference_fundamental_drift_dB=fundamental,
                reference_second_harmonic_drift_dB=second,
                stable_reference_gate=bool(max(abs(rms),abs(fundamental),abs(second))<=1
                    and controls_unchanged and not audio_status))


def measure(data,rate,trial,channel=0):
    x=np.asarray(data[:,channel] if data.ndim==2 else data,float)
    f=float(trial['nominal_Hz']);on=trial['write_start_record_s']
    band=sosfiltfilt(butter(4,[f-120,f+120],btype='bandpass',fs=rate,output='sos'),x)
    n=int(.06*rate);cs=np.r_[0,np.cumsum(band**2)];energy=(cs[n:]-cs[:-n])/n
    lo=int(max(0,on-.06)*rate);hi=min(len(energy)-1,int((on+.3)*rate))
    k=lo+int(np.argmax(energy[lo:hi+1]));raw=x[k:k+n].copy();y=raw-raw.mean()
    baseline=band[int(max(0,on-.7)*rate):int(max(.01,on-.15)*rate)]
    snr=db(np.sqrt(max(energy[k],0)))-db(np.sqrt(np.mean(baseline**2)))
    win=np.hanning(n);ff=np.fft.rfftfreq(65536,1/rate);spec=abs(np.fft.rfft(y*win,65536))*2/win.sum()
    peaks=[]
    for h in range(1,6):
        mask=(ff>f*h*.975)&(ff<f*h*1.025)
        if np.any(mask):
            j=np.flatnonzero(mask)[np.argmax(spec[mask])];peaks.append((spec[j],ff[j]/h))
    # A strong harmonic gives a better f0 estimate than a weak fundamental.
    _,seed=max(peaks)
    t=np.arange(n)/rate
    def fit(freq):
        X=np.column_stack([np.ones(n)]+[fn(2*np.pi*freq*h*t) for h in range(1,6) for fn in (np.sin,np.cos)])
        coef=np.linalg.lstsq(X,y,rcond=None)[0]
        return coef,y-X@coef
    def score(freq):
        _,e=fit(freq);return np.mean(e*e)
    freq=float(minimize_scalar(score,bounds=(seed-5,seed+5),method='bounded').x)
    coef,residual=fit(freq);amps=np.hypot(coef[1::2],coef[2::2]);fund=amps[0]
    return dict(label=trial['label'],nominal_Hz=f,estimated_fundamental_Hz=freq,channel=channel,
        fundamental_peak_FS=float(fund),fundamental_peak_dBFS=db(fund),
        second_harmonic_peak_dBFS=db(amps[1]),THD_2to5=float(np.linalg.norm(amps[1:])/max(fund,1e-15)),
        total_window_RMS_dBFS=db(np.sqrt(np.mean(y*y))),
        residual_RMS_dBFS=db(np.sqrt(np.mean(residual**2))),relative_narrowband_SNR_dB=snr,
        clipped_fraction=float(np.mean(abs(raw)>=.999)),window_start_s=k/rate,window_duration_s=n/rate,
        reliable_candidate=bool(snr>=15 and not np.any(abs(raw)>=.999)),
        absolute_SPL_dB='unknown',scope='microphone+room response; relative acoustic measurement')

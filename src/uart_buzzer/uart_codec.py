"""Actual UART bytes. Error state persists across frames, resets only on explicit silence."""
import time
import numpy as np

CANDIDATES = np.array([[0]+[(v>>b)&1 for b in range(8)]+[1] for v in range(256)],dtype=np.uint8)
ON = 1-CANDIDATES


def edge_feedback_block(target,state=0.,on_loss=.01838462745,off_tail=.10345002714,bias_offset=.034):
    """8N1 feedback using bit areas from a provisional circuit edge model.

    Rising commands lose on_loss of a bit; falling commands retain off_tail.
    The offset preserves the existing quiescent operating point. Coefficients
    require circuit simulation and music validation, not a tone detector.
    """
    values=np.asarray(target,float).ravel().tolist()
    if len(values)%10:raise ValueError('Need complete frames')
    data=bytearray(len(values)//10);error=float(state)
    for frame in range(len(data)):
        previous=False;value=0;base=frame*10
        for bit in range(10):
            wanted=error+values[base+bit]+bias_offset
            on_area=1. if previous else 1.-on_loss
            off_area=off_tail if previous else 0.
            active=True if bit==0 else False if bit==9 else wanted>=(on_area+off_area)*.5
            error=wanted-(on_area if active else off_area)
            if 0<bit<9:value|=(not active)<<(bit-1)
            previous=active
        data[frame]=value
    return bytes(data),error


def bit_feedback_block(target,state=0.):
    """Vectorized first-order UART bit feedback, with an exact saturation fallback.

    Forced boundary bits are accounted for in cumulative integer ON counts.
    This keeps the streaming encoder fast enough to refill Windows serial queues.
    """
    t=np.asarray(target,float).reshape(-1,10)
    if not len(t):return b'',float(state)
    cumulative=float(state)+np.cumsum(t.sum(axis=1))
    quantized=np.floor(cumulative-t[:,-1]+.5)
    previous=np.r_[0.,quantized[:-1]]
    wanted=quantized-previous-1
    start_error=np.r_[float(state),cumulative[:-1]-quantized[:-1]]+t[:,0]-1
    q=np.maximum(0,np.floor(start_error[:,None]+np.cumsum(t[:,1:9],axis=1)+.5))
    on=np.diff(np.c_[np.zeros(len(t)),q],axis=1)
    if np.all((wanted>=0)&(wanted<=8)) and np.all(q[:,-1]==wanted) and np.all((on==0)|(on==1)):
        values=np.sum((1-on).astype(np.uint8)*(1<<np.arange(8)),axis=1).astype(np.uint8)
        return values.tobytes(),float(cumulative[-1]-quantized[-1])
    data=bytearray(len(t));error=float(state)
    for j,frame in enumerate(t):
        error+=frame[0]-1;value=0
        for bit in range(8):
            error+=frame[bit+1];active=error>=.5;error-=active;value|=(not active)<<bit
        error+=frame[9];data[j]=value
    return bytes(data),float(error)


def frame_bits(data):
    return CANDIDATES[np.frombuffer(bytes(data),dtype=np.uint8)].ravel()


def encode(target_on, method, state=0.):
    """target_on sampled at each physical bit center, in [0.1,0.9].

    Sigma delta uses greedy 256-candidate within-frame integral squared error,
    not arbitrary bitstream. Complexity O(256*10*frames); no global optimality claim.
    """
    target=np.asarray(target_on,float)
    if len(target)%10:
        raise ValueError('Need complete frames')
    t0=time.perf_counter(); data=[];error=float(state)
    for frame in target.reshape(-1,10):
        if method=='naive_byte_density':
            n=int(np.clip(round(10*frame.mean()-1),0,8))
            byte=255 ^ ((1<<n)-1)
        elif method in ('frame_aware_pdm','frame_aware_sigma_delta'):
            carry=error if method=='frame_aware_sigma_delta' else 0.
            local=carry+np.cumsum(frame[None,:]-ON,axis=1)
            score=np.mean(local**2,axis=1)+2*local[:,-1]**2
            byte=int(np.argmin(score))
        else:
            raise ValueError(method)
        error += float(np.sum(frame-ON[byte]))
        data.append(byte)
    return bytes(data),error,dict(seconds=time.perf_counter()-t0,frames=len(data),
                                 candidates_per_frame=256 if method!='naive_byte_density' else 1)


def tone_target(f,baud,duration=.02,bias=.5,amplitude=.35):
    n=int(duration*baud/10)*10
    t=(np.arange(n)+.5)/baud
    return np.clip(bias+amplitude*np.sin(2*np.pi*f*t),.1,.9)


def events(data,baud,vio=5.,start=.0005,gaps=None):
    """PWL corners with finite 10ns edges; gap map is {frame_index: idle_seconds}."""
    times=[0.,start]; volts=[vio,vio]; now=start; old=1
    gaps=gaps or {}
    for j,value in enumerate(data):
        if j in gaps:
            if old!=1:
                times.extend((now,now+1e-8));volts.extend((0.,vio));old=1
            now += gaps[j]
        for bit in CANDIDATES[value]:
            bit=int(bit)
            if bit!=old:
                times.extend((now,now+1e-8));volts.extend((old*vio,bit*vio));old=bit
            now += 1/baud
    if old!=1:
        times.extend((now,now+1e-8));volts.extend((old*vio,vio))
    times.append(now+.0005);volts.append(vio)
    wave=np.c_[times,volts]
    # Remove redundant identical corners (initial idle and first edge share a time).
    keep=np.r_[True,np.diff(wave[:,0])>0]
    if np.any((~keep)[1:] & (np.diff(wave[:,1])!=0)):
        raise ValueError('Different voltages at identical PWL time')
    return wave[keep],now+.0005


def signal_suite(baud):
    """Short cue, original/transposed tune, note transitions, rest and sweep."""
    segments=[('cue',2000,.004),('original_A4',440,.004),('transpose_A6',1760,.004),
              ('B6',1975.53,.004),('C7',2093,.004),('rest',0,.003),('sweep',-1,.006)]
    data=bytearray(); schedule=[]; t=0
    for name,f,duration in segments:
        if f==0:
            schedule.append(dict(name=name,start_s=t,duration_s=duration,kind='idle_gap',frame_index=len(data)))
            t+=duration;continue
        n=int(duration*baud/10)*10
        local=(np.arange(n)+.5)/baud
        phase=2*np.pi*((1900*local+(.5*(2400-1900)/duration)*local**2) if f<0 else f*local)
        target=.5+.3*np.sin(phase)
        b,_,_=encode(target,'frame_aware_sigma_delta')
        schedule.append(dict(name=name,start_s=t,duration_s=n/baud,kind='bytes',frame_index=len(data)))
        data.extend(b);t+=n/baud
    return bytes(data),schedule

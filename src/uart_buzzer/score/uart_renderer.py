"""8N1 adapter. Score rests stop transmission; they are not quiet bytes."""
import hashlib
import numpy as np
from scipy.signal import butter,sosfilt
from .contracts import score_ir
from .model import render_preview,digest
from ..uart_codec import bit_feedback_block

VERSION='finite-score-uart8n1-v1'


def transport_signature(baud,volume,bias,amplitude):
    return dict(backend='uart8n1',renderer=VERSION,baud=int(baud),volume=float(volume),bias=float(bias),amplitude=float(amplitude),
                envelope='sine_0.65_rise6ms_fall12ms_16k_resampled',codec='bit_feedback_uart_v1')


def render_tone(hz,duration,baud,volume,bias,amplitude,cancel_check=None):
    from .model import SCORE_SCHEMA,hz_to_midi
    midi=float(hz_to_midi(hz));score=dict(schema=SCORE_SCHEMA,duration_s=duration,time_quantum_s=.02,voices=1,
        palette=[dict(id='tone',frequency_hz=float(hz),midi_pitch=midi)],velocity_levels=[1.],rests=[],
        events=[dict(start_s=0.,duration_s=duration,note_id='tone',frequency_hz=float(hz),midi_pitch=midi,velocity=1.,velocity_index=0,voice=0)])
    return render(score,baud,volume,bias,amplitude,cancel_check)


def render(score,baud=460800,volume=.8,bias=.3,amplitude=.18,cancel_check=None,progress=None):
    score_ir(score)
    if baud not in (460800,921600) or not np.isfinite(volume) or not 0<=volume<=1:raise ValueError('UART 参数无效')
    if not .1<=bias-amplitude<=bias+amplitude<=.9:raise ValueError('UART 密度范围无效')
    if any(not 25<=n['frequency_hz']<=6500 for n in score['palette']):raise ValueError('UART 音高范围应为25–6500Hz；其他播放器请导出通用曲谱')
    pcm=render_preview(score)*volume;segments=[];rate=16000;times=np.arange(len(pcm))/rate
    for index,event in enumerate(score['events'] if volume>0 else []):
        if cancel_check and cancel_check():raise InterruptedError('曲谱渲染已取消')
        a=round(event['start_s']*baud/10);b=round((event['start_s']+event['duration_s'])*baud/10)
        parts=[];state=0.
        for begin in range(a,b,4096):
            if cancel_check and cancel_check():raise InterruptedError('曲谱渲染已取消')
            end=min(b,begin+4096);bt=(np.arange(begin*10,end*10)+.5)/baud
            target=bias+amplitude*np.interp(bt,times,pcm)
            data,state=bit_feedback_block(target,state);parts.append(data)
        if parts:segments.append(dict(start_s=a*10/baud,end_s=b*10/baud,data=b''.join(parts)))
        if progress:progress((index+1)/max(1,len(score['events'])))
    decoded=np.zeros(len(pcm),np.float32)
    sos=butter(4,6500,btype='lowpass',fs=baud/10,output='sos')
    for segment in segments:
        data=np.frombuffer(segment['data'],np.uint8)
        # Frame-density reconstruction; no old 400 Hz high-pass on this path.
        density=np.array([9-int(v).bit_count() for v in data],float)/10
        y=sosfilt(sos,(density-bias)/amplitude);a=round(segment['start_s']*rate);b=min(len(pcm),round(segment['end_s']*rate))
        if b>a:decoded[a:b]=np.interp(np.arange(a,b)/rate,segment['start_s']+(np.arange(len(y))+.5)*10/baud,y)
    plan=dict(schema='finite-score-transport-plan-v1',backend='uart8n1',renderer=VERSION,baud=baud,format='8N1',volume=volume,duration_s=score['duration_s'],
              rest_strategy='stop_writes_idle_high_no_bytes',note_state='reset_feedback_at_each_note',
              quantization='round_score_boundaries_to_UART_frames',
              execution='bounded OS transmit queue; drain at rests then monotonic wait; physical USB timing is estimated',
              segments=[dict(start_s=s['start_s'],end_s=s['end_s'],bytes=len(s['data']),sha256=hashlib.sha256(s['data']).hexdigest()) for s in segments],
              score_hash=digest(score),absolute_SPL='unknown')
    return pcm,decoded,segments,plan

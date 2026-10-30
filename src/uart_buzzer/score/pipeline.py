"""Raw resampled PCM -> frontend melody -> finite score -> transport adapter."""
from dataclasses import asdict
import hashlib,json,sys
from pathlib import Path
import numpy as np
from . import frontend_v2
from .frontend import decode_melody
from .model import FitSettings,fit_score,digest,render_preview
from .contracts import device_profile,flat_profile,score_ir
from . import uart_renderer

VERSION='raw-pcm-finite-score-v1'
FRONTEND_VERSION='harmonic-salience-v2-product-v1'


def settings(config):
    config=dict(config or {});p=device_profile(config.get('device_profile') or flat_profile())
    fit=FitSettings(**config.get('fit',{}))
    if type(fit.palette_size) is not int or not 1<=fit.palette_size<=32 or fit.requested_voices!=1:raise ValueError('预览版最多32个字典音高、单声部')
    if any(not np.isfinite(float(v)) for v in asdict(fit).values()):raise ValueError('拟合设置必须有限')
    if not .005<=fit.time_quantum_s<=.2 or fit.pitch_sigma_semitones<=0 or fit.max_pitch_deviation_semitones<=0:raise ValueError('拟合时间格/音高界无效')
    if type(fit.automatic_palette) is not bool or type(fit.allow_octave_adaptation) is not bool:raise ValueError('拟合开关必须为布尔值')
    if not 0<fit.palette_target_coverage<=1 or not 0<=fit.palette_histogram_floor<1 or type(fit.palette_swap_passes) is not int or not 0<=fit.palette_swap_passes<=8:raise ValueError('字典筛选设置无效')
    return dict(device_profile=p,fit=asdict(fit),response_enabled=bool(config.get('response_enabled',True)))


def fingerprint():
    folder=Path(__file__).parent;files=sorted(folder.glob('*.py'))
    if files:return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assets=Path(getattr(sys,'_MEIPASS',folder.parents[2]))/'desktop_assets'
    return json.loads((assets/'score_fingerprint.json').read_text())


def convert(raw,config,baud,volume,bias,amplitude,progress=None,cancel_check=None):
    def check():
        if cancel_check and cancel_check():raise InterruptedError('离散曲谱转换已取消')
    def emit(x):
        check()
        if progress:progress(x)
    check();cfg=settings(config);x=np.asarray(raw,np.float64)
    if x.ndim!=1 or not 1600<=len(x)<=16000*600 or not np.isfinite(x).all():raise ValueError('需要0.1–600秒有限单声道16k PCM')
    reference_transport=cfg['device_profile'].get('provenance',{}).get('reference_transport')
    transport_mismatch=(cfg['device_profile']['response_kind']=='relative_acoustic' and
        reference_transport!=uart_renderer.transport_signature(baud,volume,bias,amplitude))
    if transport_mismatch:cfg['response_enabled']=False
    candidates=frontend_v2.extract_candidates(x,16000,cancel_check=cancel_check,progress=lambda v:emit(.05+.25*v))
    reference,_=decode_melody(candidates,cancel_check=cancel_check,progress=lambda v:emit(.3+.15*v))
    score=fit_score(candidates,cfg['device_profile'],FitSettings(**cfg['fit']),cfg['response_enabled'],reference_events=reference,
                    cancel_check=cancel_check,progress=lambda v:emit(.45+.4*v))
    score['source_provenance'].update(source_PCM_sha256=candidates['source_PCM_sha256'],frontend=FRONTEND_VERSION,pipeline=VERSION)
    onset=np.asarray(candidates['onset_strength']);peaks=np.flatnonzero((onset>.65)&(onset>=np.r_[0,onset[:-1]])&(onset>np.r_[onset[1:],0]))
    score['source_provenance'].update(frontend_hop_s=candidates['hop_s'],onset_markers_s=[round(float(candidates['frame_times_s'][i]),6) for i in peaks],
        analysis_midi_min=48,analysis_midi_max=96,
        timing_note='10ms frontend hop; score quantum and device dwell apply; boundaries are approximate')
    source_duration=sum(e['duration_s'] for e in reference);represented=sum(e['duration_s'] for e in score['events'])
    # Temporal representation is evaluated against the frontend's selected line,
    # never against unknown musical ground truth.
    correct=0.;reference_cursor=0;deviations=[]
    for event in score['events']:
        while reference_cursor<len(reference) and reference[reference_cursor]['onset_s']+reference[reference_cursor]['duration_s']<=event['start_s']:reference_cursor+=1
        for ref in reference[reference_cursor:]:
            if ref['onset_s']>=event['start_s']+event['duration_s']:break
            overlap=max(0,min(event['start_s']+event['duration_s'],ref['onset_s']+ref['duration_s'])-max(event['start_s'],ref['onset_s']))
            if abs(event['midi_pitch']-score['global_semitone_shift']-ref['midi'])<=cfg['fit']['max_pitch_deviation_semitones']+1e-9:correct+=overlap
            if overlap:deviations.append((overlap,event['midi_pitch']-ref['midi']))
    score['diagnostics']=dict(reference_events=len(reference),events=len(score['events']),unique_event_pitches=len({e['note_id'] for e in score['events']}),
        reference_voiced_s=source_duration,represented_s=represented,melody_temporal_coverage=min(1.,correct/source_duration) if source_duration else 1.,
        unrepresented_reference_s=max(0.,source_duration-correct),rest_s=sum(r['duration_s'] for r in score['rests']),
        duration_weighted_absolute_pitch_error_semitones=sum(w*abs(d) for w,d in deviations)/max(1e-12,sum(w for w,d in deviations)),
        max_absolute_pitch_deviation_semitones=max((abs(d) for w,d in deviations),default=0.),
        response_disabled_for_transport_mismatch=transport_mismatch,
        response_disabled_reason='reference_transport_missing_or_changed' if transport_mismatch else None,
        limitations=['仅主旋律；混音可能跟随伴奏','缺失基音可能高八度；置信分不是概率','连奏/连续重复同音可能合并；起止为网格近似','覆盖率相对前端所选旋律，非音乐GT准确率'])
    unresolved=[];cursor=0
    for ref in reference:
        a=ref['onset_s'];end=a+ref['duration_s']
        while cursor<len(score['events']) and score['events'][cursor]['start_s']+score['events'][cursor]['duration_s']<=a:cursor+=1
        for e in score['events'][cursor:]:
            if e['start_s']>=end:break
            if e['start_s']>a:unresolved.append(dict(start_s=a,duration_s=min(end,e['start_s'])-a,reference_midi=ref['midi'],reason='range_or_quantization_or_device_constraint'))
            a=max(a,e['start_s']+e['duration_s'])
        if a<end:unresolved.append(dict(start_s=a,duration_s=end-a,reference_midi=ref['midi'],reason='range_or_quantization_or_device_constraint'))
    score['unresolved']=unresolved
    score_ir(score);check()
    if cfg['device_profile'].get('transport','uart8n1')=='generic':
        pcm=render_preview(score)*volume;decoded=np.zeros(len(pcm),np.float32);segments=[]
        plan=dict(schema='finite-score-transport-plan-v1',backend='generic_score_only',score_hash=digest(score),executable=False)
    else:pcm,decoded,segments,plan=uart_renderer.render(score,baud,volume,bias,amplitude,cancel_check,lambda v:emit(.85+.14*v))
    emit(1.)
    return dict(score=score,transport_plan=plan,processed=np.asarray(pcm,np.float32),decoded=np.asarray(decoded,np.float32),segments=segments)

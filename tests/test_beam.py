"""Original Beam integration tests. Offline; no actual serial/audio devices."""
import sys,json,hashlib,tempfile,threading
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uart_buzzer.beam import settings_for,search_block,encode_segments
from uart_buzzer.beam_model import load_model,finite_gram
from uart_buzzer._beam_kernels import trajectory,reference,COMPILED
from uart_buzzer.desktop.engine import convert,LiveEncoder,PrecomputedEncoder,export_document,beam_ab_preview
from uart_buzzer.desktop.cache import ConversionCache
from uart_buzzer.music import FS,music_fixture,encode_pcm
from uart_buzzer.uart_codec import ON,frame_bits
FIT=ROOT/'desktop_assets/impedance_fit.json'
PROFILE=json.loads((ROOT/'desktop_assets/music_profile.json').read_text(encoding='utf-8'))
GOLDEN={
    460800:dict(convert='74132f332f16722305a93358846aa1cafe4fc6d07505fbe9ab530c0f8fa64bec',
        live08='12eb10295a262d2ed582722399583a0f0eae22a62ff3b1b2a9791b5a3d4dbdca',
        live10='b4416b13ae261e3e8dd6e7b28eaba36923014172aae93c71a45f5b1bf6bcd962'),
    921600:dict(convert='23be29aaa0ced611e966a54e3c7b1b51e91d71bcc31b25c123db185cb0f4b1f8',
        live08='97d911073e79981ad1fdc59e7143b9643ba0931b39c0c15f1fffc57ed9176503',
        live10='c5b40cad14036d12d9e1daf96b57771d164dde347418130780cf7ee2253b24b4')}

@pytest.mark.parametrize('baud',[460800,921600])
def test_frozen_actual_legacy_conversion_and_live_bytes_unchanged(baud):
    x=music_fixture('train')[0];p=dict(PROFILE,baud=baud)
    doc=convert('golden',p,source=x)
    assert hashlib.sha256(b''.join(s['data'] for s in doc.segments)).hexdigest()==GOLDEN[baud]['convert']
    for volume in (.8,1.):
        enc=LiveEncoder(p);n=int(len(x)/FS*baud/10);data=[]
        for k in range(0,n,2048):data.append(enc.block(doc.processed,k,min(2048,n-k),volume)[0])
        assert hashlib.sha256(b''.join(data)).hexdigest()==GOLDEN[baud]['live08' if volume==.8 else 'live10']

def beam_document(seconds=.20,**overrides):
    p=dict(PROFILE,encoder='beam_fast',beam_volume=.8,beam_settings={'budget_ratio':300,**overrides})
    return convert('fixture',p,source=music_fixture('holdout')[0][:round(FS*seconds)],fit_path=FIT)

def test_production_whole_document_objective_and_fixed_volume_export(tmp_path):
    d=beam_document();b=d.metadata['beam'];assert b['returned_SSE']<=b['baseline_SSE']+1e-12
    assert b['optimized_frames']==int(d.duration*d.profile['baud']/10)
    enc=PrecomputedEncoder(d);parts=[]
    for seg in d.segments:
        start=round(seg['start_s']*d.profile['baud']/10)
        for k in range(0,len(seg['data']),321):parts.append(enc.block(d.processed,start+k,min(321,len(seg['data'])-k),.8)[0])
    data=b''.join(parts);assert data==b''.join(s['data'] for s in d.segments)
    assert enc.block(d.processed,0,10,0)[0]==b''
    with pytest.raises(ValueError):enc.block(d.processed,0,10,.5)
    path=export_document(d,tmp_path,.8);manifest=json.loads(path.read_text(encoding='utf-8'))
    assert b''.join((tmp_path/r['file']).read_bytes() for r in manifest['segments'])==data
    with pytest.raises(ValueError):export_document(d,tmp_path/'wrong',1.)
    with pytest.raises(ValueError):LiveEncoder(d.profile)

def test_budget_fallback_returns_entire_retained_actual_baseline():
    d=beam_document(budget_ratio=1e-7);assert d.metadata['beam']['fallback_reason']=='budget'
    assert [s['data'] for s in d.segments]==[s['data'] for s in d.baseline_segments]
    assert d.metadata['beam']['returned_SSE']==d.metadata['beam']['baseline_SSE']

def test_explicit_idle_preserves_real_plant_state_and_exact_global_evaluation():
    x=np.r_[np.sin(np.arange(3200)*.71)*.4,np.zeros(3200),np.sin(np.arange(3200)*.42)*.4]
    p=dict(PROFILE,encoder='beam_fast',continuous_carrier=False,soft_carrier_edges=True,beam_volume=1.,beam_settings={'budget_ratio':300})
    d=convert('rests',p,source=x,fit_path=FIT);assert len(d.segments)>=2
    m=load_model(FIT,p['baud']);n=int(len(x)/FS*p['baud']/10);command=np.zeros(n*10);data=np.full(n,256,np.uint16)
    from uart_buzzer.desktop.engine import LiveTargetPlan
    plan=LiveTargetPlan(d.beam_pcm,d.quadratic_correction,d.profile,d.metadata['beam']['live_target_blocks'])
    for seg in d.segments:
        a=round(seg['start_s']*p['baud']/10);b=a+len(seg['data']);data[a:b]=np.frombuffer(seg['data'],np.uint8)
        command[a*10:b*10]=plan.command(a,b-a)
    g=np.vstack((m.g,np.zeros((1,6))));q=np.vstack((m.q,np.zeros((1,10))))
    y,_=trajectory(data,m.F,g,m.H,q,np.zeros(6));target,_=reference(command,m.bit_F,m.bit_g,m.half_F,m.half_g,np.zeros(6))
    assert abs(np.sum((y-target)**2)-d.metadata['beam']['returned_SSE'])<1e-8

def test_cancel_and_cache_keys_model_budget_volume_baseline(tmp_path):
    x=music_fixture('train')[0][:3200];p=dict(PROFILE,encoder='beam_fast',beam_settings={'budget_ratio':300})
    cache=ConversionCache(tmp_path/'cache');d=cache.convert('fixture',p,source=x,fit_path=FIT)
    restarted=ConversionCache(tmp_path/'cache').convert('fixture',p,source=x,fit_path=FIT)
    assert restarted.metadata['cache']['tier']=='disk'
    np.testing.assert_array_equal(restarted.beam_pcm,d.beam_pcm)
    assert [s['data'] for s in restarted.baseline_segments]==[s['data'] for s in d.baseline_segments]
    key=cache.key('fixture',d.profile,'music',FIT,x,None)
    for change in ({'beam_volume':.7},{'beam_settings':{'budget_ratio':5}},{'beam_baseline_encoder':'frame_aware_sigma_delta'}):
        assert cache.key('fixture',dict(d.profile,**change),'music',FIT,x,None)!=key
    other=tmp_path/'model.json';other.write_bytes(FIT.read_bytes()+b' ')
    assert cache.key('fixture',d.profile,'music',other,x,None)!=key
    empty=ConversionCache(tmp_path/'cancel')
    with pytest.raises(InterruptedError):empty.convert('fixture',p,source=x,fit_path=FIT,cancel_check=lambda:True)
    assert not empty.folder.exists()

def test_preview_uses_retained_target_baseline_and_same_volume():
    d=beam_document();v=beam_ab_preview(d,FIT)
    assert np.all(np.isfinite(v['current_mA'])) and len(v['frequency_Hz'])>0
    from uart_buzzer.desktop.engine import LiveTargetPlan
    plan=LiveTargetPlan(d.beam_pcm,d.quadratic_correction,d.profile,d.metadata['beam']['live_target_blocks'])
    assert v['diagnostics']['prep_hash']==plan.digest()

def test_precomputed_playback_adapter_sends_exact_completed_bytes(tmp_path):
    from uart_buzzer.desktop.devices import Playback
    d=beam_document(seconds=.10);written=[]
    class Port:
        baudrate=460800;out_waiting=0;is_open=True;break_condition=False
        def write(self,data):written.append(data);return len(data)
        def reset_output_buffer(self):pass
    class Link:
        connected=True;port=Port()
        def interrupt(self):pass
    class Mic:
        stream=None;recording=False;lock=threading.Lock()
    with patch('uart_buzzer.desktop.devices.LiveEncoder',side_effect=AssertionError('Must not re-encode Beam during playback')):
        result=Playback(Link(),d,Mic(),tmp_path,volume=.8).run()
    assert result['completed'] and b''.join(written)==b''.join(s['data'] for s in d.segments)

@pytest.mark.parametrize('baud',[460800,921600])
def test_beam_retained_incumbent_is_actual_frozen_playback_stream(baud):
    x=music_fixture('train')[0]
    p=dict(PROFILE,baud=baud,encoder='beam_fast',beam_volume=.8,beam_settings={'budget_ratio':1e-7})
    doc=convert('golden',p,source=x,fit_path=FIT)
    assert hashlib.sha256(b''.join(s['data'] for s in doc.baseline_segments)).hexdigest()==GOLDEN[baud]['live08']

def test_budget_fallback_does_not_poison_future_optimization_cache(tmp_path):
    cache=ConversionCache(tmp_path/'cache')
    p=dict(PROFILE,encoder='beam_fast',beam_settings={'budget_ratio':1e-7})
    x=music_fixture('train')[0][:1600]
    doc=cache.convert('budget',p,source=x,fit_path=FIT)
    assert doc.metadata['beam']['budget_hit']
    assert doc.metadata['cache']['skipped_reason']=='budget_fallback_is_retryable'
    assert not cache.memory and not list(cache.folder.glob('*.npz'))

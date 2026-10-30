"""Original synthetic product fixtures; no real songs or hardware."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from uart_buzzer.score import model
from uart_buzzer.score.pipeline import convert,settings
from uart_buzzer.score.contracts import flat_profile,device_profile,score_ir,load_json,ProfileStore
from uart_buzzer.score.export import export_json,export_midi
from uart_buzzer.score.uart_renderer import render
from uart_buzzer.desktop.cache import ConversionCache
from uart_buzzer.desktop.engine import convert as app_convert,PrecomputedEncoder,export_document
from uart_buzzer.uart_codec import frame_bits

ROOT=Path(__file__).resolve().parents[1]


def fixture(notes=(60,64,60),duration=.22,gap=.10):
    pieces=[]
    for note in notes:
        t=np.arange(round(duration*16000))/16000
        env=np.minimum(t/.008,1)*np.minimum((duration-t)/.008,1)
        pieces.extend([.25*env*np.sin(2*np.pi*float(model.midi_to_hz(note))*t),np.zeros(round(gap*16000))])
    return np.r_[*pieces]


def base_profile(**extra):
    p=json.loads((ROOT/'desktop_assets/music_profile.json').read_text(encoding='utf-8'))
    p.update(encoder='discrete_score',prepared_volume=.8,score_config=settings({}));p.update(extra);return p


def test_finite_monophonic_raw_frontend_and_repeated_notes(tmp_path):
    x=fixture();p=base_profile()
    with patch('uart_buzzer.desktop.engine.filter_audio',side_effect=AssertionError('old EQ must not run')):
        doc=app_convert('synthetic.wav',p,source=x)
    s=score_ir(doc.metadata['score']);events=s['events']
    assert len(events)==3 and [round(e['midi_pitch']) for e in events]==[60,64,60]
    assert len(s['palette'])<=12 and s['global_semitone_shift']==0 and s['time_quantum_s']==.02
    assert all(e['velocity'] in s['velocity_levels'] for e in events)
    assert all(abs(e['start_s']-i*.32)<.06 for i,e in enumerate(events))
    assert s['source_provenance']['analysis_midi_min']==48
    assert '重复同音可能合并' in ''.join(s['diagnostics']['limitations'])
    plan=doc.metadata['transport_plan'];assert plan['rest_strategy']=='stop_writes_idle_high_no_bytes'
    for seg in doc.segments:
        bits=frame_bits(seg['data']).reshape(-1,10)
        assert np.all(bits[:,0]==0) and np.all(bits[:,-1]==1)
        assert abs(len(seg['data'])*10/p['baud']-(seg['end_s']-seg['start_s']))<1e-8
    enc=PrecomputedEncoder(doc);seg=doc.segments[0];first=round(seg['start_s']*p['baud']/10)
    data,_=enc.block(doc.processed,first,min(50,len(seg['data'])),.8)
    assert data==seg['data'][:50]
    assert enc.block(doc.processed,first,20,0)[0]==b''
    from uart_buzzer.desktop.devices import recording_reference
    np.testing.assert_array_equal(recording_reference(doc,[dict(source_s=0.,volume=.8,ramp_s=.01)]),doc.processed)
    assert not np.any(recording_reference(doc,[dict(source_s=0.,volume=0.,ramp_s=.01)]))
    export_json(s,tmp_path/'score.json');export_midi(s,tmp_path/'score.mid')
    assert load_json(tmp_path/'score.json')==s and (tmp_path/'score.mid').read_bytes().startswith(b'MThd')
    path=export_document(doc,tmp_path/'uart',.8);manifest=json.loads(path.read_text(encoding='utf-8'))
    assert manifest['encoding']['transport_plan']==plan


@pytest.mark.parametrize('kind',['silence','noise'])
def test_silence_noise_no_fake_notes(kind):
    x=np.zeros(32000) if kind=='silence' else np.random.default_rng(17).normal(0,.05,32000)
    result=convert(x,{},460800,.8,.3,.18)
    assert not result['score']['events'] and not result['segments']


def test_reference_bin_mode_never_invents_intermediate_pitch():
    lattice=dict(schema=model.LATTICE_SCHEMA,frame_times_s=np.array([0,.01]),hop_s=.01,duration_s=.02,
        midi_pitches=np.array([60.,72.]),salience=np.ones((2,2)),activity=np.ones(2),onset_strength=np.ones(2),loudness=np.ones(2),reference_midi=np.array([60.,72.]))
    _,_,features,_=model.aggregate_lattice(lattice,.02)
    assert features['reference_midi'].tolist()==[60.]


def test_hard_range_unresolved_and_device_migration():
    x=fixture((60,60))
    a=convert(x,dict(device_profile=flat_profile('A',60,60)),460800,.8,.3,.18)['score']
    b=convert(x,dict(device_profile=flat_profile('B',72,72)),460800,.8,.3,.18)['score']
    assert len(a['events'])==2 and not b['events'] and b['unresolved']
    assert b['diagnostics']['unrepresented_reference_s']>.2
    assert a['device_profile_hash']!=b['device_profile_hash']
    assert all(abs(e['midi_pitch']-60)<=2 for e in a['events'])


def test_missing_fundamental_negative_case_is_visible():
    t=np.arange(16000)/16000;f=float(model.midi_to_hz(60))
    x=.2*np.sin(2*np.pi*f*2*t)+.12*np.sin(2*np.pi*f*3*t)
    s=convert(x,{},460800,.8,.3,.18)['score']
    assert s['events'] and round(s['events'][0]['midi_pitch'])==72
    assert '缺失基音' in ''.join(s['diagnostics']['limitations'])


def test_low_confidence_gain_and_M2_do_not_become_acoustic_reward():
    p=flat_profile();p['response_kind']='manual_relative';p['notes'][0].update(gain=.05,confidence=0.)
    gains,_,_=model._response_arrays(p,True);assert gains[0]==1.
    p['response_kind']='relative_acoustic'
    with pytest.raises(ValueError):device_profile(p)
    p['response_kind']='electrical_current_proxy'
    assert model._response_arrays(p,True)[2] is False


def test_zero_volume_and_generic_emit_no_commands():
    x=fixture((60,))
    z=convert(x,{},460800,0,.3,.18);assert not z['segments'] and z['transport_plan']['duration_s']==len(x)/16000
    g=convert(x,dict(device_profile=flat_profile('generic',48,96,'generic')),460800,.8,.3,.18)
    assert not g['segments'] and g['transport_plan']['backend']=='generic_score_only' and g['score']['backend'] is None


@pytest.mark.parametrize('update',[dict(schema='unknown'),dict(notes=[]),dict(min_note_s=float('nan')),dict(max_voices=2)])
def test_invalid_profiles_rejected(update):
    p=flat_profile();p.update(update)
    with pytest.raises((ValueError,TypeError)):device_profile(p)


def test_invalid_json_score_and_resource_guard(tmp_path):
    path=tmp_path/'bad.json';path.write_text('{"schema":"x","schema":"y"}')
    with pytest.raises(ValueError):load_json(path)
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError):load_json(path)
    with pytest.raises(ValueError):score_ir(dict(schema='unknown'))
    with pytest.raises(ValueError):convert(np.array([np.nan]*1600),{},460800,.8,.3,.18)
    with pytest.raises(ValueError):settings(dict(fit=dict(palette_size=128)))
    x=fixture((60,));s=convert(x,{},460800,.8,.3,.18)['score'];bad=copy.deepcopy(s);bad['events'][0]['frequency_hz']=float('nan')
    with pytest.raises(ValueError):score_ir(bad)


def test_cancellation_prevents_cache_commit_and_progress(tmp_path):
    p=base_profile();cache=ConversionCache(tmp_path/'cache');values=[]
    def progress(value):values.append(value)
    with pytest.raises(InterruptedError):cache.convert('synthetic',p,source=fixture(),progress=progress,cancel_check=lambda:bool(values and values[-1]>.5))
    assert values and not list((tmp_path/'cache').glob('*.npz'))


def test_profile_cache_invalidation_disk_reopen_and_content_binding(tmp_path):
    store=ProfileStore(tmp_path/'profiles');a=flat_profile('A');key=store.add(a);assert ProfileStore(store.folder).profiles[key]==a
    p=base_profile();cache=ConversionCache(tmp_path/'cache');x=fixture()
    first=cache.convert('synthetic',p,source=x);second=ConversionCache(cache.folder).convert('synthetic',p,source=x)
    assert second.metadata['cache']['tier']=='disk' and first.metadata['score']==second.metadata['score']
    assert [s['data'] for s in first.segments]==[s['data'] for s in second.segments]
    for changed in [dict(device_profile=flat_profile('different name')),dict(fit=dict(palette_size=4)),dict(response_enabled=False)]:
        other=base_profile(score_config=settings(changed))
        assert cache.key('synthetic',other,'music',None,x,None)!=cache.key('synthetic',p,'music',None,x,None)
    assert cache.key('synthetic',p,'music',None,x,None)!=cache.key('synthetic',p,'music',None,x*.9,None)


def test_acoustic_response_requires_exact_reference_transport():
    from uart_buzzer.score import uart_renderer
    p=flat_profile('measured contract');p['response_kind']='relative_acoustic'
    for note in p['notes']:note['confidence']=.9
    p['provenance']=dict(measured=True,reference_transport=uart_renderer.transport_signature(460800,.8,.3,.18))
    x=fixture((60,));cfg=dict(device_profile=p)
    assert convert(x,cfg,460800,.8,.3,.18)['score']['fitting']['response_used']
    for baud,volume,bias,amplitude in [(921600,.8,.3,.18),(460800,.7,.3,.18),(460800,.8,.31,.18),(460800,.8,.3,.17)]:
        s=convert(x,cfg,baud,volume,bias,amplitude)['score']
        assert not s['fitting']['response_used'] and s['diagnostics']['response_disabled_for_transport_mismatch']
    with patch.object(uart_renderer,'VERSION','different-renderer'):
        assert not convert(x,cfg,460800,.8,.3,.18)['score']['fitting']['response_used']

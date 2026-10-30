"""Qt integration, synthetic files/fake fits only. Never opens serial or audio."""
import copy
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.io import wavfile
from PySide6 import QtWidgets

from uart_buzzer.desktop.app import MainWindow, ASSETS
from uart_buzzer.desktop.song_calibration import freeze_recording_context, file_hash

APP = None


def wait(window, predicate=lambda w:not w.loading and not w.jobs):
    until = time.monotonic() + 15
    while not predicate(window):
        QtWidgets.QApplication.processEvents()
        if time.monotonic() > until:raise AssertionError('Qt task timeout')
        time.sleep(.005)


@pytest.fixture
def window(tmp_path):
    global APP
    APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    with patch.object(MainWindow, 'refresh_ports'), patch.object(MainWindow, 'refresh_mics'):
        w = MainWindow(tmp_path)
    w.timer.stop()
    for name,f in [('A',440),('B',660)]:
        wavfile.write(tmp_path/(name+'.wav'),16000,(.2*np.sin(2*np.pi*f*np.arange(48000)/16000)).astype(np.float32))
    try:yield w
    finally:
        wait(w);w.close();QtWidgets.QApplication.processEvents()


def session(window):
    doc = window.doc
    context = freeze_recording_context(doc)
    folder = window.root/'synthetic_session';folder.mkdir()
    reference = folder/'reference_processed.wav';capture = folder/'microphone_raw.wav'
    wavfile.write(reference,16000,doc.processed);wavfile.write(capture,16000,doc.processed*.01)
    record = dict(calibration_context=context,clipped_fraction=0.,portaudio_status=[])
    meta = dict(calibration_context=context,profile=doc.profile,source_path=doc.path,completed=True,
                stopped=False,paused=False,error=None,record_source_start_s=0.,
                calibration_reference_sha256=file_hash(reference),calibration_capture_sha256=file_hash(capture))
    (folder/'recording.json').write_text(json.dumps(record));(folder/'playback.json').write_text(json.dumps(meta))
    window.last_recording=capture
    return context


def test_fit_uses_frozen_song_profile_and_does_not_change_current_B_or_global(window):
    w=window;w.load_file(w.root/'A.wav');wait(w)
    context=session(w);aid=w.current_track_id;global_before=copy.deepcopy(w.profile)
    bid=w.library.ensure(w.root/'B.wav');gate=threading.Event();received=[]
    def fit(reference,capture,prior,fit_path):
        received.append(copy.deepcopy(prior));gate.wait(5)
        result=copy.deepcopy(prior);result['gains_db'][4]+=1
        return result,{'diagnostic':'preserved'}
    with patch('uart_buzzer.desktop.app.fit_recording',side_effect=fit):
        w.fit_latest()
        w.current_track_id=bid;w.current_path=str(w.root/'B.wav')
        w.profile['name']='cosmetic user change during job'
        gate.set();wait(w)
    assert received==[context['actual_profile']]
    assert w.library.data['tracks'][aid]['calibration']['analysis']=={'diagnostic':'preserved'}
    assert 'calibration' not in w.library.data['tracks'][bid]
    assert w.profile['gains_db']==global_before['gains_db'] and not w.settings.value('profile_path')
    w.load_file(w.root/'A.wav');wait(w)
    assert w.calibration_status.text()=='本曲已校准'
    fitted_key=w.doc.metadata['cache']['key']
    w.song_calibration_box.setChecked(False);wait(w)
    assert w.calibration_status.text()=='本曲校准已旁路'
    assert w.doc.metadata['cache']['key']!=fitted_key and w.doc.metadata['cache']['hit']
    w.song_calibration_box.setChecked(True);wait(w)
    assert w.doc.metadata['cache']['key']==fitted_key and w.doc.metadata['cache']['hit']
    w.baud.setCurrentIndex(1-w.baud.currentIndex());wait(w)
    assert w.calibration_status.text()=='设置变化，本曲校准暂不适用'
    assert '_song_calibration' not in w.doc.profile


@pytest.mark.parametrize('failure',['cancel','fit_error','source_changed','removed','session_mismatch'])
def test_incomplete_fit_cannot_install_calibration(window,failure):
    w=window;w.load_file(w.root/'A.wav');wait(w);session(w);aid=w.current_track_id
    before=copy.deepcopy(w.profile);gate=threading.Event()
    def fit(*args):
        gate.wait(5)
        if failure=='fit_error':raise ValueError('synthetic failed fit')
        return copy.deepcopy(args[2]),{'x':1}
    with patch('uart_buzzer.desktop.app.fit_recording',side_effect=fit):
        w.fit_latest()
        if failure=='cancel':w.cancel_conversion()
        elif failure=='source_changed':
            path=w.root/'A.wav';raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
        elif failure=='removed':w.library.remove([aid],'all')
        elif failure=='session_mismatch':
            p=w.last_recording.parent/'recording.json';d=json.loads(p.read_text());d['calibration_context']['track_id']='other';p.write_text(json.dumps(d))
        gate.set();wait(w)
    assert w.profile==before and not w.settings.value('profile_path')
    assert 'calibration' not in w.library.data['tracks'].get(aid,{})


def test_conversion_identity_precedes_file_replacement_then_recording(window):
    w=window;w.load_file(w.root/'A.wav');wait(w)
    original=w.doc.metadata['song_context']['source_sha256']
    raw=bytearray((w.root/'A.wav').read_bytes());raw[-1]^=1;(w.root/'A.wav').write_bytes(raw)
    context=session(w)
    assert context['source_sha256']==original and context['source_sha256']!=file_hash(w.root/'A.wav')
    with patch('uart_buzzer.desktop.app.fit_recording',side_effect=AssertionError('must reject before fit')):
        w.fit_latest()
    assert not w.loading and '校准' in w.banner.text()


def test_real_conversion_memory_disk_to_playback_without_recording(window):
    from uart_buzzer.desktop.devices import Playback,Microphone
    from uart_buzzer.desktop.cache import LibraryAudioCache
    from types import SimpleNamespace
    w=window;w.load_file(w.root/'A.wav');wait(w)
    profile=copy.deepcopy(w.doc.profile)
    for doc in [w.doc,w.cache.convert(w.root/'A.wav',profile),LibraryAudioCache(w.cache.folder).convert(w.root/'A.wav',profile)]:
        assert doc.metadata['source_PCM_sha256']
        class Port:
            out_waiting=0;break_condition=False
            def write(self,data):return len(data)
            def reset_output_buffer(self):pass
        play=Playback(SimpleNamespace(connected=True,port=Port()),doc,Microphone(),w.new_session('fake_playback'),record=False)
        assert play.calibration_context is None
        play.wait=lambda seconds:False
        assert play.run()['completed']
        record=Playback(SimpleNamespace(),doc,Microphone(),w.new_session('constructor_only'),record=True)
        if doc.metadata.get('song_context'):assert record.calibration_context['actual_profile']==doc.profile
    legacy=copy.deepcopy(w.doc);legacy.metadata.pop('source_PCM_sha256')
    assert Playback(SimpleNamespace(),legacy,Microphone(),w.new_session('legacy'),record=True).calibration_context is None


def test_QA_guard_rejects_real_existing_and_mismatched_fixture_without_writes(tmp_path):
    from uart_buzzer.desktop.calibration_qa import run
    from uart_buzzer.desktop import app as module
    protected=tmp_path/'standard_root'
    with patch.object(module,'data_root',return_value=protected):
        with pytest.raises(ValueError,match='standard user'):run(protected,tmp_path/'report.json')
        with pytest.raises(ValueError,match='standard user'):run(protected/'nested',tmp_path/'report.json')
    assert not protected.exists()
    existing=tmp_path/'unrelated';existing.mkdir();(existing/'A.wav').write_bytes(b'preserve')
    with pytest.raises(ValueError,match='non-QA'):run(existing,tmp_path/'report.json')
    assert (existing/'A.wav').read_bytes()==b'preserve' and len(list(existing.iterdir()))==1
    (existing/'calibration_qa_fixture.json').write_text(json.dumps({'fixture_only':True,'schema':'wrong'}))
    with pytest.raises(ValueError,match='matching QA'):run(existing,tmp_path/'report.json')


def test_current_song_relink_refreshes_conversion_and_calibration_controls(window):
    w=window;w.load_file(w.root/'A.wav');wait(w)
    context=session(w);old=w.current_track_id
    fitted=copy.deepcopy(w.doc.profile);fitted['gains_db'][4]+=1
    w.library.bind_calibration(context,fitted,w.last_recording)
    w.queue_ids=[old]
    moved=w.root/'moved-A.wav';(w.root/'A.wav').rename(moved)
    new=w.relink_track(old,moved);wait(w)
    assert new!=old and w.current_track_id==new and w.current_path==str(moved)
    assert w.queue_ids==[new] and w.doc.path==str(moved)
    assert w.doc.metadata['song_context']['track_id']==new
    assert w.calibration_status.text()=='本曲已校准'
    assert w.library.calibration_target(context)==new
    w.song_calibration_box.setChecked(False);wait(w)
    assert w.calibration_status.text()=='本曲校准已旁路'

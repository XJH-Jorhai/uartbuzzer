"""Application-owned Qt integration tests. Physical I/O only with --qa-hardware."""
from pathlib import Path
import json
import time
import numpy as np
from PySide6 import QtCore,QtWidgets
from scipy.io import wavfile
from .app import FS,ASSETS,VERSION
from ..music import music_fixture


def start_qa(window,folder,hardware=False,source_path=None,predistortion_test=False):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    report=dict(version=VERSION,hardware=hardware,started=time.time(),events=[]);state={'stage':0,'at':time.monotonic(),'play_started':None}
    timer=QtCore.QTimer(window)
    def finish(ok,error=None):
        timer.stop();report.update(success=ok,error=error,ended=time.time(),
             volume=window.volume.value(),last_recording=str(window.last_recording),
             port_connected=window.link.connected,conversion_complete=window.doc is not None,
             graph_points=len(window.wave_curve.xData) if window.wave_curve.xData is not None else 0)
        report['layout']={key:dict(width=w.width(),height=w.height(),visible=w.isVisible(),x=w.x(),y=w.y()) for key,w in
                          [('window',window),('wave',window.wave),('spectrum',window.spectrum)]}
        window.grab().save(str(folder/'window.png'))
        (folder/'qa.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        window.close()
    def step():
        try:
            if time.monotonic()-state['at']>100:raise RuntimeError('GUI QA timeout')
            stage=state['stage']
            if stage==0:
                if predistortion_test:window.predistortion_box.setChecked(True)
                if source_path:
                    from ..music import read_audio
                    x,_=read_audio(source_path)
                else:x,_=music_fixture('train',register_semitones=12)
                path=folder/'测试音乐_中文路径.wav';wavfile.write(path,FS,x.astype(np.float32))
                window.load_file(str(path));state['stage']=1
            elif stage==1 and not window.loading:
                if window.doc is None:raise RuntimeError(window.banner.text())
                if not state.get('memory_requested'):
                    state['memory_requested']=True;window.start_conversion();return
                if not state.get('disk_requested'):
                    if window.doc.metadata.get('cache',{}).get('tier')!='memory':raise RuntimeError('Memory cache did not apply')
                    report['events'].append('Same WAV/config reused memory cache')
                    from .cache import LibraryAudioCache
                    window.cache=LibraryAudioCache(window.root/'prepared_audio',window.root/'cache/conversions')
                    state['disk_requested']=True;window.start_conversion();return
                if window.doc.metadata.get('cache',{}).get('tier')!='disk':raise RuntimeError('Persistent disk cache did not apply')
                report['events'].append('Fresh cache instance reused persistent disk cache')
                report['cache_disk_seconds']=window.doc.metadata['cache']['seconds']
                report['events'].append('Unicode WAV imported and automatically converted')
                if source_path:report['source_path']=str(source_path)
                if window.loudness_level.currentData()!=window.profile.get('loudness_target_rms',.32):raise RuntimeError('Loudness preset does not match profile')
                report['loudness_target_rms']=window.doc.profile.get('loudness_target_rms',.32)
                report['predistortion_active']=window.doc.metadata.get('predistortion',{}).get('enabled',False)
                if predistortion_test and not report['predistortion_active']:raise RuntimeError('Predistortion did not apply')
                window.volume.setValue(65);window.signal_source.setCurrentIndex(1)
                if not hardware:
                    if len(window.doc.decoded)!=len(window.doc.processed):raise RuntimeError('Decode mismatch')
                    window.update_graphs();state['stage']=9;state['screenshot_at']=time.monotonic();return
                i=window.ports.findData('COM6')
                if i<0:raise RuntimeError('COM6 unavailable')
                window.ports.setCurrentIndex(i);window.toggle_serial()
                if not window.link.connected:raise RuntimeError(window.banner.text())
                window.toggle_mic()
                if not window.mic.stream:raise RuntimeError(window.banner.text())
                window.toggle_play();state['play_started']=time.monotonic();state['stage']=2
            elif stage==2:
                elapsed=time.monotonic()-state['play_started']
                if elapsed>1.4 and not state.get('volume_changed'):
                    window.volume.setValue(25);state['volume_changed']=True;report['events'].append('Live volume 65% -> 25%')
                if elapsed>2.5 and not state.get('muted'):
                    window.mute.setChecked(True);state['muted']=True;report['events'].append('Live mute')
                if elapsed>3.2 and not state.get('unmuted'):
                    window.mute.setChecked(False);window.volume.setValue(80);state['unmuted']=True;report['events'].append('Live unmute / volume 80%')
                if not window.player:
                    if not window.last_recording:raise RuntimeError('Synchronized recording was not saved')
                    p=window.last_recording
                    rate,x=wavfile.read(p)
                    report.update(recorded_samples=len(x),recorded_rate=rate,recording_peak=float(abs(x).max()))
                    report['recording_duration_s']=len(x)/rate
                    if len(x)/rate>window.doc.duration*1.1+.5:raise RuntimeError('Playback timing exceeded realtime tolerance')
                    window.signal_source.setCurrentIndex(2);window.update_graphs();state['stage']=3
            elif stage==3:
                # Also verify that a second play can stop quickly without blocking the UI.
                window.auto_record.setChecked(False);window.toggle_play();state['stop_at']=time.monotonic();state['stage']=4
            elif stage==4 and time.monotonic()-state['stop_at']>.7:
                window.stop_play();state['stage']=5;report['events'].append('Second playback stopped by user control')
            elif stage==5 and not window.player:
                window.toggle_serial();window.toggle_mic();window.fit_latest();state['stage']=6
            elif stage==6 and not window.loading:
                if '本机音乐拟合' not in window.model_badge.text():raise RuntimeError('GUI recording-based fit failed: '+window.banner.text())
                report['events'].append('Full-recording broadband fit and automatic reconversion passed')
                if predistortion_test:
                    window.predistortion_box.setChecked(False);state['stage']=7
                else:finish(True)
            elif stage==7 and not window.loading:
                if window.doc.metadata.get('predistortion',{}).get('enabled',False):raise RuntimeError('Predistortion bypass failed')
                report['events'].append('Predistortion enabled, live volume scaling and bypass reconversion passed')
                finish(True)
            elif stage==9 and time.monotonic()-state['screenshot_at']>1:finish(True)
        except Exception as e:finish(False,str(e))
    timer.timeout.connect(step);timer.start(100)
    window._qa_timer=timer

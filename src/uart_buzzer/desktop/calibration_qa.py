"""Explicit isolated synthetic two-process release QA; no serial/microphone open."""
import copy
import hashlib
import json
from pathlib import Path
import time


def run(data_dir, report_path):
    import numpy as np
    from scipy.io import wavfile
    from PySide6 import QtWidgets
    from .app import MainWindow, VERSION, data_root
    from .song_calibration import freeze_recording_context, file_hash

    root=Path(data_dir).resolve()
    protected=[data_root().resolve(),Path.home()/'AppData/Local/ZipLab/UART Audio Studio']
    if any(root.is_relative_to(p.resolve()) for p in protected):raise ValueError('QA cannot use the real standard user data directory')
    marker=root/'calibration_qa_fixture.json'
    if root.exists() and not marker.exists() and any(root.iterdir()):
        raise ValueError('QA seed requires a new empty directory; existing non-QA data is forbidden')
    state=json.loads(marker.read_text()) if marker.exists() else None
    if state and (state.get('schema')!='uart-calibration-release-QA-v1' or state.get('fixture_only') is not True
                  or state.get('root')!=str(root) or state.get('phase')!=1):raise ValueError('Not a completed matching QA seed')
    if state:
        for name in ['A','B']:
            if file_hash(root/(name+'.wav'))!=state[name]['source_sha256']:raise ValueError('QA seed source changed')
    report_path=Path(report_path).resolve()
    if report_path.exists() or not report_path.is_relative_to(root.parent):raise ValueError('Use a new report next to the isolated QA fixture')
    root.mkdir(parents=True,exist_ok=True)

    class Window(MainWindow):
        def refresh_ports(self):pass
        def refresh_mics(self):pass

    def wait(w):
        deadline=time.monotonic()+30
        while w.loading or w.jobs:
            QtWidgets.QApplication.processEvents()
            if time.monotonic()>deadline:raise TimeoutError('Synthetic QA conversion timeout')
            time.sleep(.005)

    def payload_hash(doc):return hashlib.sha256(b''.join(s['data'] for s in doc.segments)).hexdigest()
    w=Window(root);w.timer.stop();w.show();w.resize(1160,850)
    try:
        assert w.encoder_backend.currentData()=='legacy' and w.play_mode.currentData()=='single'
        assert not w.link.connected and w.mic.stream is None
        if state is None:
            for name,frequency in [('A',440),('B',660)]:
                x=(.2*np.sin(2*np.pi*frequency*np.arange(48000)/16000)).astype(np.float32)
                wavfile.write(root/(name+'.wav'),16000,x)
            state=dict(schema='uart-calibration-release-QA-v1',fixture_only=True,phase=1,root=str(root))
            for name in ['A','B']:
                w.load_file(root/(name+'.wav'));wait(w)
                state[name]=dict(key=w.doc.metadata['cache']['key'],BINsha256=payload_hash(w.doc),source_sha256=file_hash(root/(name+'.wav')))
                if name=='A':context=freeze_recording_context(w.doc);prior=copy.deepcopy(w.doc.profile)
            fitted=copy.deepcopy(prior);fitted['gains_db'][4]+=1
            aid=w.library.calibration_target(context)
            w.library.bind_calibration(context,fitted,'synthetic QA only',{'kind':'synthetic lifecycle check, not acoustic fit'})
            w.load_file(root/'A.wav');wait(w)
            state['calibrated_A']=dict(key=w.doc.metadata['cache']['key'],BINsha256=payload_hash(w.doc))
            assert w.calibration_status.text()=='本曲已校准'
            phase='seed'
        else:
            phase='restart'
            for name,key in [('A','calibrated_A'),('B','B')]:
                w.load_file(root/(name+'.wav'));wait(w)
                assert w.doc.metadata['cache']['tier']=='disk'
                assert w.doc.metadata['cache']['key']==state[key]['key'] and payload_hash(w.doc)==state[key]['BINsha256']
            w.load_file(root/'A.wav');wait(w)
        global_gains=copy.deepcopy(w.profile['gains_db'])
        w.song_calibration_box.setChecked(False);wait(w)
        assert w.doc.metadata['cache']['hit'] and w.doc.metadata['cache']['key']==state['A']['key']
        assert payload_hash(w.doc)==state['A']['BINsha256']
        w.song_calibration_box.setChecked(True);wait(w)
        assert w.doc.metadata['cache']['hit'] and w.doc.metadata['cache']['key']==state['calibrated_A']['key']
        assert w.profile['gains_db']==global_gains and not w.settings.value('profile_path')
        for widget in w.findChildren(QtWidgets.QToolButton):
            if widget.text().startswith('配置与录音拟合'):widget.setChecked(True)
        w.library_dock.show();w.update_graphs()
        QtWidgets.QApplication.processEvents()
        screenshot=Path(report_path).with_suffix('.png');w.grab().save(str(screenshot))
        marker.write_text(json.dumps(state,indent=2),encoding='utf-8')
        w.settings.sync()
        result=dict(success=True,version=VERSION,phase=phase,root=str(root),fixture=state,
                    song_A_only=True,global_gains_unchanged=True,bypass_original_cache_hit=True,
                    restart_disk_hits=phase=='restart',default_encoder='B0',default_play_mode='single',
                    serial_opened=False,microphone_opened=False,acoustic_quality_test=False,screenshot=str(screenshot))
        Path(report_path).write_text(json.dumps(result,indent=2),encoding='utf-8')
    finally:
        wait(w);w.close();QtWidgets.QApplication.processEvents()

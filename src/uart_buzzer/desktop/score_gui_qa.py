"""Native Qt preview verification with synthetic audio and fake I/O only."""
import json,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PySide6 import QtWidgets
from .score_qa import fixture


def run(root):
    from .app import MainWindow,data_root
    from .score_ui import ScoreDialog,CharacterizationDialog
    from .characterization import run as characterize
    root=Path(root).resolve()
    if root.is_relative_to(data_root().resolve()) or root.is_relative_to((Path.home()/'AppData/Local/ZipLab/UART Audio Studio').resolve()):raise ValueError('需要隔离目录')
    if root.exists() and any(root.iterdir()):raise ValueError('使用新的空GUI检查目录')
    root.mkdir(parents=True,exist_ok=True)
    def wait(w):
        until=time.monotonic()+30
        while w.loading or w.jobs:
            QtWidgets.QApplication.processEvents()
            if time.monotonic()>until:raise AssertionError('UI job timeout')
            time.sleep(.005)
    with patch.object(MainWindow,'refresh_ports'),patch.object(MainWindow,'refresh_mics'):w=MainWindow(root/'user')
    try:
        w.timer.stop();assert w.encoder_backend.currentData()=='legacy'
        w.current_source=fixture();w.current_path='qa://original-synthetic';w.encoder_backend.setCurrentIndex(w.encoder_backend.findData('discrete_score'));wait(w)
        assert w.doc.metadata['score']['events'];w.show();QtWidgets.QApplication.processEvents()
        w.ensure_visible();w.update_graphs()
        for _ in range(10):QtWidgets.QApplication.processEvents();time.sleep(.02)
        w.grab().save(str(root/'main.png'))
        d=ScoreDialog(w);d.show();QtWidgets.QApplication.processEvents();d.grab().save(str(root/'settings.png'))
        tabs=d.findChild(QtWidgets.QTabWidget);tabs.setCurrentIndex(2);QtWidgets.QApplication.processEvents();d.grab().save(str(root/'events.png'))
        w.ports.addItem('FAKE · offline only','FAKE');w.mics.addItem('SIMULATED microphone',dict(index=0,rate=16000))
        measurement=CharacterizationDialog(w,d);measurement.name.setText('SIMULATED UI QA only');measurement.low.setValue(60);measurement.high.setValue(61);measurement.show()
        QtWidgets.QApplication.processEvents();measurement.grab().save(str(root/'characterization_before_start.png'))
        class FakeIO:
            def __init__(self,cfg):self.rate=cfg['rate']
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def background(self,seconds,cancel):return np.random.default_rng(9).normal(0,.0001,round(seconds*self.rate))
            def note(self,hz,volume,duration,gap,cancel):
                t=np.arange(round(duration*self.rate))/self.rate
                return np.r_[.2*volume*np.sin(2*np.pi*hz*t),np.zeros(round(gap*self.rate))],dict(statuses=[])
        count=len(w.device_profiles.profiles)
        with patch('uart_buzzer.desktop.characterization.run',lambda *args:characterize(*args,io_factory=FakeIO)):
            measurement.begin();wait(w);QtWidgets.QApplication.processEvents()
        assert measurement.pending is not None and len(w.device_profiles.profiles)==count
        assert measurement.pending['response_kind']=='synthetic_relative' and not measurement.pending['provenance']['measured']
        measurement.grab().save(str(root/'characterization_pending_review.png'));measurement.accept_profile()
        assert len(w.device_profiles.profiles)==count+1
        assert w.score_config['device_profile']['response_kind']=='unmeasured_flat'
        # Exercise cancellation while the Qt event loop remains responsive.
        tab_count=tabs.count();d.close();w.current_source=np.tile(fixture(),20);w.start_conversion();w.cancel_conversion();wait(w)
        report=dict(success=True,physical_IO=False,native_Qt=True,source='original synthetic only',tabs=tab_count,
                    profile_pending_review_then_explicit_save=True,profile_selection_unchanged=True,cancelled_background_job=not w.loading,
                    screen_size=[w.screen().availableGeometry().width(),w.screen().availableGeometry().height()])
        (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report
    finally:
        for job in w.jobs:job.requestInterruption()
        wait(w);w.close();QtWidgets.QApplication.processEvents()

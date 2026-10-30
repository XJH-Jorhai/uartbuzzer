"""Qt product entry points with synthetic audio and no physical I/O."""
import json,time
from unittest.mock import patch
import pytest
from PySide6 import QtWidgets
from uart_buzzer.desktop.app import MainWindow
from uart_buzzer.desktop.score_ui import ScoreDialog,CharacterizationDialog
from uart_buzzer.desktop.score_qa import fixture
from uart_buzzer.score.contracts import flat_profile

APP=None


def wait(w):
    until=time.monotonic()+30
    while w.loading or w.jobs:
        QtWidgets.QApplication.processEvents()
        if time.monotonic()>until:raise AssertionError('Qt worker failed to finish')
        time.sleep(.005)


@pytest.fixture
def window(tmp_path):
    global APP
    APP=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    with patch.object(MainWindow,'refresh_ports'),patch.object(MainWindow,'refresh_mics'):
        w=MainWindow(tmp_path/'user')
    w.timer.stop()
    try:yield w
    finally:wait(w);w.close();QtWidgets.QApplication.processEvents()


def test_score_ui_default_selection_tables_and_back_to_B0(window):
    w=window;assert w.encoder_backend.currentData()=='legacy'
    original=w.profile['gains_db'][:];w.current_path='qa://original';w.current_source=fixture()
    w.encoder_backend.setCurrentIndex(w.encoder_backend.findData('discrete_score'));wait(w)
    assert w.doc.metadata.get('score') and not w.mode.isEnabled() and not w.song_calibration_box.isEnabled()
    d=ScoreDialog(w);assert d.findChild(QtWidgets.QTabWidget).count()==4
    assert '原主线时长覆盖' in d.summary.text();d.close()
    assert w.profile['gains_db']==original
    w.encoder_backend.setCurrentIndex(0);wait(w)
    assert not w.doc.metadata.get('score') and w.mode.isEnabled()


def test_explicit_generic_profile_restart_and_no_playback(window):
    w=window;key=w.device_profiles.add(flat_profile('generic test',48,96,'generic'))
    w.current_source=fixture();w.current_path='qa://original';d=ScoreDialog(w);d.reload_profiles(key);d.apply();wait(w)
    assert w.doc.metadata['transport_plan']['backend']=='generic_score_only' and not w.play_button.isEnabled()
    with patch.object(w.link,'open',side_effect=AssertionError('must not open transport')):w.toggle_play()
    assert w.score_config['device_profile']['device_id']=='generic test'
    saved=json.loads(w.settings.value('score_config'));assert saved['device_profile']['transport']=='generic'
    with patch.object(MainWindow,'refresh_ports'),patch.object(MainWindow,'refresh_mics'):
        restored=MainWindow(w.root)
    try:assert restored.score_config==w.score_config and restored.encoder_backend.currentData()=='discrete_score'
    finally:restored.timer.stop();restored.close()


def test_measurement_dialog_is_inert_until_begin(window):
    with patch('uart_buzzer.desktop.characterization.HardwareIO.__enter__',side_effect=AssertionError('no automatic I/O')):
        d=ScoreDialog(window);measurement=CharacterizationDialog(window,d)
        assert measurement.job is None and measurement.pending is None and not measurement.save.isEnabled()
        measurement.close();d.close()

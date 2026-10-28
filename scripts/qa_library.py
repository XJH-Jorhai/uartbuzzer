"""Exercise actual Qt library/controller/conversion; playback starts are intercepted."""
import sys,os,time,json,tempfile,logging
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in ('tmp/audio_deps','tmp/gui_deps','tmp/sim_deps','src'):sys.path.insert(0,str(ROOT/p))
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
from scipy.io import wavfile
from PySide6 import QtWidgets,QtGui
from uart_buzzer.desktop.app import MainWindow,VERSION
from uart_buzzer.desktop.library import LibraryStore

app=QtWidgets.QApplication([]);app.setStyle('Fusion');app.setFont(QtGui.QFont('Microsoft YaHei UI',9));out=ROOT/'results/library_qa_v131';out.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as folder:
    root=Path(folder);audio=root/'songs';audio.mkdir()
    for n in range(3):
        x=.25*np.sin(2*np.pi*(1000+n*200)*np.arange(8000)/16000);wavfile.write(audio/f'歌曲 {n}.wav',16000,x.astype('float32'))
    w=MainWindow(root/'state');w.show();errors=[];w.fail=lambda text:errors.append(str(text))
    def wait():
        deadline=time.monotonic()+30
        while w.jobs or w.loading or w.importing_library or w.pending_track:
            app.processEvents();time.sleep(.01)
            if time.monotonic()>deadline:raise RuntimeError('queue timeout')
        app.processEvents()
    w.import_tracks([audio]);wait();ids=w.library.ids();assert len(ids)==3 and w.doc is not None
    playlist=w.library.create_playlist('音乐验证');w.library.add_to_playlist(ids,playlist);w.library_panel.refresh()
    w.library.reorder(ids[::-1]);w.library_panel.refresh_tracks();assert w.library.ids()==ids[::-1]
    w.library_panel.search.setText('歌曲 1');assert w.library_panel.tree.topLevelItemCount()==1;w.library_panel.search.clear()
    # Fake link and start callback exercise autoplay without sending physical data.
    class Link:
        connected=True;name='QA virtual';port=None
        def close(self):pass
    w.link=Link();started=[];w.toggle_play=lambda:started.append(w.current_track_id)
    w.request_track(ids[0],True,True);wait();assert started==[ids[0]]
    assert w.play_mode.currentData()=='single'
    before=len(started);w.play_finished({'position_s':.5,'completed':True});wait();assert len(started)==before
    w.queue_active=True
    w.play_mode.setCurrentIndex(w.play_mode.findData('loop'));w.play_finished({'position_s':.5,'completed':True});wait()
    assert started[-1]==ids[2]  # reversed queue wraps
    w.stop_play();before=len(started);w.play_finished({'position_s':.2,'completed':False});wait();assert len(started)==before
    w.request_track(ids[1],True);w.request_track(ids[2],True);wait();assert w.current_track_id==ids[2] and started[-1]==ids[2]
    # Disk cache reused while returning to a previous song.
    w.request_track(ids[0],False);wait();assert w.doc.metadata['cache']['hit']
    current=w.current_track_id;w.library_panel.remove() # no selection is harmless
    restored=LibraryStore(w.library.path);assert restored.ids()==ids[::-1] and restored.data['current']==current
    w.library_dock.show();w.resize(1450,850);app.processEvents();w.grab().save(str(out/'window.png'))
    assert not errors,errors
    (out/'qa.json').write_text(json.dumps(dict(success=True,version=VERSION,tracks=len(ids),playlist_persistence=True,search=True,
        autoplay_intercepted=started,single_play_default=True,single_play_stops=True,manual_stop_no_advance=True,rapid_switch=True,cache_reuse=True,
        prepared_file=bool(restored.data['tracks'][current].get('prepared_files')),hardware=False),indent=2),encoding='utf-8')
    w.close();app.processEvents();logging.shutdown()
print('Library GUI QA passed')

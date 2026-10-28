import tempfile,unittest,wave,random
from pathlib import Path
from types import SimpleNamespace
from uart_buzzer.desktop.library import LibraryStore,inspect_paths,probe_track,move_items,next_track

class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.paths=[]
        for name in ['中文歌曲','second','third']:
            p=self.root/(name+'.wav')
            with wave.open(str(p),'wb') as f:f.setparams((1,2,16000,0,'NONE',''));f.writeframes(b'\0\0'*8000)
            self.paths.append(p)
        self.store=LibraryStore(self.root/'library.json');self.ids=self.store.add(inspect_paths(self.paths))
    def tearDown(self):self.tmp.cleanup()
    def test_metadata_and_dedup(self):
        self.assertEqual(probe_track(self.paths[0])['duration_s'],.5)
        self.store.add(inspect_paths([self.root,self.paths[0]]));self.assertEqual(len(self.store.ids()),3)
    def test_playlist_persistence_and_reference_only_remove(self):
        p=self.store.create_playlist('喜欢');self.store.add_to_playlist(self.ids[:2],p);self.store.reorder(self.ids[:2][::-1],p)
        restored=LibraryStore(self.store.path);self.assertEqual(restored.ids(),self.ids[:2][::-1])
        restored.remove([self.ids[0]]);self.assertEqual(len(restored.ids()),1);self.assertEqual(len(restored.ids('all')),3)
        restored.remove([self.ids[0]],'all');self.assertTrue(self.paths[0].exists());self.assertNotIn(self.ids[0],restored.ids(p))
    def test_missing_and_relink(self):
        self.paths[0].rename(self.root/'moved.wav');self.assertNotIn(self.ids[0],self.store.playable(self.ids))
        new=self.store.relink(self.ids[0],self.root/'moved.wav');self.assertEqual(self.store.ids()[0],new);self.assertIn(new,self.store.ids('default'))
    def test_cache_reference_and_changed_file(self):
        doc=SimpleNamespace(duration=.5,metadata={'cache':{'key':'a'*64}},profile={'baud':460800});self.store.converted(self.ids[0],doc)
        self.store.add(inspect_paths([self.paths[0]]));self.assertEqual(self.store.data['tracks'][self.ids[0]]['cache_keys'],['a'*64])
        with self.paths[0].open('ab') as f:f.write(b'0')
        self.store.ensure(self.paths[0]);self.assertEqual(self.store.data['tracks'][self.ids[0]]['cache_keys'],[])
    def test_navigation(self):
        a,b,c=self.ids
        self.assertIsNone(next_track(self.ids,b,'single',automatic=True));self.assertEqual(next_track(self.ids,b,'single'),c)
        self.assertIsNone(next_track(self.ids,c,automatic=True));self.assertEqual(next_track(self.ids,c,'loop',automatic=True),a)
        self.assertEqual(next_track(self.ids,b,'repeat_one',automatic=True),b);self.assertEqual(next_track(self.ids,b,'repeat_one'),c)
        self.assertNotEqual(next_track(self.ids,b,'shuffle',rng=random.Random(0)),b);self.assertEqual(next_track(self.ids,a,direction=-1),c)
        self.assertIsNone(next_track([],a));self.assertEqual(move_items(self.ids,[b,c],-1),[b,c,a])
    def test_corrupt_and_invalid(self):
        self.store.path.write_text('{bad');restored=LibraryStore(self.store.path);self.assertTrue(restored.warning);self.assertEqual(restored.ids(),self.ids)
        with self.assertRaises(ValueError):self.store.create_playlist(' ')
        with self.assertRaises(ValueError):self.store.reorder(self.ids[:1])
    def test_both_corrupt_cannot_overwrite(self):
        self.store.path.write_text('{bad');self.store.path.with_suffix('.json.bak').write_text('{bad')
        restored=LibraryStore(self.store.path)
        with self.assertRaises(ValueError):restored.add(inspect_paths(self.paths))
        self.assertEqual(self.store.path.read_text(),'{bad')
    def test_prepared_files_persist_on_reimport(self):
        artifact=self.root/'prepared.npz';artifact.write_bytes(b'test')
        doc=SimpleNamespace(duration=.5,metadata={'cache':{'key':'a'*64}},profile={'baud':921600})
        self.store.converted(self.ids[0],doc,artifact);self.store.add(inspect_paths([self.paths[0]]))
        restored=LibraryStore(self.store.path);self.assertEqual(restored.data['tracks'][self.ids[0]]['prepared_files'][0]['file'],str(artifact))

if __name__=='__main__':unittest.main()

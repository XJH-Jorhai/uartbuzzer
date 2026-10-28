import copy,json,os,unittest,uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.io import wavfile
from uart_buzzer.music import FS,music_fixture
from uart_buzzer.desktop.cache import ConversionCache,LibraryAudioCache

ROOT=Path(__file__).resolve().parents[1]


class ConversionCacheTests(unittest.TestCase):
    def setUp(self):
        self.folder=ROOT/'results/cache_tests'/uuid.uuid4().hex[:12];self.folder.mkdir(parents=True)
        self.profile=json.loads((ROOT/'desktop_assets/music_profile.json').read_text(encoding='utf-8'))
        self.x=music_fixture('train')[0][:FS//2]
        self.wav=self.folder/'中文歌曲.wav';wavfile.write(self.wav,FS,self.x.astype(np.float32))
        self.cache=ConversionCache(self.folder/'cache')

    def test_memory_and_restarted_disk_hits_skip_conversion_and_preserve_bytes(self):
        first=self.cache.convert(self.wav,self.profile)
        self.assertTrue(first.metadata['cache']['persisted']);self.assertFalse(first.metadata['cache']['hit'])
        first.metadata['conditioning']['RMS']=-123
        with patch('uart_buzzer.desktop.cache.convert',side_effect=AssertionError('Unexpected reconversion')):
            memory=self.cache.convert(self.wav,self.profile)
            self.assertEqual(memory.metadata['cache']['tier'],'memory')
            disk=ConversionCache(self.folder/'cache').convert(self.wav,self.profile)
            self.assertEqual(disk.metadata['cache']['tier'],'disk')
        self.assertNotEqual(memory.metadata['conditioning']['RMS'],-123)
        for a,b in zip(memory.segments,disk.segments):self.assertEqual(a['data'],b['data'])
        np.testing.assert_array_equal(memory.processed,disk.processed)
        np.testing.assert_array_equal(memory.decoded,disk.decoded)

    def test_library_files_migrate_and_survive_cache_eviction(self):
        first=self.cache.convert(self.wav,self.profile)
        prepared=LibraryAudioCache(self.folder/'prepared',self.cache.folder,max_entries=1)
        with patch('uart_buzzer.desktop.cache.convert',side_effect=AssertionError('Unexpected reconversion')):
            migrated=prepared.convert(self.wav,self.profile)
        key=first.metadata['cache']['key'];self.assertTrue((prepared.folder/(key+'.npz')).exists())
        # A second configuration must not evict the first song's prepared file.
        changed=dict(self.profile,baud=921600);prepared.convert(self.wav,changed);prepared.prune()
        self.assertEqual(len(list(prepared.folder.glob('*.npz'))),2)
        self.cache.max_entries=0;self.cache.prune()
        with patch('uart_buzzer.desktop.cache.convert',side_effect=AssertionError('Unexpected reconversion')):
            restarted=LibraryAudioCache(prepared.folder).convert(self.wav,self.profile)
        self.assertEqual(restarted.metadata['cache']['tier'],'disk')
        self.assertEqual([s['data'] for s in migrated.segments],[s['data'] for s in restarted.segments])

    def test_file_content_change_invalidates_even_with_same_size_and_timestamp(self):
        first=self.cache.convert(self.wav,self.profile);stat=self.wav.stat()
        wavfile.write(self.wav,FS,(-self.x).astype(np.float32));os.utime(self.wav,ns=(stat.st_atime_ns,stat.st_mtime_ns))
        self.assertEqual(self.wav.stat().st_size,stat.st_size)
        second=self.cache.convert(self.wav,self.profile)
        self.assertFalse(second.metadata['cache']['hit']);self.assertNotEqual(first.metadata['cache']['key'],second.metadata['cache']['key'])

    def test_relevant_settings_and_fit_content_invalidate_but_labels_do_not(self):
        key=lambda p,mode='music',fit=None:self.cache.key(self.wav,p,mode,fit,None,None)
        original=key(self.profile);renamed=dict(self.profile,name='新名称',evidence='新说明')
        self.assertEqual(original,key(renamed))
        with patch.dict('uart_buzzer.desktop.cache.PROCESSOR',{'engine.py':'different-algorithm'}):
            self.assertNotEqual(original,key(self.profile))
        variants=[]
        for field,value in [('baud',921600),('loudness_target_rms',.32),('continuous_carrier',False),('soft_carrier_edges',False),('encoder','frame_density_feedback')]:
            p=dict(self.profile);p[field]=value;variants.append(key(p))
        p=copy.deepcopy(self.profile);p['gains_db'][0]+=.5;variants.append(key(p))
        p=dict(self.profile,predistortion=self.profile['available_predistortion']);variants.append(key(p))
        self.assertTrue(all(k!=original for k in variants));self.assertEqual(len(set(variants)),len(variants))
        fit=self.folder/'fit.json';fit.write_text('{"a":1}')
        before=key(self.profile,'electrical',fit);fit.write_text('{"a":2}')
        self.assertNotEqual(before,key(self.profile,'electrical',fit))

    def test_predistortion_arrays_survive_disk_roundtrip_and_current_name_is_used(self):
        p=dict(self.profile,predistortion=self.profile['available_predistortion'])
        first=self.cache.convert(self.wav,p)
        renamed=self.folder/'另一个名称.wav';renamed.write_bytes(self.wav.read_bytes())
        second=ConversionCache(self.folder/'cache').convert(renamed,p)
        self.assertTrue(second.metadata['cache']['hit']);self.assertEqual(second.name,renamed.name);self.assertEqual(second.path,str(renamed))
        np.testing.assert_array_equal(first.base_processed,second.base_processed)
        np.testing.assert_array_equal(first.quadratic_correction,second.quadratic_correction)

    def test_corrupt_entry_rebuilds_and_cancellation_does_not_create_entry(self):
        first=self.cache.convert(self.wav,self.profile);key=first.metadata['cache']['key']
        (self.folder/'cache'/(key+'.npz')).write_bytes(b'broken archive')
        rebuilt=ConversionCache(self.folder/'cache').convert(self.wav,self.profile)
        self.assertFalse(rebuilt.metadata['cache']['hit']);self.assertTrue(rebuilt.metadata['cache']['persisted'])
        empty=self.folder/'cancelled'
        with self.assertRaises(InterruptedError):ConversionCache(empty).convert(self.wav,self.profile,cancel_check=lambda:True)
        self.assertFalse(empty.exists())

    def test_storage_failure_does_not_prevent_playable_document(self):
        with patch.object(self.cache,'save',side_effect=OSError('disk full')):
            doc=self.cache.convert(self.wav,self.profile)
        self.assertTrue(doc.segments);self.assertFalse(doc.metadata['cache']['persisted'])
        self.assertTrue(self.cache.convert(self.wav,self.profile).metadata['cache']['hit'])

    def test_lru_evicts_only_cache_entries_within_limits(self):
        cache=ConversionCache(self.folder/'cache',memory_bytes=1,max_entries=2)
        cache.folder.mkdir();foreign=cache.folder/'unrelated.npz';foreign.write_bytes(b'user file')
        keys=[]
        for gain in (.32,.52,.7):
            p=dict(self.profile,loudness_target_rms=gain);keys.append(cache.convert(self.wav,p).metadata['cache']['key'])
        self.assertEqual(len(list((self.folder/'cache').glob('*.npz'))),3)
        self.assertFalse((self.folder/'cache'/(keys[0]+'.npz')).exists())
        self.assertTrue(self.wav.exists());self.assertTrue(foreign.exists());self.assertEqual(cache.used,0)

if __name__=='__main__':unittest.main()

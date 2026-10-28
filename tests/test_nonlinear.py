import unittest
import numpy as np
from scipy.signal import butter,sosfiltfilt
from uart_buzzer.music import FS,music_fixture
from uart_buzzer.nonlinear import quadratic_feature,predistort,align_repeat,validate_predistortion
from uart_buzzer.desktop.engine import LiveEncoder
from uart_buzzer.desktop.engine import convert,export_document


class NonlinearMusicTests(unittest.TestCase):
    def test_universal_default_does_not_apply_track_scoped_candidate(self):
        import json
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        profile=json.loads((root/'desktop_assets/music_profile.json').read_text(encoding='utf-8'))
        self.assertFalse(profile.get('predistortion_default_enabled',False))
        self.assertNotIn('predistortion',profile)
        self.assertIn('available_predistortion',profile)
        x,_=music_fixture('holdout');doc=convert('default.wav',profile,source=x[:FS])
        self.assertFalse(doc.metadata['predistortion']['enabled'])

    def test_conversion_and_export_use_the_same_volume_aware_predistortion(self):
        import json,uuid
        from pathlib import Path
        from uart_buzzer.music import encode_pcm
        root=Path(__file__).resolve().parents[1]
        p=json.loads((root/'config/music_legacy_v1.json').read_text(encoding='utf-8'))
        kernel=np.zeros(33);kernel[16]=.25
        p['predistortion']=dict(kind='quadratic_memory_v1',kernel2=kernel.tolist(),strength=1)
        x,_=music_fixture('train');doc=convert('test.wav',p,source=x[:FS*2])
        self.assertTrue(doc.metadata['predistortion']['enabled'])
        np.testing.assert_allclose(doc.processed,doc.base_processed+doc.quadratic_correction,atol=1e-7)
        expected,_=encode_pcm(doc.base_processed*.5+doc.quadratic_correction*.25,
                              baud=p['baud'],bias=p['bias'],amplitude=p['amplitude'],method=p['encoder'])
        folder=root/'results/nonlinear_tests'/uuid.uuid4().hex[:12]
        manifest=export_document(doc,folder,.5);records=json.loads(manifest.read_text(encoding='utf-8'))['segments']
        self.assertEqual(len(records),len(expected))
        for record,segment in zip(records,expected):self.assertEqual((folder/record['file']).read_bytes(),segment['data'])

    def test_live_volume_scales_quadratic_correction_as_volume_squared(self):
        import json
        from pathlib import Path
        p=json.loads((Path(__file__).resolve().parents[1]/'config/music_legacy_v1.json').read_text(encoding='utf-8'))
        x,_=music_fixture('holdout');kernel=np.zeros(33);kernel[16]=.3
        _,delta,_=predistort(x*.8,dict(kind='quadratic_memory_v1',kernel2=kernel.tolist(),strength=1))
        e=LiveEncoder(p);e.gain=.5
        _,actual=e.block(x*.8,1000,2048,.5,correction=delta)
        times=(np.arange(1000*10,(1000+2048)*10)+.5)/p['baud'];tx=np.arange(len(x))/FS
        expected=np.interp(times,tx,x*.8)*.5+np.interp(times,tx,delta)*.25
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        wrong=np.interp(times,tx,x*.8+delta)*.5
        self.assertGreater(np.std(actual-wrong),1e-4)

    def test_quadratic_feature_is_even_and_rejects_dc(self):
        x,_=music_fixture('holdout');a=quadratic_feature(x);b=quadratic_feature(-x)
        np.testing.assert_allclose(a,b,atol=1e-13)
        self.assertLess(abs(np.mean(a)),.001)
        self.assertGreater(np.std(a),.005)

    def test_bounded_correction_preserves_clean_path_and_density_headroom(self):
        x,_=music_fixture('train');x*=.95
        kernel=np.zeros(33);kernel[16]=3.;kernel[15]=-1.
        model=dict(kind='quadratic_memory_v1',kernel2=kernel.tolist(),strength=1.)
        y,delta,meta=predistort(x,model)
        np.testing.assert_allclose(y,x+delta)
        self.assertLessEqual(max(abs(y)),.985001)
        self.assertLessEqual(np.std(delta),np.std(x)*.2501)
        self.assertTrue(meta['enabled'])
        bypass,zero,_=predistort(x,dict(model,strength=0))
        np.testing.assert_array_equal(bypass,x);self.assertEqual(np.max(abs(zero)),0)

    def test_invalid_models_rejected(self):
        for kernel in ([1,2],[1,float('nan'),1],[3,3,3]):
            with self.assertRaises(ValueError):validate_predistortion(dict(kind='quadratic_memory_v1',kernel2=kernel))

    def test_subsample_alignment_recovers_delay_and_relative_clock(self):
        rng=np.random.default_rng(72);n=FS*5;t=np.arange(n)/FS
        x=sosfiltfilt(butter(3,[1000,4500],fs=FS,btype='bandpass',output='sos'),rng.normal(size=n))
        delay=.003173;drift=.000026
        shifted=np.interp((t-delay)/(1+drift),t,-x,left=0,right=0)
        aligned,meta=align_repeat(x,shifted,polarity=-1)
        self.assertAlmostEqual(meta['offset_s'],delay,delta=.00003)
        self.assertAlmostEqual(meta['relative_clock_drift'],drift,delta=.00001)
        self.assertGreater(np.corrcoef(x[1000:-1000],-aligned[1000:-1000])[0,1],.99)

if __name__=='__main__':unittest.main()

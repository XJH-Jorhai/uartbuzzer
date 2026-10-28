import tempfile
import unittest
from io import BytesIO
from scipy.io import wavfile
import numpy as np
from uart_buzzer.music import FS,KNOTS,music_fixture,read_audio,filter_audio,encode_pcm,fit_acoustic_response,music_metrics
from uart_buzzer.uart_codec import frame_bits


class MusicTests(unittest.TestCase):
    def test_unsigned_pcm_silence_is_centered_before_resampling(self):
        stream=BytesIO();wavfile.write(stream,8000,np.full(800,128,dtype=np.uint8));stream.seek(0)
        y,sr=read_audio(stream)
        self.assertEqual(sr,FS);self.assertLess(abs(y).max(),1e-12)
    def test_sources_are_polyphonic_and_holdout_is_distinct(self):
        a,am=music_fixture('train');b,bm=music_fixture('holdout')
        self.assertEqual(len(a),6*FS)
        self.assertGreater(sum(n['start_s']==0 for n in am['notes']),3)
        self.assertFalse(np.array_equal(a,b))
        self.assertTrue(np.all(a[int(2.82*FS):int(2.98*FS)]==0))

    def test_arbitrary_pcm_is_encoded_as_true_8n1(self):
        rng=np.random.default_rng(71);x=rng.normal(0,.1,1600)
        segments,meta=encode_pcm(x,baud=460800)
        data=b''.join(s['data'] for s in segments);bits=frame_bits(data).reshape(-1,10)
        self.assertTrue(np.all(bits[:,0]==0));self.assertTrue(np.all(bits[:,-1]==1))
        expected=.3+.18*np.interp((np.arange(len(bits)*10)+.5)/460800,np.arange(len(x))/FS,x)
        self.assertLess(abs(np.sum(expected-(1-bits.ravel()))),1.5)
        self.assertFalse(meta['tone_specific_harmonic_correction'])

    def test_real_rests_are_scheduled_idle(self):
        x=np.r_[np.full(1600,.2),np.zeros(3200),np.full(1600,.2)]
        segs,meta=encode_pcm(x)
        self.assertEqual(len(segs),2)
        self.assertGreater(segs[1]['start_s']-segs[0]['end_s'],.15)
        self.assertGreater(meta['real_idle_s'],.15)

    def test_frame_density_feedback_has_two_edges_and_bounded_error(self):
        x,_=music_fixture('train');x=x[:1600]*.5
        segs,meta=encode_pcm(x,method='frame_density_feedback')
        bits=frame_bits(b''.join(s['data'] for s in segs)).reshape(-1,10)
        self.assertTrue(np.all(np.count_nonzero(np.diff(bits,axis=1),axis=1)==1))
        self.assertLess(abs(meta['segment_final_integral_errors'][0]),.500001)

    def test_multinote_clock_scale_recovery(self):
        x,_=music_fixture('train');x,_=filter_audio(x)
        warped=np.interp(np.arange(len(x))/1.006,np.arange(len(x)),x)*.02
        fitted=fit_acoustic_response(x,warped)
        self.assertLess(abs(fitted['clock_scale']-1.006),.001)

    def test_filter_peak_budget_and_band_error(self):
        x,_=music_fixture('holdout');y,m=filter_audio(x,np.linspace(-6,6,len(KNOTS)))
        self.assertLessEqual(abs(y).max(),.9500001)
        r=music_metrics(y,y)
        self.assertLess(r['log_band_error_RMS_dB'],1e-8)
        self.assertGreater(r['envelope_correlation'],.999)

    def test_broadband_fitting_uses_music_not_tone(self):
        x,_=music_fixture('train');x,_=filter_audio(x)
        shaped,_=filter_audio(x,np.linspace(-4,4,len(KNOTS)))
        fitted=fit_acoustic_response(x,shaped*.02)
        self.assertGreater(fitted['excited_frequency_bins'],100)
        self.assertLess(abs(fitted['clock_scale']-1),.002)
        self.assertGreater(np.corrcoef(fitted['relative_chain_response_db'],np.linspace(-4,4,len(KNOTS)))[0,1],.8)


if __name__=='__main__':unittest.main()

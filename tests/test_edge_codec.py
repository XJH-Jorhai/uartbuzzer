import unittest
import numpy as np
from scipy.signal import butter,sosfilt
from uart_buzzer.uart_codec import edge_feedback_block,bit_feedback_block,frame_bits
from uart_buzzer.music import FS,music_fixture


class EdgeAwareCodecTests(unittest.TestCase):
    def test_zero_delay_model_matches_existing_encoder_exactly(self):
        t=np.random.default_rng(91).uniform(.2,.75,20000)
        a,sa=bit_feedback_block(t,.17)
        b,sb=edge_feedback_block(t,.17,on_loss=0,off_tail=0,bias_offset=0)
        self.assertEqual(a,b);self.assertAlmostEqual(sa,sb,places=8)

    def test_delivered_area_integral_and_state_survive_packet_boundaries(self):
        t=.45+.25*np.sin(np.arange(20000)*.03)
        a,s=edge_feedback_block(t[:10000]);b,s=edge_feedback_block(t[10000:],s)
        whole,sw=edge_feedback_block(t)
        self.assertEqual(a+b,whole);self.assertAlmostEqual(s,sw)
        bits=frame_bits(whole).reshape(-1,10)
        self.assertTrue(np.all(bits[:,0]==0));self.assertTrue(np.all(bits[:,-1]==1))
        on=(1-bits).ravel();previous=np.r_[0,on[:-1]]
        delivered=on*(1-.01838462745*(1-previous))+(1-on)*.10345002714*previous
        self.assertAlmostEqual(float(np.sum(t+.034-delivered)),s,places=8)
        self.assertLess(abs(s),1.5)

    def test_circuit_area_model_improves_broadband_music_tracking(self):
        baud=460800;x,_=music_fixture('holdout');x=x[:FS//4]*.75
        n=int(len(x)/FS*baud)//10*10
        target=.45+.3*np.interp((np.arange(n)+.5)/baud,np.arange(len(x))/FS,x)
        sos=butter(4,[400,5000],fs=baud,btype='bandpass',output='sos')
        desired=sosfilt(sos,target)
        errors=[]
        for method in (bit_feedback_block,edge_feedback_block):
            data,_=method(target);on=1-frame_bits(data);previous=np.r_[0,on[:-1]]
            area=on*(1-.01838462745*(1-previous))+(1-on)*.10345002714*previous
            result=sosfilt(sos,area)
            errors.append(float(np.mean((result[int(.025*baud):]-desired[int(.025*baud):])**2)))
        self.assertLess(errors[1],errors[0]*.8)

if __name__=='__main__':unittest.main()

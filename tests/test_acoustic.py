import unittest
import numpy as np
from uart_buzzer.acoustic import measure,reference_gate,propose_input_gain_db


class AcousticTests(unittest.TestCase):
    def test_gain_does_not_change_when_headroom_is_sufficient(self):
        self.assertEqual(propose_input_gain_db(12,-18.8,False,-17.25,12,.75),12)

    def test_overload_reduces_gain_and_respects_device_limits(self):
        self.assertEqual(propose_input_gain_db(12,-.1,True,-17.25,12,.75),0)
        self.assertEqual(propose_input_gain_db(12,-2,False,-17.25,12,.75),1.5)
        self.assertEqual(propose_input_gain_db(-17.25,0,True,-17.25,12,.75),-17.25)

    def test_reference_total_stability_does_not_hide_spectral_drift(self):
        first=dict(total_window_RMS_dBFS=-36,fundamental_peak_dBFS=-46,second_harmonic_peak_dBFS=-34)
        last=dict(total_window_RMS_dBFS=-36.3,fundamental_peak_dBFS=-49,second_harmonic_peak_dBFS=-34.5)
        self.assertFalse(reference_gate(first,last)['stable_reference_gate'])

    def test_known_signal_and_strong_second_harmonic(self):
        rate=44100;t=np.arange(int(2*rate))/rate;rng=np.random.default_rng(25)
        x=rng.normal(0,1e-5,len(t));gate=(t>=1.03)&(t<1.13)
        x[gate]+=.003*np.sin(2*np.pi*1987.3*t[gate])+.01*np.sin(4*np.pi*1987.3*t[gate])
        r=measure(x,rate,dict(label='test',nominal_Hz=2000,write_start_record_s=1.))
        self.assertTrue(r['reliable_candidate'])
        self.assertAlmostEqual(r['estimated_fundamental_Hz'],1987.3,delta=.3)
        self.assertAlmostEqual(r['THD_2to5'],.01/.003,delta=.05)
        self.assertEqual(r['absolute_SPL_dB'],'unknown')

    def test_noise_is_not_ranked_as_tone(self):
        rng=np.random.default_rng(11);x=rng.normal(0,1e-4,88200)
        r=measure(x,44100,dict(label='noise',nominal_Hz=2000,write_start_record_s=1.))
        self.assertFalse(r['reliable_candidate'])

    def test_clipping_is_not_accepted(self):
        rate=44100;t=np.arange(88200)/rate
        x=np.zeros(len(t));g=(t>=1)&(t<1.13)
        x[g]=np.clip(2*np.sin(2*np.pi*2000*t[g]),-1,1)
        r=measure(x,rate,dict(label='clip',nominal_Hz=2000,write_start_record_s=1.))
        self.assertFalse(r['reliable_candidate']);self.assertGreater(r['clipped_fraction'],.1)

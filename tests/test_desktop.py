import hashlib
import json
from pathlib import Path
import threading
import unittest
import uuid
import numpy as np
from scipy.io import wavfile
from uart_buzzer.music import FS,KNOTS,music_fixture,music_loudness,encode_pcm
from uart_buzzer.desktop.transport import SerialWriter
from uart_buzzer.desktop.engine import LiveEncoder,convert,decode_segments,export_document,validate_profile
from uart_buzzer.desktop.devices import Playback,Microphone
from uart_buzzer.uart_codec import frame_bits
from uart_buzzer.uart_codec import bit_feedback_block

ROOT=Path(__file__).resolve().parents[1]


def profile():return json.loads((ROOT/'config/music_legacy_v1.json').read_text(encoding='utf-8'))


class DesktopEngineTests(unittest.TestCase):
    def test_boost_increases_music_rms_with_the_same_peak_ceiling(self):
        x,_=music_fixture('train');normal,_=music_loudness(x,target_rms=.32);boost,_=music_loudness(x,target_rms=.52)
        self.assertGreater(np.std(boost),np.std(normal)*1.08)
        self.assertLessEqual(max(abs(boost)),.95000001)
        for value in (float('nan'),float('inf'),.9,-1):
            with self.assertRaises(ValueError):music_loudness(x,target_rms=value)
            p=profile();p['loudness_target_rms']=value
            with self.assertRaises(ValueError):validate_profile(p)
    def test_music_limiter_does_not_turn_down_track_for_one_spike(self):
        t=np.arange(FS*3)/FS;x=.04*np.sin(2*np.pi*1300*t);x[FS]=1.
        y,info=music_loudness(x)
        self.assertGreater(np.std(y),.24);self.assertLessEqual(max(abs(y)),.95000001)
        self.assertGreater(info['limiter_fraction'],0)
        self.assertLess(np.mean(abs(y)>.94999),.0001)

    def test_continuous_music_preserves_carrier_through_internal_rests(self):
        x,_=music_fixture('train')
        segments,_=encode_pcm(x,continuous=True,soft_edges=True,bias=.45,amplitude=.35)
        self.assertEqual(len(segments),1);self.assertEqual(len(segments[0]['data']),6*46080)
        silence,_=encode_pcm(np.zeros(FS),continuous=True,soft_edges=True,bias=.45,amplitude=.35)
        self.assertEqual(silence,[])

    def test_async_transport_queues_two_owned_buffers_and_cancels_before_release(self):
        requests=[];stop=threading.Event()
        class Port:out_waiting=0
        class Request:
            def __init__(self,port,data):self.size=len(data);self.complete=False;self.released=False;requests.append(self)
            def done(self):return self.complete
            def cancel(self):self.complete=True
            def release(self):self.assert_complete();self.released=True
            def assert_complete(self):
                if not self.complete:raise AssertionError('Released pending I/O memory')
        def wait(_):
            next(r for r in requests if not r.complete).complete=True
        writer=SerialWriter(Port(),stop,wait,request_factory=Request)
        writer.write(b'first');writer.write(b'second')
        self.assertEqual(len(writer.pending),2);self.assertFalse(requests[0].complete)
        writer.write(b'third');self.assertEqual(writer.maximum_pending,2)
        self.assertTrue(requests[0].released);self.assertFalse(requests[-1].released)
        writer.close(cancel=True);self.assertTrue(all(r.complete and r.released for r in requests))

    def test_vectorized_feedback_matches_sequential_reference(self):
        target=np.random.default_rng(21).uniform(.13,.47,20000)
        actual,error=bit_feedback_block(target,.21)
        state=.21;expected=bytearray()
        for frame in target.reshape(-1,10):
            state+=frame[0]-1;value=0
            for bit in range(8):
                state+=frame[bit+1];on=state>=.5;state-=on;value|=(not on)<<bit
            state+=frame[9];expected.append(value)
        self.assertEqual(actual,bytes(expected));self.assertAlmostEqual(error,state,places=7)
    def test_profile_rejects_nonfinite_or_unsafe_density(self):
        p=profile();p['gains_db'][0]=float('nan')
        with self.assertRaises(ValueError):validate_profile(p)
        p=profile();p['amplitude']=.3
        with self.assertRaises(ValueError):validate_profile(p)

    def test_zero_volume_is_actual_no_transmission(self):
        encoder=LiveEncoder(profile());x=np.ones(FS,dtype=float)*.5
        data,pcm=encoder.block(x,0,512,volume=0)
        self.assertEqual(data,b'');self.assertTrue(np.all(pcm==0))

    def test_mute_has_one_fade_block_then_no_bytes(self):
        e=LiveEncoder(profile());x=np.ones(FS)*.5
        a,_=e.block(x,0,512,1);fade,_=e.block(x,512,512,0);mute,_=e.block(x,1024,512,0)
        self.assertTrue(a and fade);self.assertEqual(mute,b'')

    def test_live_volume_is_bounded_true_8n1_and_continuous_state(self):
        e=LiveEncoder(profile());x,_=music_fixture('train')
        a,_=e.block(x,0,512,.8);b,_=e.block(x,512,512,.25)
        bits=frame_bits(a+b).reshape(-1,10)
        self.assertEqual(len(bits),1024);self.assertTrue(np.all(bits[:,0]==0));self.assertTrue(np.all(bits[:,-1]==1))
        self.assertLess(abs(e.state),1.6)

    def test_volume_ramp_reduces_modulation_without_changing_framing(self):
        x=np.sin(2*np.pi*1200*np.arange(FS)/FS)*.7
        def output(level):
            e=LiveEncoder(profile());e.gain=level
            data,_=e.block(x,0,4608,level)
            d=(1-frame_bits(data).reshape(-1,10)).mean(axis=1)
            return np.std(d)
        self.assertLess(output(.25),output(1.)*.55)

    def test_conversion_and_export_include_real_rest_schedule(self):
        folder=ROOT/'results/gui_tests'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
        x,_=music_fixture('train');path=folder/'中文乐曲.wav';wavfile.write(path,FS,x.astype(np.float32))
        doc=convert(path,profile())
        self.assertEqual(len(doc.decoded),len(doc.processed));self.assertGreater(len(doc.segments),1)
        manifest=export_document(doc,folder/'export',.5);p=json.loads(manifest.read_text(encoding='utf-8'))
        self.assertGreater(p['encoding']['real_idle_s'],.1)
        for s in p['segments']:
            self.assertEqual(hashlib.sha256((manifest.parent/s['file']).read_bytes()).hexdigest(),s['sha256'])

    def test_import_cancellation_is_honored_inside_encoder(self):
        x,_=music_fixture('train')
        with self.assertRaises(InterruptedError):convert('demo',profile(),source=x,cancel_check=lambda:True)

    def test_fake_serial_stop_clears_queue_and_keeps_break_off(self):
        folder=ROOT/'results/gui_tests'/uuid.uuid4().hex[:12];folder.mkdir(parents=True)
        x,_=music_fixture('train');doc=convert('demo',profile(),source=x[:FS])
        class FakePort:
            baudrate=460800;out_waiting=0;is_open=True;break_condition=False;cleared=False
            def write(self,data):play.stop_event.set();return len(data)
            def reset_output_buffer(self):self.cleared=True
        class Link:
            connected=True;port=FakePort()
            def interrupt(self):pass
        play=Playback(Link(),doc,Microphone(),folder,record=False)
        r=play.run()
        self.assertFalse(r['completed']);self.assertTrue(play.link.port.cleared);self.assertFalse(play.link.port.break_condition)


if __name__=='__main__':unittest.main()

"""Explicit I/O lifecycle is exercised with synthetic microphone/serial doubles."""
import json
import numpy as np
import pytest
from uart_buzzer.desktop.characterization import analyze_tone,run


def test_known_command_f0_harmonics_clipping_and_capture_error():
    rate=16000;t=np.arange(6400)/rate;x=.2*np.sin(2*np.pi*500*t)
    result=analyze_tone(x,rate,500,.0002)
    assert result['valid'] and abs(result['actual_fundamental_hz']-500)<4
    for samples,status in [(1.2*np.sin(2*np.pi*500*t),[]),(.2*np.sin(2*np.pi*2000*t),[]),(x,['input overflow'])]:
        bad=analyze_tone(samples,rate,500,.0002,status)
        assert not bad['valid'] and bad['actual_fundamental_hz'] is None


class FakeIO:
    closed=False
    def __init__(self,cfg):self.rate=cfg['rate']
    def __enter__(self):return self
    def __exit__(self,*args):type(self).closed=True
    def background(self,seconds,cancel):return np.random.default_rng(2).normal(0,.0001,round(seconds*self.rate))
    def note(self,hz,volume,duration,gap,cancel):
        t=np.arange(round(duration*self.rate))/self.rate
        return np.r_[.2*volume*np.sin(2*np.pi*hz*t),np.zeros(round(gap*self.rate))],dict(statuses=[])


def config():
    from uart_buzzer.score.uart_renderer import VERSION
    return dict(name='SIMULATED test profile',port='FAKE',microphone=0,rate=16000,midi_min=60,midi_max=61,volume=.8,duration_s=.35,baud=460800,bias=.45,amplitude=.35,renderer=VERSION)


def test_sweep_pending_profile_evidence_and_cancel_cleanup(tmp_path):
    FakeIO.closed=False;folder=tmp_path/'complete';profile=run(config(),folder,io_factory=FakeIO)
    assert FakeIO.closed and len(profile['notes'])==2 and (folder/'pending_profile.json').is_file()
    proof=json.loads((folder/'evidence.json').read_text(encoding='utf-8'))
    assert proof['complete'] and not proof['physical_measurement'] and all(r['valid'] for r in proof['records'])
    values=[];folder=tmp_path/'cancelled';FakeIO.closed=False
    with pytest.raises(InterruptedError):run(config(),folder,cancel_check=lambda:bool(values),progress=values.append,io_factory=FakeIO)
    assert FakeIO.closed and json.loads((folder/'evidence.json').read_text(encoding='utf-8'))['cancelled']
    assert not (folder/'pending_profile.json').exists()

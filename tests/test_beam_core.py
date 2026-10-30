"""Original pure model/kernel tests, independent of desktop integration changes."""
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from uart_buzzer.beam_model import load_model,finite_gram
from uart_buzzer._beam_kernels import trajectory,terminal_linear,COMPILED
from uart_buzzer.uart_codec import ON
FIT=ROOT/'desktop_assets/impedance_fit.json'

def test_all_256_cached_frame_actions_match_physical_bit_steps():
    m=load_model(FIT,460800);rng=np.random.default_rng(3)
    for byte in range(256):
        initial=rng.normal(size=6)*.1;x=initial.copy();values=[]
        for bit in ON[byte]:values.append((m.half_F@x+m.half_g*bit)[0]);x=m.bit_F@x+m.bit_g*bit
        np.testing.assert_allclose(x,m.F@initial+m.g[byte],atol=2e-14)
        np.testing.assert_allclose(values,m.H@initial+m.q[byte],atol=2e-14)
    assert np.all(ON[:,0]==1) and np.all(ON[:,9]==0)

def test_gram_doubling_and_state_carry_match_direct_simulation():
    m=load_model(FIT,460800);x=np.arange(6)*.04;current=x.copy();total=0.
    for _ in range(135):total+=np.sum((m.H@current)**2);current=m.F@current
    assert abs(total-x@finite_gram(m.F,m.H,135)@x)<1e-11
    data=np.random.default_rng(2).integers(0,256,531,dtype=np.uint8)
    y,s=trajectory(data,m.F,m.g,m.H,m.q,x)
    a,sa=trajectory(data[:131],m.F,m.g,m.H,m.q,x);b,sb=trajectory(data[131:],m.F,m.g,m.H,m.q,sa[-1])
    np.testing.assert_allclose(y,np.r_[a,b],atol=1e-14);np.testing.assert_allclose(s[-1],sb[-1],atol=1e-14)

def test_compiled_kernel_matches_python_reference():
    if not COMPILED:pytest.skip('Numba optional')
    m=load_model(FIT,460800);data=np.arange(32,dtype=np.uint8)
    a=trajectory(data,m.F,m.g,m.H,m.q,np.zeros(6));b=trajectory.py_func(data,m.F,m.g,m.H,m.q,np.zeros(6))
    np.testing.assert_allclose(a[0],b[0],atol=1e-13)

def test_quadratic_terminal_is_exact_baseline_rollout_difference():
    m=load_model(FIT,460800);rng=np.random.default_rng(81);base=rng.integers(0,256,100,dtype=np.uint8)
    by,bs=trajectory(base,m.F,m.g,m.H,m.q,np.zeros(6));target=rng.uniform(.2,.6,(100,10))
    rr=terminal_linear(by-target,m.F,m.H,50);gram=finite_gram(m.F,m.H,50);k=12;delta=rng.normal(size=6)*.05
    actual,_=trajectory(base[k:k+50],m.F,m.g,m.H,m.q,bs[k]+delta)
    difference=np.sum((actual-target[k:k+50])**2)-np.sum((by[k:k+50]-target[k:k+50])**2)
    assert abs(difference-(delta@gram@delta+2*delta@rr[k]))<1e-10

"""S140 integration gates. Run only in the App validation resource window."""
import copy
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from uart_buzzer import _beam_kernels as K
from uart_buzzer import beam as B

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('version,helper,compiled,reason',[
    ('0.62.1',lambda:None,True,None),
    ('0.63.0',lambda:None,True,'unsupported_numba_version'),
    ('0.62.1',None,True,'private_selector_unavailable'),
    (None,None,False,'numba_unavailable'),
])
def test_private_backend_support_policy(version,helper,compiled,reason):
    enabled,actual=K._scratch_support(version,helper,compiled)
    assert enabled is (reason is None) and actual==reason


def test_compilation_failure_uses_original_without_changing_inputs():
    class CompileFailure(Exception):pass
    values=np.arange(5,dtype=np.float64);before=values.copy();answer=object()
    calls=[]
    def original(*args):calls.append(args);return answer
    with patch.object(K,'_scratch_enabled',True),patch.object(K,'_scratch_reason',None),\
         patch.object(K,'_scratch_errors',(CompileFailure,)),\
         patch.object(K,'beam_chunk_topw',side_effect=CompileFailure()),\
         patch.object(K,'beam_chunk_original',side_effect=original):
        assert K.beam_chunk(values) is answer
        assert K.backend_info()['fallback_reason']=='scratch_compilation_failed:CompileFailure'
        assert K.beam_chunk(values) is answer and len(calls)==2
    np.testing.assert_array_equal(values,before)


@pytest.mark.parametrize('bad',[np.nan,np.inf,1e200])
def test_nonfinite_and_extreme_diagnostics_keep_original_route(bad):
    args=[None]*23;args[0]=np.full((2,10),.4);args[10]=np.array([bad,0.,0.,0.,0.,0.])
    expected=object()
    with patch.object(K,'_scratch_enabled',True),\
         patch.object(K,'beam_chunk_topw',side_effect=AssertionError('unsupported numeric diagnostic')) ,\
         patch.object(K,'beam_chunk_original',return_value=expected):
        assert K.beam_chunk(*args) is expected


@pytest.mark.parametrize('baud',[460800,921600])
@pytest.mark.parametrize('mode',['beam_fast','beam_quality'])
def test_public_original_and_prepared_search_bytes_states_and_counts_equal(baud,mode):
    if not K._scratch_support(K._numba_version,K._selector,K.COMPILED)[0]:pytest.skip('private backend unavailable')
    m=B.load_model(ROOT/'desktop_assets/impedance_fit.json',baud)
    settings=B.settings_for(mode)
    rng=np.random.default_rng(13);base=rng.integers(0,256,35,dtype=np.uint8)
    xb=rng.normal(0,.03,6);xc=rng.normal(0,.03,6)
    target=rng.uniform(.2,.6,(len(base),10))
    by,bs=K.trajectory(base,m.F,m.g,m.H,m.q,xb)
    with patch.object(B,'beam_chunk',K.beam_chunk_original):
        original=B.search_block(target,base.tobytes(),m,settings,xc,xb,None,29)
    with patch.object(K,'_scratch_enabled',True),patch.object(K,'_scratch_reason',None):
        public=B.search_block(target,base.tobytes(),m,settings,xc,xb,None,29)
        prepared=B._search_prepared(target,base,m,settings,xc,by,bs,None,29,B._model_terms(m))
    for value in (public,prepared):
        assert value[0]==original[0] and value[2]==original[2]
        np.testing.assert_array_equal(value[1],original[1])
    with patch.object(K,'_scratch_enabled',False):
        fallback=B.search_block(target,base.tobytes(),m,settings,xc,xb,None,29)
    assert fallback[0]==original[0] and fallback[2]==original[2]
    np.testing.assert_array_equal(fallback[1],original[1])


def test_model_terms_are_document_local_and_cannot_be_mutated():
    m=B.load_model(ROOT/'desktop_assets/impedance_fit.json',460800)
    terms=B._model_terms(m)
    for actual,expected in zip(terms,(np.sum(m.q**2,axis=1),m.q@m.H,m.H.T@m.H)):
        np.testing.assert_array_equal(actual,expected)
        assert not actual.flags.writeable
        with pytest.raises(ValueError):actual.setflags(write=True)


@pytest.mark.parametrize('reason',['unsupported_numba_version','private_selector_unavailable','numba_unavailable'])
def test_unsupported_missing_or_noncompiled_backend_returns_original_bytes(reason):
    m=B.load_model(ROOT/'desktop_assets/impedance_fit.json',460800)
    settings=B.settings_for('beam_fast');rng=np.random.default_rng(19)
    base=rng.integers(0,256,9,dtype=np.uint8).tobytes();target=rng.uniform(.2,.6,(9,10))
    initial=np.zeros(6)
    original=K.beam_chunk_original
    with patch.object(B,'beam_chunk',original):expected=B.search_block(target,base,m,settings,initial)
    # With no Numba the original decorator is an identity function.
    fallback=original.py_func if reason=='numba_unavailable' and hasattr(original,'py_func') else original
    with patch.object(K,'_scratch_enabled',False),patch.object(K,'_scratch_reason',reason),\
         patch.object(K,'beam_chunk_original',fallback):
        actual=B.search_block(target,base,m,settings,initial)
    assert actual[0]==expected[0] and actual[2]==expected[2]
    np.testing.assert_array_equal(actual[1],expected[1])


def test_selector_build_attestation_and_complete_numba_notices(tmp_path):
    spec=importlib.util.spec_from_file_location('s140_builder',ROOT/'scripts/build_calibrated_candidate.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    a=builder.selector_attestation()
    assert a==dict(numba_version=K.SELECTOR_NUMBA_VERSION,arraymath_sha256=K.SELECTOR_SOURCE_SHA256)
    with patch.object(builder,'SELECTOR_SOURCE_SHA256','0'*64):
        with pytest.raises(RuntimeError,match='Unverified'):builder.selector_attestation()
    import shutil
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    shutil.copytree(ROOT/'desktop_assets',snapshot/'desktop_assets')
    shutil.copytree(ROOT/'src',snapshot/'src',ignore=shutil.ignore_patterns('__pycache__'))
    builder.prepare_assets(snapshot,ROOT)
    license_file=snapshot/'desktop_assets/licenses/numba_LICENSES.third-party'
    assert license_file.is_file() and 'appdirs' in license_file.read_text(encoding='utf-8')


def test_selector_attestation_participates_in_actual_source_fingerprint():
    from uart_buzzer.desktop import cache
    assert cache.PROCESSOR['numba_selector_dependency']['numba_version']==K.SELECTOR_NUMBA_VERSION
    assert cache.PROCESSOR['numba_selector_dependency']['arraymath_sha256']==K.SELECTOR_SOURCE_SHA256

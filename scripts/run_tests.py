"""Run all offline tests including SPICE; requires an ngspice installation."""
import os,sys,tempfile,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'tmp/numba_test_cache'))
temp=ROOT/'tmp'/('offline-tests-'+uuid.uuid4().hex)
temp.mkdir(parents=True)
tempfile.tempdir=str(temp)
import pytest
raise SystemExit(pytest.main(['tests','-q','--basetemp',str(temp/'pytest')]))

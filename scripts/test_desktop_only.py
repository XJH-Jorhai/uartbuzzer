"""Run deterministic desktop/DSP tests without hardware or ngspice."""
import os,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for folder in ('tmp/plot_deps','tmp/audio_deps','tmp/gui_deps','tmp/sim_deps','src','tests'):
    sys.path.insert(0,str(ROOT/folder))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib_cache'))
names=['test_acoustic','test_conversion_cache','test_desktop','test_edge_codec','test_library','test_music','test_nonlinear',
       'test_study.UARTTests','test_study.NetworkTests','test_study.FittingTests']
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(names))
raise SystemExit(not result.wasSuccessful())

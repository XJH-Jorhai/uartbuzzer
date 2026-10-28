"""Run meaningful unit and saved-SPICE integration checks without touching hardware."""
import os
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT/'tmp/plot_deps',ROOT/'tmp/audio_deps',ROOT/'tmp/gui_deps',ROOT/'tmp/sim_deps',ROOT/'src'):
    if p.exists():sys.path.insert(0,str(p))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib_cache'))
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())

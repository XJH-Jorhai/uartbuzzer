"""Source launcher. Frozen releases include their own Python/Qt/runtime/models."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
if not getattr(sys,'frozen',False):
    for folder in ('tmp/plot_deps','tmp/audio_deps','tmp/gui_deps','tmp/sim_deps','src'):
        p=ROOT/folder
        if p.exists():sys.path.insert(0,str(p))
os.environ.setdefault('PYQTGRAPH_QT_LIB','PySide6')
from uart_buzzer.desktop.app import main

if __name__=='__main__':main()

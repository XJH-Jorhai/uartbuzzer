"""One-command offline study. Does not open serial ports or control instruments."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for folder in (ROOT/'tmp/plot_deps', ROOT/'tmp/sim_deps', ROOT/'src'):
    if folder.exists():
        sys.path.insert(0, str(folder))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'tmp/matplotlib_cache'))
from uart_buzzer.study import main

if __name__ == '__main__':
    main()

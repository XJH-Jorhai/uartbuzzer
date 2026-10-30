"""Independent trapezoid versus Gear check for stored UART waveform."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import run
import numpy as np
from uart_buzzer.spice import Engine,VECTORS
from uart_buzzer.metrics import average
from uart_buzzer.io_data import dump

out=Path(sys.argv[1]);name='pmos_frame_aware_sigma_delta_921600_2000'
deck=(out/'spice'/(name+'.cir')).read_text()
engine=Engine()
a=engine.run(deck.replace('method=trap','method=gear maxord=2'),out/'spice/gear_method_check.cir',VECTORS)
old=np.load(out/'spice'/(name+'.npz'))
tt=np.arange(.016,.032,1e-7)
oldi=np.interp(tt,old['time'],old['coil']);newi=np.interp(tt,a['time'],a['coil'])
def rms(x):return float(np.sqrt(np.mean(x*x)))
report=dict(coil_rms_difference_A=rms(oldi-newi),coil_rms_gear_A=rms(newi),coil_rms_trap_A=rms(oldi),
            diode_rms_gear_A=np.sqrt(average(a['time'],a['diode']**2)),
            diode_rms_trap_A=np.sqrt(average(old['time'],old['diode']**2)),
            diode_peak_gear_A=float(a['diode'].max()),diode_peak_trap_A=float(old['diode'].max()))
np.savez_compressed(out/'spice/gear_method_check.npz',**a)
dump(out/'integration_method_check.json',report);print(report)

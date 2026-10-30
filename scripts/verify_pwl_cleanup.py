"""Check cleaned PWL corners against saved run; original warnings are preserved."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import run
import json
import numpy as np
from uart_buzzer.spice import Engine,simulate
from uart_buzzer.uart_codec import tone_target,encode,events
from uart_buzzer.io_data import dump

folder=Path(sys.argv[1])
fit=json.loads((folder/'fit/fit.json').read_text(encoding='utf-8'))
p=np.array(fit['selected_parameters']);model=fit['selected']
b,_,_=encode(tone_target(2000,921600,.032),'frame_aware_sigma_delta')
w,dur=events(b,921600)
engine=Engine()
path=folder/'spice/clean_pwl_verification.cir'
a=simulate(engine,p,model,w,dur,path,{'topology':'pmos','rg':220},1/921600/12,False)
with np.load(folder/'spice/pmos_frame_aware_sigma_delta_921600_2000.npz') as old:
    t=np.arange(.016,dur-.0005,1e-7)
    diff=np.interp(t,a['time'],a['coil'])-np.interp(t,old['time'],old['coil'])
report=dict(max_coil_difference_A=float(abs(diff).max()),rms_coil_difference_A=float(np.sqrt(np.mean(diff**2))),
            new_PWL_strictly_increasing=bool(np.all(np.diff(w[:,0])>0)),
            warning_removed='non-increasing PWL' not in path.with_suffix('.log').read_text(encoding='utf-8'),
            note='Original run had redundant same-voltage initial PWL corner; waveform equivalence verified')
dump(folder/'pwl_cleanup_verification.json',report)
assert report['max_coil_difference_A']<1e-5 and report['warning_removed']
print(report)

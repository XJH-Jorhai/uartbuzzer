"""Bundle portable model data and generate a code-drawn application icon."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
for f in ('tmp/plot_deps','tmp/gui_deps','tmp/sim_deps'):
    sys.path.insert(0,str(ROOT/f))
import json,hashlib

out=ROOT/'desktop_assets';out.mkdir(exist_ok=True)
processor=[ROOT/'src/uart_buzzer'/name for name in ('music.py','nonlinear.py','uart_codec.py','impedance.py','desktop/engine.py')]
(out/'dsp_fingerprint.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in processor},indent=2),encoding='utf-8')
source_profile=ROOT/'config/music_active.json'
if not source_profile.exists():source_profile=out/'music_profile.json'
p=json.loads(source_profile.read_text(encoding='utf-8'))
keep=('name','knots_Hz','gains_db','baud','encoder','bias','amplitude','band_Hz','clock_scale_estimate','status',
      'conditioning','continuous_carrier','soft_carrier_edges','evidence','loudness_target_rms',
      'available_predistortion','predistortion','predistortion_default_enabled','reset_predistortion_on_import','edge_model')
profile={k:p[k] for k in keep if k in p}
profile.setdefault('evidence','Polyphonic music microphone relative validation; not SPL calibration')
(out/'music_profile.json').write_text(json.dumps(profile,indent=2,ensure_ascii=False),encoding='utf-8')
source_fit=ROOT/'results/study/20261028_221317/fit/fit.json'
if not source_fit.exists():source_fit=out/'impedance_fit.json'
fit=json.loads(source_fit.read_text(encoding='utf-8'))
(out/'impedance_fit.json').write_text(json.dumps({k:fit[k] for k in ('selected','selected_parameters','seed','scope')},indent=2),encoding='utf-8')
if not (out/'app.ico').exists():
    from PIL import Image,ImageDraw
    image=Image.new('RGBA',(256,256),(0,0,0,0));d=ImageDraw.Draw(image)
    d.rounded_rectangle((5,5,251,251),radius=52,fill='#20272f',outline='#34546a',width=4)
    d.rounded_rectangle((37,60,67,196),radius=10,fill='#3688b8')
    points=[(52,127),(85,127),(105,83),(129,177),(151,57),(174,149),(192,127),(218,127)]
    d.line(points,fill='#6bd6f3',width=13,joint='curve')
    for x,y in points:d.ellipse((x-5,y-5,x+5,y+5),fill='#6bd6f3')
    image.save(out/'app.ico',sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
(out/'NOTICE.txt').write_text('UART Audio Studio 1.3.1\nBuilt with Python, NumPy, SciPy, PySide6/Qt, pyqtgraph, pyserial and sounddevice.\nQt/PySide6: LGPLv3 (or commercial); source modifications and third-party licenses included with the portable distribution.\nBundled measurement-derived model and relative music profile are specific to the tested hardware.\n',encoding='utf-8')
print(out)

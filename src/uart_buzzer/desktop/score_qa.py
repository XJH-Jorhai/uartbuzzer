"""Explicit isolated offline preview checks. Never opens serial or microphone."""
import hashlib,json
from pathlib import Path
import numpy as np


def fixture():
    pieces=[]
    for midi in (60,64,60,67):
        t=np.arange(4800)/16000;env=np.minimum(t/.008,1)*np.minimum((.3-t)/.008,1)
        pieces.extend([.25*env*np.sin(2*np.pi*440*2**((midi-69)/12)*t),np.zeros(1600)])
    return np.concatenate(pieces)


def run(root,report_path,assets):
    from .app import data_root
    from .cache import ConversionCache
    from ..score.pipeline import settings
    from ..score.contracts import flat_profile,score_ir,ProfileStore
    from ..score.export import export_json,export_midi
    root=Path(root).resolve();report_path=Path(report_path).resolve()
    protected=[data_root().resolve(),Path.home()/'AppData/Local/ZipLab/UART Audio Studio']
    if any(root.is_relative_to(p.resolve()) for p in protected) or not report_path.is_relative_to(root):raise ValueError('需要独立测试目录和其中的报告路径')
    if root.exists() and any(root.iterdir()):raise ValueError('使用新的空测试目录')
    root.mkdir(parents=True,exist_ok=True)
    base=json.loads((assets/'music_profile.json').read_text(encoding='utf-8'));x=fixture()
    profile=dict(base,encoder='discrete_score',prepared_volume=.8,score_config=settings({}))
    store=ProfileStore(root/'device_profiles');key=store.add(flat_profile('offline simulated range'))
    assert key in ProfileStore(store.folder).profiles
    cache=ConversionCache(root/'prepared');doc=cache.convert('original-synthetic.wav',profile,source=x)
    resumed=ConversionCache(cache.folder).convert('original-synthetic.wav',profile,source=x)
    score=score_ir(doc.metadata['score']);assert len(score['events'])==4
    assert [round(e['midi_pitch']) for e in score['events']]==[60,64,60,67]
    assert resumed.metadata['cache']['tier']=='disk' and resumed.metadata['score']==score
    assert [s['data'] for s in resumed.segments]==[s['data'] for s in doc.segments]
    export_json(score,root/'score.json');export_midi(score,root/'score.mid')
    report=dict(schema='score-product-selftest-v1',success=True,physical_IO=False,events=len(score['events']),palette=len(score['palette']),
        source='original synthetic notes only',disk_restart=True,device_profile_restart=True,JSON_MIDI_export=True,
        frontend_range=[48,96],duration_s=doc.duration,transport_plan=doc.metadata['transport_plan'],
        BIN_sha256=hashlib.sha256(b''.join(s['data'] for s in doc.segments)).hexdigest())
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8');return report

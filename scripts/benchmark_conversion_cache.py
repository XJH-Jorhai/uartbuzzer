"""Benchmark the supplied real WAV and prewarm the user's application cache."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));import desktop_app
from uart_buzzer.desktop.cache import ConversionCache,PROCESSOR
ap=argparse.ArgumentParser();ap.add_argument('--cache-dir',type=Path,required=True);args=ap.parse_args()
profile=json.loads((ROOT/'desktop_assets/music_profile.json').read_text(encoding='utf-8'))
wav=Path(r'E:\CloudMusic\未知歌手 - 01 音轨01.wav');rows=[]
cache=ConversionCache(args.cache_dir)
def trial(label,c,p):
    t=time.perf_counter();doc=c.convert(wav,p,fit_path=ROOT/'desktop_assets/impedance_fit.json')
    row=dict(label=label,wall_seconds=time.perf_counter()-t,duration_s=doc.duration,bytes=doc.metadata['total_bytes'],**doc.metadata['cache'])
    rows.append(row);print(json.dumps(row),flush=True)
    (ROOT/'results/conversion_cache_benchmark.json').write_text(json.dumps(dict(trials=rows,processor=PROCESSOR),indent=2),encoding='utf-8')
    return doc
a=trial('baseline_first_open',cache,profile)
b=trial('baseline_memory_hit',cache,profile)
c=trial('baseline_after_restart',ConversionCache(args.cache_dir),profile)
assert [s['data'] for s in a.segments]==[s['data'] for s in b.segments]==[s['data'] for s in c.segments]
pd=dict(profile,predistortion=profile['available_predistortion'])
trial('predistortion_first_open',cache,pd)
trial('predistortion_memory_hit',cache,pd)
trial('back_to_baseline',cache,profile)

"""Explicit isolated S140 benchmark, shared verbatim by reference and candidate."""
import hashlib
import json
from pathlib import Path
import time


def run(data_dir,config_path,assets,version):
    import numpy as np
    import psutil
    from .app import data_root
    from .cache import ConversionCache,PROCESSOR
    from . import cache as C
    from . import song_calibration as S
    from . import engine as E
    from .. import beam as B
    from .. import _beam_kernels as K

    root=Path(data_dir).resolve();config_path=Path(config_path).resolve()
    config=json.loads(config_path.read_text(encoding='utf-8'))
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protected=[data_root().resolve(),Path.home()/'AppData/Local/ZipLab/UART Audio Studio']
    if any(root.is_relative_to(p.resolve()) for p in protected):raise ValueError('Benchmark cannot use real user data')
    if config.get('schema')!='uart-beam-benchmark-v1' or config.get('fixture_only') is not True:
        raise ValueError('Explicit benchmark fixture config required')
    if root.exists() and any(root.iterdir()):raise ValueError('Use a new empty benchmark output')
    if config.get('repeats') not in (1,3):raise ValueError('Benchmark repeats must be explicitly 1 or 3')
    for name,digest in config['expected_DSP'].items():
        if PROCESSOR.get(name)!=digest:raise ValueError('Benchmark DSP mismatch: '+name)
    for case in config['cases']+[config.get('cold_case',config['cases'][0])]:
        if sha(case['path'])!=case['sha256']:raise ValueError('Benchmark source fixture changed')
    attestation=json.loads((assets/'benchmark_harness.json').read_text(encoding='utf-8'))
    if attestation.get('sha256')!=config['harness_sha256']:raise ValueError('Benchmark harness attestation mismatch')
    process=psutil.Process();process.cpu_affinity([int(config['CPU_core'])])
    profile=json.loads((assets/'music_profile.json').read_text(encoding='utf-8'))
    root.mkdir(parents=True,exist_ok=True)
    (root/'benchmark_fixture.json').write_text(json.dumps(dict(schema=config['schema'],fixture_only=True,config_sha256=sha(config_path))),encoding='utf-8')
    report=dict(schema=config['schema'],version=version,variant=config['variant'],frozen=bool(getattr(__import__('sys'),'frozen',False)),
                harness_sha256=attestation['sha256'],
                declared_harness_sha256=config['harness_sha256'],processor=PROCESSOR,CPU_core=process.cpu_affinity(),
                ready_perf_counter=time.perf_counter(),rows=[],phases_are_nested=True,
                timing_scope='ConversionCache.convert including stable WAV/hash/decode/frontend/full Beam/reconstruction/NPZ write; no Qt worker wait',
                physical_IO=False,success=False)
    phases={};counts={};restores=[]
    def wrap(module,name):
        original=getattr(module,name)
        def measured(*args,**kwargs):
            start=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:
                phases[name]=phases.get(name,0.)+time.perf_counter()-start
                counts[name]=counts.get(name,0)+1
        restores.append((module,name,original));setattr(module,name,measured)
    for name in ('filter_audio','encode_pcm','live_baseline_plan','decode_segments'):wrap(E,name)
    wrap(C,'read_audio');wrap(S,'file_hash')
    for name in ('load_model','reference','trajectory','terminal_linear','beam_chunk','encode_segments'):wrap(B,name)
    if hasattr(B,'_search_prepared'):wrap(B,'_search_prepared')
    else:wrap(B,'search_block')

    def save():
        temporary=root/'report.pending'
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        temporary.replace(root/'report.json')

    def physical_state(doc):
        # Independent bounded-memory replay outside the conversion timing.
        model=B.load_model(assets/'impedance_fit.json',doc.profile['baud'])
        state=np.zeros(6);cursor=0;h=hashlib.sha256();baud=model.baud
        for seg in doc.segments:
            start=round(seg['start_s']*baud/10)
            if start>cursor:
                state=np.linalg.matrix_power(model.F,start-cursor)@state
                h.update(state.tobytes())
            data=np.frombuffer(seg['data'],np.uint8)
            for offset in range(0,len(data),4096):
                _,states=K.trajectory(data[offset:offset+4096],model.F,model.g,model.H,model.q,state)
                h.update(states[1:].tobytes());state=states[-1].copy()
            cursor=start+len(data)
        end=int(doc.duration*baud/10)
        if end>cursor:state=np.linalg.matrix_power(model.F,end-cursor)@state;h.update(state.tobytes())
        return state.tolist(),h.hexdigest()

    counter=0
    def convert(case,baud,preset,stage,repeat,budget=None):
        nonlocal counter
        p=dict(profile,baud=baud,encoder=preset,beam_volume=.8)
        if budget is not None:p['beam_settings']={'budget_ratio':budget}
        phases.clear();counts.clear();counter+=1
        # A distinct empty store per conversion: no prepared cache lookup can hit.
        cache=ConversionCache(root/'prepared'/f'{counter:03d}')
        last_emit=[0.]
        def progress(value):
            now=time.perf_counter()
            if now-last_emit[0]>=1.:
                last_emit[0]=now
                (root/'progress.json').write_text(json.dumps(dict(case=case['id'],baud=baud,preset=preset,stage=stage,repeat=repeat,progress=value,perf_counter=now)),encoding='utf-8')
        start=time.perf_counter();cpu0=time.process_time()
        doc=cache.convert(case['path'],p,fit_path=assets/'impedance_fit.json',progress=progress)
        wall=time.perf_counter()-start;cpu=time.process_time()-cpu0
        measured_phases=dict(phases);measured_counts=dict(counts)
        b=doc.metadata['beam'];cache_info=doc.metadata['cache']
        active=sum(len(s['data']) for s in doc.segments)
        state,state_sha=physical_state(doc)
        row=dict(case=case['id'],source_sha256=case['sha256'],duration_s=doc.duration,baud=baud,preset=preset,
                 stage=stage,repeat=repeat,wall_s=wall,CPU_s=cpu,phase_s=measured_phases,phase_calls=measured_counts,
                 settings=b['parameters'],model_hash=b['model_hash'],prep_hash=b['prep_hash'],
                 source_PCM_sha256=doc.metadata['source_PCM_sha256'],
                 processed_PCM_sha256=hashlib.sha256(doc.processed.tobytes()).hexdigest(),
                 BIN_sha256=hashlib.sha256(b''.join(s['data'] for s in doc.segments)).hexdigest(),
                 schedule=[dict(start_s=s['start_s'],end_s=s['end_s'],bytes=len(s['data'])) for s in doc.segments],
                 final_state=state,physical_state_path_sha256=state_sha,
                 baseline_SSE=b['baseline_SSE'],raw_SSE=b['raw_candidate_SSE'],returned_SSE=b['returned_SSE'],
                 expanded=b['expanded_candidates'],optimized_frames=b['optimized_frames'],total_frames=b['total_frames'],
                 active_frames=active,budget_hit=b['budget_hit'],fallback=b['fallback'],fallback_reason=b['fallback_reason'],
                 kernel_backend=b.get('kernel_backend',dict(name='original_compiled' if K.COMPILED else 'original_python')),
                 native_compiled=bool(b['compiled']),prepared_cache_hit=bool(cache_info['hit']),
                 RSS_bytes=process.memory_info().rss)
        report['rows'].append(row);save()
        if row['prepared_cache_hit']:raise AssertionError('Prepared cache hit in encoding benchmark')
        if stage=='warm' and (row['budget_hit'] or row['fallback'] or row['optimized_frames']!=active or not row['native_compiled']):
            raise AssertionError('Incomplete or nonnative warm conversion cannot be a speed sample')
        return row

    try:
        first=config.get('cold_case',config['cases'][0])
        first_setting=config['settings'][0]
        cold=convert(first,*first_setting,'cold',0)
        # Only if needed, separate a sufficient-budget initialization from GUI-budget timings.
        if cold['budget_hit']:convert(first,*first_setting,'initialization_not_speed',0,300.)
        for case in config['cases']:
            for baud,preset in config['settings']:
                for repeat in range(config['repeats']):convert(case,baud,preset,'warm',repeat)
        report['success']=True;report['ended_perf_counter']=time.perf_counter();save()
    except BaseException as error:
        report['error']=repr(error);save();raise
    finally:
        for module,name,original in reversed(restores):setattr(module,name,original)

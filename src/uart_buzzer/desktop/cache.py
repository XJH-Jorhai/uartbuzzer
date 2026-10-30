"""Bounded RAM + persistent conversion cache. No pickle or executable payloads."""
from collections import OrderedDict
from dataclasses import replace
import copy
import importlib.metadata
from contextlib import contextmanager
import hashlib
import json
import os
import sys
from pathlib import Path
import threading
import tempfile
import time
import uuid
import zipfile
import numpy as np
import scipy
from .engine import AudioDocument,convert,validate_profile
from ..music import FS,read_audio

# Bump when conditioning, predistortion, codec or reconstruction semantics change.
SCHEMA='uart-conversion-20261029-v1'


def _selector_dependency():
    try:
        dist=importlib.metadata.distribution('numba')
        path=Path(dist.locate_file('numba/np/arraymath.py'))
        return dict(numba_version=dist.version,arraymath_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except (importlib.metadata.PackageNotFoundError,OSError):
        return dict(numba_version=None,arraymath_sha256=None)

def _processor_fingerprint():
    package=Path(__file__).resolve().parents[1]
    paths=[package/'music.py',package/'nonlinear.py',package/'uart_codec.py',package/'impedance.py',package/'desktop/engine.py',
           package/'beam.py',package/'beam_model.py',package/'_beam_kernels.py']
    if all(p.is_file() for p in paths):
        return dict({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    numba_selector_dependency=_selector_dependency())
    assets=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[3]))/'desktop_assets'
    return json.loads((assets/'dsp_fingerprint.json').read_text(encoding='utf-8'))

PROCESSOR=_processor_fingerprint()
DEFAULTS=dict(baud=460800,encoder='bit_feedback_uart_v1',bias=.3,amplitude=.18,
              conditioning='legacy',loudness_target_rms=.32,continuous_carrier=False,soft_carrier_edges=False)


class ConversionCache:
    def __init__(self,folder,memory_bytes=192*1024**2,disk_bytes=1024**3,max_entries=32):
        self.folder=Path(folder).resolve();self.memory_bytes=memory_bytes;self.disk_bytes=disk_bytes
        self.max_entries=max_entries;self.memory=OrderedDict();self.used=0;self.lock=threading.RLock()

    def key(self,path,profile,mode,fit_path,source,cancel_check,content_identity=None):
        h=hashlib.sha256()
        if content_identity is not None:
            source_digest,source_kind=content_identity
        elif source is None:
            with Path(path).open('rb') as f:
                while chunk:=f.read(1024*1024):
                    if cancel_check and cancel_check():raise InterruptedError('Conversion cancelled')
                    h.update(chunk)
            source_kind='wav'
        else:
            a=np.ascontiguousarray(source,dtype=np.float64)
            h.update(str(a.shape).encode());h.update(memoryview(a).cast('B'));source_kind='pcm64'
        if content_identity is None:source_digest=h.hexdigest()
        settings={k:profile.get(k,v) for k,v in DEFAULTS.items()}
        settings['gains_db']=profile['gains_db'] if mode=='music' else [0.]*13
        pd=profile.get('predistortion')
        if pd:settings['predistortion']={k:pd.get(k,v) for k,v in dict(kind=None,kernel2=None,strength=1.,sample_rate_Hz=FS).items()}
        if settings['encoder']=='edge_feedback_uart_v1':settings['edge_model']=profile.get('edge_model') or {}
        if settings['encoder'].startswith('beam_'):
            from ..beam_model import model_identity
            from ..beam import ENCODER_VERSION
            if fit_path is None:raise ValueError('Beam缓存需要固定M2模型路径')
            settings.update(beam_model_hash=model_identity(fit_path)[0],beam_version=ENCODER_VERSION,
                beam_settings=profile.get('beam_settings',{}),beam_volume=profile.get('beam_volume',1.),
                beam_baseline_encoder=profile.get('beam_baseline_encoder','bit_feedback_uart_v1'))
        if settings['encoder']=='discrete_score':
            from ..score.pipeline import VERSION,FRONTEND_VERSION,fingerprint,settings as score_settings
            from ..score.uart_renderer import VERSION as renderer_version
            settings.update(score_config=score_settings(profile.get('score_config')),score_volume=profile.get('prepared_volume',.8),
                score_pipeline=VERSION,score_frontend=FRONTEND_VERSION,score_modules=fingerprint(),score_renderer=renderer_version,
                score_backend='explicit_segmented_uart8n1_or_generic')
        if mode=='electrical':settings['fit_sha256']=hashlib.sha256(Path(fit_path).read_bytes()).hexdigest()
        if profile.get('_song_calibration'):
            settings['song_calibration']=profile['_song_calibration']
        record=dict(schema=SCHEMA,processor=PROCESSOR,numpy=np.__version__,scipy=scipy.__version__,source=source_digest,source_kind=source_kind,mode=mode,settings=settings)
        return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    @staticmethod
    def size(doc):
        return sum(a.nbytes for a in (doc.source,doc.processed,doc.decoded,doc.base_processed,doc.quadratic_correction,doc.beam_pcm) if a is not None)+sum(len(s['data']) for s in doc.segments)+sum(len(s['data']) for s in doc.baseline_segments or [])

    def remember(self,key,doc):
        size=self.size(doc)
        if size>self.memory_bytes:return
        for a in (doc.source,doc.processed,doc.decoded,doc.base_processed,doc.quadratic_correction,doc.beam_pcm):
            if a is not None:a.setflags(write=False)
        if key in self.memory:self.used-=self.size(self.memory.pop(key))
        self.memory[key]=replace(doc,profile=copy.deepcopy(doc.profile),metadata=copy.deepcopy(doc.metadata),segments=[dict(s) for s in doc.segments],
             baseline_segments=[dict(s) for s in doc.baseline_segments] if doc.baseline_segments is not None else None);self.used+=size
        while self.used>self.memory_bytes or len(self.memory)>self.max_entries:
            _,old=self.memory.popitem(last=False);self.used-=self.size(old)

    def save(self,key,doc):
        if self.size(doc)>self.disk_bytes:return False
        self.folder.mkdir(parents=True,exist_ok=True)
        path=self.folder/(key+'.npz');temporary=self.folder/(key+'.tmp-'+uuid.uuid4().hex)
        records=[];offset=0;parts=[]
        for s in doc.segments:
            records.append(dict(start_s=s['start_s'],end_s=s['end_s'],offset=offset,length=len(s['data'])))
            parts.append(s['data']);offset+=len(s['data'])
        manifest=dict(schema=SCHEMA,key=key,profile=doc.profile,metadata=doc.metadata,segments=records)
        arrays=dict(source=doc.source,processed=doc.processed,decoded=doc.decoded,
                    manifest=np.frombuffer(json.dumps(manifest,ensure_ascii=False,allow_nan=False).encode('utf-8'),np.uint8),
                    payload=np.frombuffer(b''.join(parts),np.uint8))
        if doc.base_processed is not None:arrays['base']=doc.base_processed;arrays['correction']=doc.quadratic_correction
        if doc.beam_pcm is not None:
            arrays['beam_pcm']=doc.beam_pcm
            arrays['baseline_payload']=np.frombuffer(b''.join(s['data'] for s in doc.baseline_segments),np.uint8)
        try:
            # Stored NPZ avoids spending seconds compressing/decompressing whole songs.
            with temporary.open('wb') as f:np.savez(f,**arrays)
            os.replace(temporary,path)
            self.prune()
            return True
        finally:
            if temporary.exists():temporary.unlink()

    def prune(self):
        owned=[p for p in self.folder.glob('*.npz') if p.is_file() and len(p.stem)==64 and all(c in '0123456789abcdef' for c in p.stem)]
        entries=sorted(owned,key=lambda p:p.stat().st_mtime_ns,reverse=True)
        total=0
        for i,p in enumerate(entries):
            total+=p.stat().st_size
            if i>=self.max_entries or total>self.disk_bytes:
                if p.resolve().parent==self.folder:p.unlink()

    def read(self,key):
        path=self.folder/(key+'.npz')
        if not path.exists():return None
        try:
            with zipfile.ZipFile(path) as archive:
                if sum(v.file_size for v in archive.infolist())>384*1024**2:raise ValueError('Oversize cache')
            with np.load(path,allow_pickle=False) as z:
                m=json.loads(z['manifest'].tobytes().decode('utf-8'))
                if m['schema']!=SCHEMA or m['key']!=key:raise ValueError('Stale cache')
                if not isinstance(m['metadata'],dict) or not isinstance(m['segments'],list):raise ValueError('Invalid manifest')
                arrays=[z[k] for k in ('source','processed','decoded')]
                n=len(arrays[0])
                if not FS*.1<=n<=FS*600:raise ValueError('Invalid cache duration')
                for a in arrays:
                    if a.dtype!=np.float32 or a.shape!=(n,) or not np.all(np.isfinite(a)):raise ValueError('Invalid cached PCM')
                base=z['base'] if 'base' in z else None;correction=z['correction'] if 'correction' in z else None
                for a in (base,correction):
                    if a is not None and (a.dtype!=np.float32 or a.shape!=(n,) or not np.all(np.isfinite(a))):raise ValueError('Invalid correction cache')
                if (base is None)!=(correction is None):raise ValueError('Incomplete correction cache')
                payload=z['payload']
                if payload.dtype!=np.uint8 or payload.ndim!=1:raise ValueError('Invalid cached UART payload')
                segments=[];cursor=0;last_end=0
                if len(payload)>np.ceil(n/FS*m['profile']['baud']/10)+2:raise ValueError('Oversize UART payload')
                for s in m['segments']:
                    if s['offset']!=cursor or s['length']<0 or cursor+s['length']>len(payload):raise ValueError('Invalid cache offsets')
                    if not 0<=s['start_s']<=s['end_s']<=n/FS+.001:raise ValueError('Invalid segment timing')
                    if s['start_s']<last_end-.00001 or abs((s['end_s']-s['start_s'])*m['profile']['baud']/10-s['length'])>1.1:raise ValueError('Invalid segment length')
                    segments.append(dict(start_s=s['start_s'],end_s=s['end_s'],data=payload[cursor:cursor+s['length']].tobytes()))
                    cursor+=s['length'];last_end=s['end_s']
                if cursor!=len(payload):raise ValueError('Incomplete UART payload')
                baseline_segments=None;beam_pcm=None
                if m['profile'].get('encoder','').startswith('beam_'):
                    beam_pcm=z['beam_pcm'];bp=z['baseline_payload']
                    if beam_pcm.dtype!=np.float64 or beam_pcm.shape!=(n,) or not np.all(np.isfinite(beam_pcm)):raise ValueError('Invalid Beam target PCM')
                    if bp.dtype!=np.uint8 or bp.shape!=payload.shape or not m['metadata'].get('beam'):raise ValueError('Incomplete Beam baseline')
                    baseline_segments=[];cursor=0
                    for seg in segments:
                        count=len(seg['data']);baseline_segments.append(dict(seg,data=bp[cursor:cursor+count].tobytes()));cursor+=count
                if m['profile'].get('encoder')=='discrete_score':
                    from ..score.contracts import score_ir
                    from ..score.model import digest
                    score=score_ir(m['metadata']['score']);plan=m['metadata']['transport_plan']
                    if plan.get('schema')!='finite-score-transport-plan-v1' or plan.get('score_hash')!=digest(score):raise ValueError('Invalid cached score transport')
                    expected=plan.get('segments',[])
                    if len(expected)!=len(segments):raise ValueError('Invalid cached score schedule')
                    for proof,segment in zip(expected,segments):
                        if proof['sha256']!=hashlib.sha256(segment['data']).hexdigest() or proof['start_s']!=segment['start_s'] or proof['end_s']!=segment['end_s']:raise ValueError('Invalid cached score payload')
                doc=AudioDocument('','',*arrays,segments,validate_profile(m['profile']),m['metadata'],base,correction,baseline_segments,beam_pcm)
            os.utime(path,None)
            return doc
        except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,OverflowError,EOFError,zipfile.BadZipFile):
            try:path.unlink()
            except OSError:pass
            return None

    @contextmanager
    def source_snapshot(self,path,source,cancel_check):
        from .song_calibration import file_hash
        def check():
            if cancel_check and cancel_check():raise InterruptedError('Conversion cancelled')
        if source is not None:
            pcm=np.array(source,dtype=np.float64,order='C',copy=True)
            h=hashlib.sha256();h.update(str(pcm.shape).encode());h.update(memoryview(pcm).cast('B'))
            yield lambda:pcm,(h.hexdigest(),'pcm64'),check
            return
        # Hash and decode the same private snapshot, even if the original path
        # is replaced or changes A -> B -> A between hash and decoding.
        with tempfile.SpooledTemporaryFile(max_size=8*1024**2,mode='w+b') as snapshot:
            h=hashlib.sha256()
            with Path(path).open('rb') as stream:
                while chunk:=stream.read(1024*1024):check();h.update(chunk);snapshot.write(chunk)
            identity=(h.hexdigest(),'wav');decoded=None
            def load():
                nonlocal decoded
                if decoded is None:snapshot.seek(0);decoded=read_audio(snapshot)[0]
                return decoded
            def fresh():
                check()
                if file_hash(path,cancel_check)!=identity[0]:raise ValueError('转换期间歌曲文件已改变，请重新加载')
            yield load,identity,fresh

    def convert(self,path,profile,mode='music',fit_path=None,progress=None,cancel_check=None,source=None,name=None):
        with self.source_snapshot(path,source,cancel_check) as (load,identity,fresh):
            return self._convert_snapshot(path,profile,mode,fit_path,progress,cancel_check,source,name,load,identity,fresh)

    def _convert_snapshot(self,path,profile,mode,fit_path,progress,cancel_check,source,name,load,identity,fresh):
        start=time.perf_counter();p=validate_profile(profile)
        if cancel_check and cancel_check():raise InterruptedError('Conversion cancelled')
        key=self.key(path,p,mode,fit_path,source,cancel_check,content_identity=identity)
        if progress:progress(.02)
        tier=None
        with self.lock:
            doc=self.memory.get(key)
            if doc is not None:self.memory.move_to_end(key);tier='memory'
            else:
                doc=self.read(key)
                if doc is not None:tier='disk'
            if doc is not None:
                migrate=doc.metadata.get('_legacy_migration_pending',False)
                actual_PCM=hashlib.sha256(np.asarray(doc.source,np.float32).tobytes()).hexdigest()
                proof=doc.metadata.get('source_commit',{})
                verified=(proof.get('version')=='stable-source-v1' and proof.get('source_sha256')==identity[0]
                          and proof.get('source_kind')==identity[1]
                          and proof.get('source_PCM_sha256')==doc.metadata.get('source_PCM_sha256')==actual_PCM)
                if not verified:
                    expected=hashlib.sha256(np.asarray(load(),np.float32).tobytes()).hexdigest()
                    if expected!=doc.metadata.get('source_PCM_sha256') or expected!=actual_PCM:
                        if key in self.memory:self.used-=self.size(self.memory.pop(key))
                        doc=None;tier=None
                    else:
                        doc=replace(doc,metadata=copy.deepcopy(doc.metadata))
                        doc.metadata['source_commit']=dict(version='stable-source-v1',source_sha256=identity[0],source_kind=identity[1],source_PCM_sha256=expected)
                fresh()
                if doc is not None:
                    doc.metadata.pop('_legacy_migration_pending',None)
                    self.remember(key,doc)
                    if not verified or migrate:
                        try:self.save(key,doc)
                        except OSError:pass
        if cancel_check and cancel_check():raise InterruptedError('Conversion cancelled')
        if doc is not None:
            # Cosmetic fields, file name and path belong to the current request.
            effective=copy.deepcopy(p)
            if mode!='music':effective['gains_db']=doc.profile['gains_db']
            if mode=='electrical':effective['name']='电气模型补偿'
            meta=copy.deepcopy(doc.metadata)
            meta['cache']=dict(hit=True,tier=tier,key=key,seconds=time.perf_counter()-start)
            if progress:progress(1.)
            return replace(doc,name=name or Path(path).name,path=str(path),profile=effective,metadata=meta,
                           segments=[dict(s) for s in doc.segments],baseline_segments=[dict(s) for s in doc.baseline_segments] if doc.baseline_segments is not None else None)
        doc=convert(path,p,mode,fit_path,progress,cancel_check,load(),name)
        fresh()  # No RAM or disk commit from a rejected source/cancelled job.
        doc.metadata['source_commit']=dict(version='stable-source-v1',source_sha256=identity[0],source_kind=identity[1],source_PCM_sha256=doc.metadata['source_PCM_sha256'])
        if cancel_check and cancel_check():raise InterruptedError('Conversion cancelled')
        if doc.metadata.get('beam',{}).get('budget_hit'):
            # A cold JIT or short budget must not permanently cache an unoptimized incumbent.
            doc.metadata['cache']=dict(hit=False,persisted=False,key=key,seconds=time.perf_counter()-start,
                error=None,skipped_reason='budget_fallback_is_retryable')
            return doc
        persisted=False;error=None
        try:
            with self.lock:self.remember(key,doc);persisted=self.save(key,doc)
        except (OSError,ValueError,TypeError) as e:error=str(e)
        doc.metadata['cache']=dict(hit=False,persisted=persisted,key=key,seconds=time.perf_counter()-start,error=error)
        if cancel_check and cancel_check():raise InterruptedError('Conversion cancelled')
        return doc


class LibraryAudioCache(ConversionCache):
    """Durable prepared song files. Bounded RAM, no automatic disk eviction."""
    def __init__(self,folder,legacy_folder=None,**kwargs):
        super().__init__(folder,disk_bytes=384*1024**2,**kwargs)
        self.legacy=ConversionCache(legacy_folder) if legacy_folder else None

    def prune(self):
        # These are library assets, rather than disposable LRU entries.
        pass

    def read(self,key):
        doc=super().read(key)
        if doc is None and self.legacy:
            doc=self.legacy.read(key)
            if doc is not None:
                # Migration is committed only after stable source validation.
                doc.metadata['_legacy_migration_pending']=True
        return doc

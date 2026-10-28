"""Bounded RAM + persistent conversion cache. No pickle or executable payloads."""
from collections import OrderedDict
from dataclasses import replace
import copy
import hashlib
import json
import os
import sys
from pathlib import Path
import threading
import time
import uuid
import zipfile
import numpy as np
import scipy
from .engine import AudioDocument,convert,validate_profile
from ..music import FS

# Bump when conditioning, predistortion, codec or reconstruction semantics change.
SCHEMA='uart-conversion-20261029-v1'
def _processor_fingerprint():
    package=Path(__file__).resolve().parents[1]
    paths=[package/'music.py',package/'nonlinear.py',package/'uart_codec.py',package/'impedance.py',package/'desktop/engine.py']
    if all(p.is_file() for p in paths):
        return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assets=Path(getattr(sys,'_MEIPASS',Path(__file__).resolve().parents[3]))/'desktop_assets'
    return json.loads((assets/'dsp_fingerprint.json').read_text(encoding='utf-8'))

PROCESSOR=_processor_fingerprint()
DEFAULTS=dict(baud=460800,encoder='bit_feedback_uart_v1',bias=.3,amplitude=.18,
              conditioning='legacy',loudness_target_rms=.32,continuous_carrier=False,soft_carrier_edges=False)


class ConversionCache:
    def __init__(self,folder,memory_bytes=192*1024**2,disk_bytes=1024**3,max_entries=32):
        self.folder=Path(folder).resolve();self.memory_bytes=memory_bytes;self.disk_bytes=disk_bytes
        self.max_entries=max_entries;self.memory=OrderedDict();self.used=0;self.lock=threading.RLock()

    def key(self,path,profile,mode,fit_path,source,cancel_check):
        h=hashlib.sha256()
        if source is None:
            with Path(path).open('rb') as f:
                while chunk:=f.read(1024*1024):
                    if cancel_check and cancel_check():raise InterruptedError('Conversion cancelled')
                    h.update(chunk)
            source_kind='wav'
        else:
            a=np.ascontiguousarray(source,dtype=np.float64)
            h.update(str(a.shape).encode());h.update(memoryview(a).cast('B'));source_kind='pcm64'
        settings={k:profile.get(k,v) for k,v in DEFAULTS.items()}
        settings['gains_db']=profile['gains_db'] if mode=='music' else [0.]*13
        pd=profile.get('predistortion')
        if pd:settings['predistortion']={k:pd.get(k,v) for k,v in dict(kind=None,kernel2=None,strength=1.,sample_rate_Hz=FS).items()}
        if settings['encoder']=='edge_feedback_uart_v1':settings['edge_model']=profile.get('edge_model') or {}
        if mode=='electrical':settings['fit_sha256']=hashlib.sha256(Path(fit_path).read_bytes()).hexdigest()
        record=dict(schema=SCHEMA,processor=PROCESSOR,numpy=np.__version__,scipy=scipy.__version__,source=h.hexdigest(),source_kind=source_kind,mode=mode,settings=settings)
        return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

    @staticmethod
    def size(doc):
        return sum(a.nbytes for a in (doc.source,doc.processed,doc.decoded,doc.base_processed,doc.quadratic_correction) if a is not None)+sum(len(s['data']) for s in doc.segments)

    def remember(self,key,doc):
        size=self.size(doc)
        if size>self.memory_bytes:return
        for a in (doc.source,doc.processed,doc.decoded,doc.base_processed,doc.quadratic_correction):
            if a is not None:a.setflags(write=False)
        if key in self.memory:self.used-=self.size(self.memory.pop(key))
        self.memory[key]=replace(doc,profile=copy.deepcopy(doc.profile),metadata=copy.deepcopy(doc.metadata),segments=[dict(s) for s in doc.segments]);self.used+=size
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
                doc=AudioDocument('','',*arrays,segments,validate_profile(m['profile']),m['metadata'],base,correction)
            os.utime(path,None)
            return doc
        except (OSError,ValueError,KeyError,TypeError,IndexError,AttributeError,OverflowError,EOFError,zipfile.BadZipFile):
            try:path.unlink()
            except OSError:pass
            return None

    def convert(self,path,profile,mode='music',fit_path=None,progress=None,cancel_check=None,source=None,name=None):
        start=time.perf_counter();p=validate_profile(profile)
        if cancel_check and cancel_check():raise InterruptedError('Conversion cancelled')
        key=self.key(path,p,mode,fit_path,source,cancel_check)
        if progress:progress(.02)
        tier=None
        with self.lock:
            doc=self.memory.get(key)
            if doc is not None:self.memory.move_to_end(key);tier='memory'
            else:
                doc=self.read(key)
                if doc is not None:self.remember(key,doc);tier='disk'
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
                           segments=[dict(s) for s in doc.segments])
        doc=convert(path,p,mode,fit_path,progress,cancel_check,source,name)
        if cancel_check and cancel_check():raise InterruptedError('Conversion cancelled')
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
                try:self.save(key,doc)
                except OSError:pass
        return doc

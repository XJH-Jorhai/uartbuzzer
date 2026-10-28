"""Persistent local WAV references and named playlists. Never deletes audio files."""
import copy,hashlib,json,os,random,struct,uuid
from pathlib import Path

def canonical(path):return os.path.normcase(str(Path(path).resolve()))
def track_id(path):return hashlib.sha256(canonical(path).encode('utf-8')).hexdigest()[:24]

def probe_track(path):
    p=Path(path).resolve();result=dict(id=track_id(p),path=str(p),title=p.stem,duration_s=None,error=None,cache_keys=[])
    try:
        stat=p.stat();result.update(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
        if p.suffix.lower()!='.wav':raise ValueError('仅支持 WAV')
        with p.open('rb') as f:
            header=f.read(12)
            if len(header)!=12 or header[:4] not in (b'RIFF',b'RIFX') or header[8:]!=b'WAVE':raise ValueError('WAV 文件头无效')
            order='<' if header[:4]==b'RIFF' else '>';fmt=None;data_size=None
            while f.tell()+8<=stat.st_size:
                kind=f.read(4);length=struct.unpack(order+'I',f.read(4))[0];start=f.tell()
                if kind==b'fmt ':
                    raw=f.read(min(length,64))
                    if len(raw)<16:raise ValueError('WAV 格式块不完整')
                    code,channels,rate,_,align,bits=struct.unpack(order+'HHIIHH',raw[:16])
                    if code==65534 and len(raw)>=40:code=struct.unpack(order+'H',raw[24:26])[0]
                    if code not in (1,3) or not 1<=channels<=32 or not 1000<=rate<=768000 or not align or not bits:raise ValueError('不支持的 WAV 格式')
                    fmt=(channels,rate,align,bits)
                elif kind==b'data':data_size=min(length,max(0,stat.st_size-start))
                if fmt and data_size is not None:break
                f.seek(start+length+(length&1))
            if not fmt or data_size is None:raise ValueError('缺少 WAV 音频数据')
            channels,rate,align,bits=fmt
            result.update(duration_s=(data_size//align)/rate,sample_rate_Hz=rate,channels=channels,bits=bits)
    except (OSError,ValueError,struct.error) as e:result['error']=str(e)
    return result

def inspect_paths(paths,limit=10000):
    files=[];seen=set()
    for item in paths:
        p=Path(item)
        candidates=p.rglob('*') if p.is_dir() else [p]
        for f in candidates:
            if f.suffix.lower()!='.wav' or f.is_dir():continue
            key=canonical(f)
            if key in seen:continue
            seen.add(key);files.append(probe_track(f))
            if len(files)>=limit:return files
    return files

def move_items(ids,selected,direction):
    result=list(ids);selected=set(selected)
    indices=range(1,len(result)) if direction<0 else range(len(result)-2,-1,-1)
    for i in indices:
        j=i-1 if direction<0 else i+1
        if result[i] in selected and result[j] not in selected:result[i],result[j]=result[j],result[i]
    return result

def next_track(ids,current,mode='sequential',direction=1,automatic=False,rng=None):
    ids=list(ids)
    if automatic and mode=='single':return None
    if not ids:return None
    if automatic and mode=='repeat_one' and current in ids:return current
    if mode=='shuffle' and len(ids)>1:return (rng or random).choice([i for i in ids if i!=current])
    if current not in ids:return ids[0] if direction>0 else ids[-1]
    index=ids.index(current)+direction
    if automatic and mode=='sequential' and not 0<=index<len(ids):return None
    return ids[index%len(ids)]

class LibraryStore:
    def __init__(self,path):
        self.path=Path(path);self.warning=None
        self.data=dict(version=1,tracks={},order=[],playlists=[dict(id='default',name='默认歌单',tracks=[])],active='all',current=None)
        self.write_blocked=False;errors=[]
        for candidate in (self.path,self.path.with_suffix('.json.bak')):
            if not candidate.exists():continue
            try:
                if candidate.stat().st_size>16*1024**2:raise ValueError('文件库过大')
                data=json.loads(candidate.read_text(encoding='utf-8'))
                if data['version']!=1 or not isinstance(data['tracks'],dict) or len(data['tracks'])>10000:raise ValueError('文件库格式无效')
                tracks={k:v for k,v in data['tracks'].items() if isinstance(v,dict) and isinstance(v.get('path'),str) and Path(v['path']).suffix.lower()=='.wav'}
                lists=[]
                for p in data['playlists'][:128]:
                    if isinstance(p.get('id'),str) and isinstance(p.get('name'),str):lists.append(dict(id=p['id'],name=p['name'],tracks=list(dict.fromkeys(i for i in p['tracks'] if i in tracks))))
                if not any(p['id']=='default' for p in lists):lists.insert(0,dict(id='default',name='默认歌单',tracks=[]))
                order=list(dict.fromkeys(i for i in data.get('order',[]) if i in tracks));order.extend(i for i in tracks if i not in order)
                self.data=dict(version=1,tracks=tracks,order=order,playlists=lists,
                               active=data.get('active','all'),current=data.get('current'))
                if self.data['active'] not in ['all']+[p['id'] for p in lists]:self.data['active']='all'
                if candidate!=self.path:self.warning='主歌单读取失败，已恢复备份；原文件保留到下次成功保存。'
                break
            except (OSError,ValueError,KeyError,TypeError,AttributeError) as e:errors.append(str(e))
        else:
            if errors:self.warning='歌单及备份读取失败，已禁止覆盖。请打开数据目录检查：'+'; '.join(errors);self.write_blocked=True

    def commit(self,data):
        if self.write_blocked:raise ValueError(self.warning)
        backup=self.data if self.path.exists() and self.data['tracks'] else data
        self._atomic_write(self.path.with_suffix('.json.bak'),backup)
        self._atomic_write(self.path,data);self.data=data

    def _atomic_write(self,path,data):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
        try:
            with temporary.open('w',encoding='utf-8') as f:
                json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
        finally:
            if temporary.exists():temporary.unlink()

    def ids(self,playlist=None):
        playlist=playlist or self.data['active']
        if playlist=='all':return list(self.data['order'])
        return list(next((p['tracks'] for p in self.data['playlists'] if p['id']==playlist),[]))

    def add(self,tracks,playlist=None):
        data=copy.deepcopy(self.data);ids=[];playlist=playlist or data['active']
        target=next((p for p in data['playlists'] if p['id']==('default' if playlist=='all' else playlist)),data['playlists'][0])
        for item in tracks:
            item=copy.deepcopy(item);i=item['id'];old=data['tracks'].get(i)
            if not old and len(data['tracks'])>=10000:raise ValueError('文件库最多保存 10000 首歌曲')
            if old and old.get('size')==item.get('size') and old.get('mtime_ns')==item.get('mtime_ns'):
                item['cache_keys']=old.get('cache_keys',[]);item['prepared_files']=old.get('prepared_files',[])
            if i not in data['tracks']:data['order'].append(i)
            data['tracks'][i]=item
            if i not in target['tracks']:target['tracks'].append(i)
            ids.append(i)
        if data!=self.data:self.commit(data)
        return list(dict.fromkeys(ids))

    def ensure(self,path):
        i=track_id(path);old=self.data['tracks'].get(i)
        if old:
            try:
                st=Path(path).stat()
                if old.get('size')==st.st_size and old.get('mtime_ns')==st.st_mtime_ns:return i
            except OSError:return i
        return self.add([probe_track(path)])[0]

    def create_playlist(self,name):
        if len(self.data['playlists'])>=128:raise ValueError('最多保存 128 个歌单')
        name=name.strip()
        if not name or len(name)>80:raise ValueError('歌单名称需要1到80个字符')
        data=copy.deepcopy(self.data);i=uuid.uuid4().hex
        data['playlists'].append(dict(id=i,name=name,tracks=[]));data['active']=i;self.commit(data);return i

    def rename_playlist(self,i,name):
        if i=='all':raise ValueError('全部音乐不可重命名')
        name=name.strip()
        if not name or len(name)>80:raise ValueError('歌单名称需要1到80个字符')
        data=copy.deepcopy(self.data)
        next(p for p in data['playlists'] if p['id']==i)['name']=name;self.commit(data)

    def delete_playlist(self,i):
        if i in ('all','default'):raise ValueError('保留全部音乐和默认歌单')
        data=copy.deepcopy(self.data);data['playlists']=[p for p in data['playlists'] if p['id']!=i]
        if data['active']==i:data['active']='all'
        self.commit(data)

    def activate(self,i):
        if i not in ['all']+[p['id'] for p in self.data['playlists']]:return
        data=copy.deepcopy(self.data);data['active']=i;self.commit(data)

    def add_to_playlist(self,ids,playlist):
        data=copy.deepcopy(self.data);p=next(p for p in data['playlists'] if p['id']==playlist)
        p['tracks']=list(dict.fromkeys(p['tracks']+[i for i in ids if i in data['tracks']]));self.commit(data)

    def remove(self,ids,playlist=None):
        playlist=playlist or self.data['active'];ids=set(ids);data=copy.deepcopy(self.data)
        if playlist=='all':
            data['tracks']={k:v for k,v in data['tracks'].items() if k not in ids};data['order']=[i for i in data['order'] if i not in ids]
        for p in data['playlists']:
            if playlist=='all' or p['id']==playlist:p['tracks']=[i for i in p['tracks'] if i not in ids]
        self.commit(data)

    def reorder(self,ids,playlist=None):
        playlist=playlist or self.data['active']
        if len(ids)!=len(set(ids)) or set(ids)!=set(self.ids(playlist)):raise ValueError('歌单顺序不完整')
        data=copy.deepcopy(self.data)
        if playlist=='all':data['order']=list(ids)
        else:next(p for p in data['playlists'] if p['id']==playlist)['tracks']=list(ids)
        self.commit(data)

    def relink(self,old,path):
        item=probe_track(path)
        if item['error']:raise ValueError(item['error'])
        data=copy.deepcopy(self.data);new=item['id'];data['tracks'][new]=item
        if old!=new:data['tracks'].pop(old,None)
        data['order']=list(dict.fromkeys(new if i==old else i for i in data['order']))
        for p in data['playlists']:p['tracks']=list(dict.fromkeys(new if i==old else i for i in p['tracks']))
        if data['current']==old:data['current']=new
        self.commit(data);return new

    def converted(self,i,doc,prepared_file=None):
        if i not in self.data['tracks']:return
        data=copy.deepcopy(self.data);track=data['tracks'][i];key=doc.metadata.get('cache',{}).get('key')
        track['duration_s']=doc.duration;track['error']=None
        if key:track['cache_keys']=list(dict.fromkeys([key]+track.get('cache_keys',[])))[:16]
        if key and prepared_file:
            entry=dict(key=key,file=str(prepared_file),baud=doc.profile['baud'])
            track['prepared_files']=[entry]+[p for p in track.get('prepared_files',[]) if p.get('key')!=key]
        data['current']=i
        if data!=self.data:self.commit(data)

    def playable(self,ids):
        result=[]
        for i in ids:
            t=self.data['tracks'].get(i)
            if t and not t.get('error') and (t.get('duration_s') is None or .1<=t['duration_s']<=600) and Path(t['path']).is_file():result.append(i)
        return result

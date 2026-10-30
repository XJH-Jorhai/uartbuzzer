"""Bounded JSON contracts and named profiles; no device access."""
import json
from pathlib import Path
import numpy as np
from .model import PROFILE_SCHEMA,SCORE_SCHEMA,validate_profile,digest,midi_to_hz,hz_to_midi

MAX_JSON_BYTES=8*1024*1024


def load_json(path):
    path=Path(path)
    if path.stat().st_size>MAX_JSON_BYTES:raise ValueError('JSON 文件超过 8 MiB')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('JSON 存在重复字段')
            result[key]=value
        return result
    def bad(value):raise ValueError('JSON 不允许 NaN/Infinity')
    return json.loads(path.read_text(encoding='utf-8-sig'),object_pairs_hook=pairs,parse_constant=bad)


def device_profile(value):
    if not isinstance(value,dict):raise ValueError('需要器件档案对象')
    try:p=validate_profile(value)
    except (KeyError,TypeError,OverflowError) as e:raise ValueError('器件档案字段无效') from e
    if not isinstance(p['device_id'],str) or not 1<=len(p['device_id'])<=120:raise ValueError('档案名称过长或无效')
    if len(p['notes'])>128:raise ValueError('最多128个器件音高点')
    if len(p.get('velocity_levels',[.25,.5,.75,1.]))>8:raise ValueError('最多8档力度')
    if p.get('transport','uart8n1') not in ('uart8n1','generic'):raise ValueError('未知传输类型')
    if type(p.get('max_voices',1)) is not int or p.get('max_voices',1)!=1:raise ValueError('预览版仅支持单声部')
    if type(p.get('require_rest_between_notes',False)) is not bool:raise ValueError('休止开关必须为布尔值')
    for note in p['notes']:
        if len(note['id'])>80:raise ValueError('音符标识过长')
        for k,default in [('frequency_hz',0),('gain',1.),('confidence',0.)]:note[k]=float(note.get(k,default))
        if p['response_kind']=='relative_acoustic' and note['confidence']<.5:raise ValueError('相对声学档案只能包含可靠度≥0.5的有效点；失败证据另存')
    return p


def flat_profile(name='未测平坦 · MIDI48–96',midi_min=48,midi_max=96,transport='uart8n1'):
    if type(midi_min) is not int or type(midi_max) is not int or not 24<=midi_min<=midi_max<=108:raise ValueError('音高范围应为 MIDI24–108')
    return device_profile(dict(schema=PROFILE_SCHEMA,device_id=name,response_kind='unmeasured_flat',
        notes=[dict(id=f'MIDI{m}',frequency_hz=float(midi_to_hz(m)),gain=1.,confidence=0.) for m in range(midi_min,midi_max+1)],
        velocity_levels=[.25,.5,.75,1.],min_note_s=.08,min_rest_s=.02,max_voices=1,
        max_pitch_step_semitones=48.,require_rest_between_notes=False,transport=transport,
        provenance=dict(measured=False,description='手动能力范围；平坦响应假设，非声压测量')))


def score_ir(value):
    """Validate untrusted saved IR before export, cache reuse or rendering."""
    if not isinstance(value,dict) or value.get('schema')!=SCORE_SCHEMA:raise ValueError('未知曲谱 schema')
    # Reject nonfinite numbers anywhere, including unexamined metadata.
    json.dumps(value,allow_nan=False)
    duration=value.get('duration_s');quantum=value.get('time_quantum_s')
    if not isinstance(duration,(int,float)) or not 0<duration<=600:raise ValueError('曲谱时长无效')
    if not isinstance(quantum,(int,float)) or not .005<=quantum<=.2 or value.get('voices')!=1:raise ValueError('曲谱量化或声部无效')
    palette=value.get('palette',[]);events=value.get('events',[]);rests=value.get('rests',[])
    if not isinstance(palette,list) or not 1<=len(palette)<=64 or len(events)>30000 or len(rests)>30001:raise ValueError('曲谱规模无效')
    allowed={}
    for n in palette:
        if not isinstance(n.get('id'),str) or n['id'] in allowed or not 0<float(n['frequency_hz'])<1e6:raise ValueError('曲谱字典无效')
        if abs(float(hz_to_midi(n['frequency_hz']))-float(n['midi_pitch']))>1e-6:raise ValueError('音高坐标不一致')
        allowed[n['id']]=n
    levels=value.get('velocity_levels',[])
    if not levels or len(levels)>8 or any(not 0<float(v)<=1 for v in levels) or sorted(set(levels))!=levels:raise ValueError('力度字典无效')
    intervals=[];last=0.
    for event in events:
        start=float(event['start_s']);length=float(event['duration_s']);note=allowed.get(event['note_id'])
        if note is None or event['voice']!=0 or event['velocity'] not in levels or event['velocity_index']!=levels.index(event['velocity']):raise ValueError('曲谱事件字典无效')
        if event['frequency_hz']!=note['frequency_hz'] or abs(event['midi_pitch']-note['midi_pitch'])>1e-6:raise ValueError('事件音高不一致')
        if start<last-1e-7 or start<0 or length<=0 or start+length>duration+1e-7:raise ValueError('事件时间无效')
        if abs(start/quantum-round(start/quantum))>1e-6:raise ValueError('起音不在量化格')
        if start+length<duration-1e-7 and abs(length/quantum-round(length/quantum))>1e-6:raise ValueError('时值不在量化格')
        last=start+length;intervals.append((start,last))
    for rest in rests:
        a=float(rest['start_s']);b=a+float(rest['duration_s'])
        if a<0 or b<=a or b>duration+1e-7:raise ValueError('休止时间无效')
        intervals.append((a,b))
    cursor=0.
    for a,b in sorted(intervals):
        if abs(a-cursor)>1e-6:raise ValueError('音符与休止必须完整且无重叠地覆盖时间线')
        cursor=b
    if abs(cursor-duration)>1e-6:raise ValueError('曲谱时间线不完整')
    return value


class ProfileStore:
    """Append content-addressed profiles. Import never changes selection."""
    def __init__(self,folder):
        self.folder=Path(folder);self.profiles={};self.errors=[]
        if self.folder.exists():
            for path in sorted(self.folder.glob('*.json')):
                try:
                    p=device_profile(load_json(path));key=digest(p)
                    if path.stem!=key:raise ValueError('档案内容哈希不符')
                    self.profiles[key]=p
                except Exception as e:self.errors.append(path.name+': '+str(e))

    def add(self,value):
        p=device_profile(value);key=digest(p);self.folder.mkdir(parents=True,exist_ok=True)
        path=self.folder/(key+'.json')
        if not path.exists():
            temporary=self.folder/(key+'.pending')
            temporary.write_text(json.dumps(p,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temporary.replace(path)
        self.profiles[key]=p
        return key

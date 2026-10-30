"""Portable score JSON and optional standard MIDI, without a model dependency."""
import json,struct
from pathlib import Path
from .contracts import score_ir


def export_json(score,path):
    score_ir(score);Path(path).write_text(json.dumps(score,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def _vlq(value):
    out=[value&127];value>>=7
    while value:out.insert(0,(value&127)|128);value>>=7
    return bytes(out)


def export_midi(score,path):
    score_ir(score);messages=[]
    for e in score['events']:
        pitch=round(e['midi_pitch'])
        if not 0<=pitch<=127:raise ValueError('曲谱音高超出MIDI；请导出JSON')
        start=round(e['start_s']*960);end=max(start+1,round((e['start_s']+e['duration_s'])*960))
        messages.extend([(start,1,bytes([0x90,pitch,max(1,round(e['velocity']*127))])),(end,0,bytes([0x80,pitch,0]))])
    track=bytearray(b'\x00\xff\x51\x03\x07\xa1\x20');last=0
    for tick,_,data in sorted(messages):track+=_vlq(tick-last)+data;last=tick
    track+=b'\x00\xff\x2f\x00'
    Path(path).write_bytes(b'MThd'+struct.pack('>IHHH',6,0,1,480)+b'MTrk'+struct.pack('>I',len(track))+track)

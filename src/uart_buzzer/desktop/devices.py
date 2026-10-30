"""Serial playback and microphone ownership, independent of Qt widgets."""
from collections import deque
import copy
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import ctypes
import numpy as np
import serial
from serial.tools import list_ports
import sounddevice as sd
from scipy.io import wavfile
from .engine import LiveEncoder,COUNTS
from ..music import FS
from .transport import SerialWriter
from .song_calibration import freeze_recording_context,file_hash


def serial_devices():
    return [dict(port=p.device,description=p.description or p.device) for p in list_ports.comports()]


def microphones():
    apis=sd.query_hostapis();result=[]
    for i,d in enumerate(sd.query_devices()):
        if d['max_input_channels']>0 and apis[d['hostapi']]['name']=='Windows WASAPI':
            result.append(dict(index=i,name=d['name'],rate=int(d['default_samplerate'])))
    return result


class Microphone:
    def __init__(self):
        self.lock=threading.Lock();self.stream=None;self.ring=deque(maxlen=120);self.chunks=[]
        self.recording=False;self.rate=44100;self.total_frames=0;self.start_frame=0;self.statuses=[]

    def open(self,index,rate):
        if self.stream:self.close()
        self.rate=rate;self.device=index;self.total_frames=0;self.ring.clear()
        def callback(data,frames,timing,status):
            a=data[:,0].copy()
            with self.lock:
                self.ring.append(a);self.total_frames+=len(a)
                if self.recording:self.chunks.append(a)
                if status:self.statuses.append(str(status))
        self.stream=sd.InputStream(device=index,samplerate=rate,channels=1,dtype='float32',blocksize=1024,callback=callback)
        try:self.stream.start()
        except Exception:
            self.stream.close();self.stream=None;raise

    def recent(self,count=8192):
        with self.lock:
            a=list(self.ring)
        return np.concatenate(a)[-count:] if a else np.zeros(count,np.float32)

    def start(self):
        if not self.stream:raise RuntimeError('请先启用麦克风')
        with self.lock:
            if self.recording:raise RuntimeError('录音已经开始')
            self.chunks=[];self.statuses=[];self.start_frame=self.total_frames;self.recording=True

    def stop(self,folder,metadata=None):
        with self.lock:
            if not self.recording:return None
            self.recording=False;parts=self.chunks;self.chunks=[];statuses=list(self.statuses)
        data=np.concatenate(parts) if parts else np.zeros(0,np.float32)
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        path=folder/'microphone_raw.wav';wavfile.write(path,self.rate,data)
        meta=dict(metadata or {},sample_rate_Hz=self.rate,samples=len(data),duration_s=len(data)/self.rate,
                  clipped_fraction=float(np.mean(abs(data)>=.999)) if len(data) else 0.,
                  portaudio_status=statuses,absolute_SPL_dB='unknown',normalized=False)
        (folder/'recording.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        return path

    def close(self):
        if self.stream:
            self.stream.stop();self.stream.close();self.stream=None


class SerialLink:
    def __init__(self):self.port=None;self.name=''
    @property
    def connected(self):return bool(self.port and self.port.is_open)
    def open(self,name,baud):
        self.close()
        p=serial.Serial(port=None,baudrate=baud,timeout=.02,write_timeout=.5)
        p.port=name;p.dtr=False;p.rts=False
        try:p.open();p.break_condition=False
        except Exception:
            if p.is_open:p.close()
            raise
        self.port=p;self.name=name
    def close(self):
        if self.port and self.port.is_open:
            try:self.port.break_condition=False
            finally:self.port.close()
        self.port=None
    def interrupt(self):
        if self.connected:
            try:
                if hasattr(self.port,'_port_handle'):
                    from serial import win32
                    win32.CancelIoEx(self.port._port_handle,None)
                else:self.port.cancel_write()
            except (AttributeError,serial.SerialException,OSError):pass


class Playback:
    """Bounded serial queue; live gain ramps and mute never enqueue zero bytes."""
    def __init__(self,link,document,mic,folder,volume=1.,record=False,telemetry=None):
        self.link=link;self.doc=replace(document,profile=copy.deepcopy(document.profile),metadata=copy.deepcopy(document.metadata));self.mic=mic;self.folder=Path(folder)
        self.calibration_context=freeze_recording_context(self.doc) if record else None
        self.volume=volume;self.muted=False;self.paused=False;self.stop_event=threading.Event()
        self.record=record;self.telemetry=telemetry;self.last_recording=None;self.any_pause=False
        self.volume_events=[]

    def stop(self):self.stop_event.set();self.link.interrupt()

    def wait(self,seconds):
        # Event.wait() can round tiny waits to a Windows scheduler tick. Python's
        # high-resolution sleep keeps musical rests and serial refill timing accurate.
        end=time.perf_counter()+max(0,seconds)
        while not self.stop_event.is_set():
            remaining=end-time.perf_counter()
            if remaining<=0:return False
            time.sleep(min(.005,remaining))
        return True

    def run(self):
        if not self.link.connected:raise RuntimeError('请先连接串口')
        self.folder.mkdir(parents=True,exist_ok=True)
        baud=self.doc.profile['baud'];self.link.port.baudrate=baud
        if self.doc.profile.get('encoder','').startswith('beam_'):
            from .engine import PrecomputedEncoder
            encoder=PrecomputedEncoder(self.doc)
        else:encoder=LiveEncoder(self.doc.profile)
        n=int(self.doc.duration*baud/10)
        bounds=[(round(s['start_s']*baud/10),round(s['end_s']*baud/10)) for s in self.doc.segments]
        frame=0;segment=0;written=0;was_idle=True;owns_record=False;last_gain=None
        record_offset=0.;last_emit=0.;error=None;start=None;paused_time=0.;late_max=0.;pending_decoded=[];timer_active=False
        writer=SerialWriter(self.link.port,self.stop_event,self.wait)
        try:
            if hasattr(ctypes,'windll'):timer_active=ctypes.windll.winmm.timeBeginPeriod(1)==0
            if self.record and self.mic.stream and not self.mic.recording:
                self.mic.start();owns_record=True;self.wait(.15)
            with self.mic.lock:
                record_offset=(self.mic.total_frames-self.mic.start_frame)/self.mic.rate if owns_record else 0.
            start=time.monotonic()
            while frame<n and not self.stop_event.is_set():
                if self.paused:
                    self.any_pause=True;t=time.monotonic()
                    while self.paused and not self.wait(.02):pass
                    paused_time+=time.monotonic()-t
                    continue
                while segment<len(bounds) and frame>=bounds[segment][1]:segment+=1
                active=segment<len(bounds) and frame>=bounds[segment][0]
                boundary=(bounds[segment][1] if active else bounds[segment][0]) if segment<len(bounds) else n
                count=min(2048,n-frame,max(1,boundary-frame))
                gain=0. if self.muted else float(self.volume)
                if gain!=last_gain:self.volume_events.append(dict(source_s=frame*10/baud,volume=gain,ramp_s=min(.01,count*10/baud)));last_gain=gain
                source=self.doc.base_processed if self.doc.base_processed is not None else self.doc.processed
                data,pcm=encoder.block(source,frame,count,gain,active,correction=self.doc.quadratic_correction)
                if self.stop_event.is_set():break
                if data:
                    # Refill by actual driver queue consumption, not host sleeps at nominal
                    # baud: this avoids periodic packet gaps if the device clock differs.
                    waiting=time.monotonic()
                    while writer.out_waiting>4096 and not self.wait(.001):
                        if time.monotonic()-waiting>1:raise RuntimeError('串口队列持续阻塞')
                    if self.stop_event.is_set():break
                    sent=writer.write(data)
                    if self.stop_event.is_set():break
                    if sent!=len(data):raise RuntimeError('串口数据未完整写入')
                    written+=len(data);was_idle=False
                else:
                    if not was_idle:
                        waiting=time.monotonic()
                        while writer.out_waiting and not self.wait(.001):
                            if time.monotonic()-waiting>1:raise RuntimeError('静音前队列排空超时')
                        self.wait(.004);was_idle=True
                    self.wait(count*10/baud)
                frame+=count
                dec=(COUNTS[np.frombuffer(data,np.uint8)]/10-self.doc.profile['bias'])/self.doc.profile['amplitude'] if data else np.zeros(count)
                pending_decoded.append(dec.astype(np.float32))
                now=time.monotonic()
                if self.telemetry and now-last_emit>.035:
                    last_emit=now
                    pos=max(0,frame-writer.out_waiting)*10/baud
                    self.telemetry(dict(position_s=pos,bytes=written,volume=gain,
                                        decoded=np.concatenate(pending_decoded),decoded_rate=baud/10,late_max_s=late_max))
                    pending_decoded=[]
            # Bound waiting for OS queue; physical USB FIFO completion remains an estimate.
            until=time.monotonic()+1.
            while writer.out_waiting and not self.stop_event.is_set():
                if time.monotonic()>until:raise RuntimeError('串口输出队列排空超时')
                self.wait(.002)
            if self.stop_event.is_set():self.link.port.reset_output_buffer()
            else:
                self.wait(.02)
        except Exception as e:
            if not self.stop_event.is_set():error=str(e);raise
        finally:
            try:writer.close(cancel=self.stop_event.is_set() or bool(error))
            except Exception as e:
                if not self.stop_event.is_set():error=error or str(e)
            if timer_active:ctypes.windll.winmm.timeEndPeriod(1)
            try:
                if self.link.connected:
                    self.link.port.break_condition=False
                    if self.stop_event.is_set() or error:self.link.port.reset_output_buffer()
            except Exception:pass  # USB removal must not skip saving the microphone capture.
            metadata=dict(source=self.doc.name,source_path=self.doc.path,profile=self.doc.profile,
                          completed=frame>=n,stopped=self.stop_event.is_set(),source_position_s=frame*10/baud,
                          volume_events=self.volume_events,paused=self.any_pause,error=error,
                          transmitted_bytes=written,host_lateness_max_s=late_max,
                          serial_transport='windows_overlapped_double_buffer' if writer.factory else 'synchronous_fallback',
                          maximum_pending_writes=writer.maximum_pending,
                          record_source_start_s=record_offset,physical_wire_timing_measured=False,
                          reconstruction_is_digital=True)
            metadata['calibration_context']=copy.deepcopy(self.calibration_context)
            metadata['predistortion']=self.doc.metadata.get('predistortion',dict(enabled=False))
            if owns_record:
                time.sleep(.15)
                self.last_recording=self.mic.stop(self.folder,metadata)
                ref=self.doc.base_processed.copy() if self.doc.base_processed is not None else self.doc.processed.copy()
                envelope=np.zeros(len(ref));times=np.arange(len(ref))/FS
                previous=0.;cursor=0
                for event in self.volume_events:
                    a=int(event['source_s']*FS);b=min(len(ref),a+max(1,int(event['ramp_s']*FS)))
                    envelope[cursor:a]=previous
                    envelope[a:b]=np.linspace(previous,event['volume'],b-a)
                    previous=event['volume'];cursor=b
                envelope[cursor:]=previous
                ref*=envelope
                if self.doc.quadratic_correction is not None:ref+=self.doc.quadratic_correction*envelope**2
                wavfile.write(self.folder/'reference_processed.wav',FS,ref)
                metadata['calibration_reference_sha256']=file_hash(self.folder/'reference_processed.wav')
                metadata['calibration_capture_sha256']=file_hash(self.last_recording)
            (self.folder/'playback.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        if error:raise RuntimeError(error)
        return dict(position_s=frame*10/baud,bytes=written,recording=str(self.last_recording) if self.last_recording else None,
                    paused=self.any_pause,completed=frame>=n)

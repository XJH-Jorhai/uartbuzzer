"""Bounded Windows overlapped writes: keep the next packet queued on the wire.

Each request owns its buffer, event and OVERLAPPED until completion/cancellation.
See https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-writefile
"""
from collections import deque
import ctypes
import time


class _WriteRequest:
    def __init__(self,port,data):
        from serial import win32
        self.api=win32;self.handle=port._port_handle;self.size=len(data)
        self.buffer=ctypes.create_string_buffer(data);self.overlap=win32.OVERLAPPED()
        self.overlap.hEvent=win32.CreateEvent(None,True,False,None)
        if not self.overlap.hEvent:raise OSError('Cannot allocate serial write event')
        self.closed=False
        count=win32.DWORD()
        ok=win32.WriteFile(self.handle,self.buffer,self.size,ctypes.byref(count),ctypes.byref(self.overlap))
        code=win32.GetLastError() if not ok else 0
        if not ok and code!=win32.ERROR_IO_PENDING:
            self.release();raise OSError(code,'Serial asynchronous write failed')

    def done(self):
        count=self.api.DWORD()
        ok=self.api.GetOverlappedResult(self.handle,ctypes.byref(self.overlap),ctypes.byref(count),False)
        code=self.api.GetLastError() if not ok else 0
        if not ok and code in (996,self.api.ERROR_IO_PENDING):return False
        if not ok:raise OSError(code,'Serial write completion failed')
        if count.value!=self.size:raise OSError('Incomplete asynchronous serial write')
        return True

    def cancel(self):
        self.api.CancelIoEx(self.handle,ctypes.byref(self.overlap))
        count=self.api.DWORD()
        # Cancellation is asynchronous; retain all memory until the request completes.
        self.api.GetOverlappedResult(self.handle,ctypes.byref(self.overlap),ctypes.byref(count),True)

    def release(self):
        if not self.closed:self.api.CloseHandle(self.overlap.hEvent);self.closed=True


class SerialWriter:
    def __init__(self,port,stop_event,wait,request_factory=None):
        self.port=port;self.stop=stop_event;self.wait=wait;self.pending=deque();self.maximum_pending=0
        self.factory=request_factory or (_WriteRequest if hasattr(port,'_port_handle') else None)

    def reap(self):
        while self.pending and self.pending[0].done():self.pending.popleft().release()

    @property
    def out_waiting(self):
        self.reap()
        return max(sum(r.size for r in self.pending),self.port.out_waiting)

    def write(self,data):
        if self.factory is None:return self.port.write(data)
        deadline=time.monotonic()+1.5
        while len(self.pending)>=2:
            if self.stop.is_set():return 0
            self.reap()
            if len(self.pending)<2:break
            if time.monotonic()>deadline:raise TimeoutError('Serial asynchronous queue stalled')
            self.wait(.001)
        if self.stop.is_set():return 0
        self.pending.append(self.factory(self.port,data));self.maximum_pending=max(self.maximum_pending,len(self.pending))
        return len(data)

    def close(self,cancel=False):
        try:
            if cancel:
                for request in self.pending:request.cancel()
            else:
                deadline=time.monotonic()+1.5
                while self.pending:
                    if self.stop.is_set():
                        for request in self.pending:request.cancel()
                        break
                    self.reap()
                    if time.monotonic()>deadline:raise TimeoutError('Serial drain stalled')
                    self.wait(.001)
        finally:
            # On an I/O failure, explicitly settle every request before releasing memory.
            for request in self.pending:
                request.cancel();request.release()
            self.pending.clear()

"""One SQLite writer per stage; workers keep independent read connections.

Requests use a private local Unix socket. Completion is acknowledged only after
SQLite commits. The server never accepts arbitrary file-write commands.
"""
from contextlib import contextmanager
import json
import fcntl
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import threading
from .common import BASE
from .state import State
from .failure_policy import InfrastructureError

MAX_MESSAGE=128*1024*1024
HEARTBEAT_SECONDS=5.


def receive(stream):
    def exact(count):
        result=bytearray()
        while len(result)<count:
            part=stream.recv(count-len(result))
            if not part:raise EOFError('Persistence connection closed')
            result.extend(part)
        return result
    size=struct.unpack('!I',exact(4))[0]
    if size>MAX_MESSAGE:raise ValueError('Persistence message too large')
    return json.loads(exact(size))


def send(stream,value):
    data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
    if len(data)>MAX_MESSAGE:raise ValueError('Persistence message too large')
    stream.sendall(struct.pack('!I',len(data))+data)


class WriterStopped(RuntimeError):
    """Cooperative cancellation, not an OSError/persistence failure."""


class StateWriter:
    def __init__(self,path,timeout=120,stop_requested=None):
        self.path=str(path);self.timeout=timeout;self.stop_requested=stop_requested

    def check_stop(self):
        if self.stop_requested is not None and self.stop_requested():
            raise WriterStopped('Stopped while waiting for durable writer')

    def put_many(self,rows):
        try:
            # Backpressure before opening a socket: queue time must not consume
            # a request's transport timeout. flock releases automatically on exit.
            with open(self.path+'.clients.lock','a') as admission:
                while True:
                    self.check_stop()
                    try:
                        fcntl.flock(admission,fcntl.LOCK_EX|fcntl.LOCK_NB)
                        break
                    except BlockingIOError:time.sleep(.001)
                self.check_stop()
                with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as stream:
                    stream.settimeout(self.timeout);stream.connect(self.path)
                    send(stream,dict(rows=rows))
                    while True:
                        result=receive(stream)
                        if result.get('waiting'):
                            self.check_stop()
                            continue
                        if not result.get('ok'):raise InfrastructureError(result.get('error','Writer failed'))
                        break
        except InterruptedError:raise
        except (OSError,EOFError,ValueError) as error:
            raise InfrastructureError(f'Durable writer unavailable: {error}') from error

    def put(self,stage,model,item,signature,status,output=None,error=None,seconds=0):
        self.put_many([(stage,model,item,signature,status,output,error,seconds)])


def serve(root,path):
    path=str(path)
    state=State(root);running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as server:
        server.bind(path);os.chmod(path,0o600);server.listen(32);server.settimeout(.2)
        try:
            while running:
                try:stream,_=server.accept()
                except socket.timeout:continue
                with stream:
                    try:
                        stream.settimeout(120);request=receive(stream);rows=request['rows']
                        if not isinstance(rows,list) or any(len(row)!=8 for row in rows):raise ValueError('Invalid state rows')
                        finished=threading.Event();send_lock=threading.Lock()
                        def heartbeat():
                            while not finished.wait(HEARTBEAT_SECONDS):
                                try:
                                    with send_lock:send(stream,dict(waiting=True))
                                except OSError:return
                        pulse=threading.Thread(target=heartbeat,daemon=True)
                        pulse.start()
                        try:
                            state.put_many(rows)
                        finally:
                            finished.set();pulse.join(timeout=1)
                        with send_lock:send(stream,dict(ok=True))
                    except Exception as error:
                        try:send(stream,dict(ok=False,error=str(error)))
                        except OSError:pass
        finally:state.db.close();Path(path).unlink(missing_ok=True)


@contextmanager
def writer_service(root,scratch):
    Path(scratch).mkdir(parents=True,exist_ok=True)
    # Linux Unix-domain socket names are limited to 107 bytes. A relocated
    # workspace can exceed that even when its files and directories are valid.
    socket_root=scratch if len(os.fsencode(str(Path(scratch).absolute())))+32<104 else '/tmp'
    with tempfile.TemporaryDirectory(prefix='writer-',dir=socket_root) as directory:
        path=str(Path(directory)/'state.sock')
        with open(Path(directory)/'server.log','w') as log:
            process=subprocess.Popen([sys.executable,'-m','audio_pipeline.writer',str(root),path],stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+20
                while not Path(path).exists():
                    if process.poll() is not None:raise RuntimeError('Writer failed: '+Path(log.name).read_text())
                    if time.monotonic()>deadline:raise TimeoutError('Writer startup timed out')
                    time.sleep(.02)
                yield path
            finally:
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait()


if __name__=='__main__':serve(sys.argv[1],sys.argv[2])

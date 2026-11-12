"""Runtime headroom checks shared by standalone and resident workers."""
from .memory import available_host_bytes
from .failure_policy import InfrastructureError
import json
import time


class MemoryPressureError(InfrastructureError):
    pass


def check_host(settings,stop_requested=lambda:False):
    minimum=settings.get('min_free_gib',0)
    fraction=settings.get('cache_reclaim_fraction',.5)
    timeout=settings.get('wait_timeout_seconds',0)
    poll=settings.get('wait_poll_seconds',2)
    if timeout<0 or poll<=0:raise ValueError('Memory wait timeout must be nonnegative and poll interval positive')
    started=time.monotonic();last_log=None
    while True:
        if stop_requested():raise InterruptedError('Stopped while waiting for host RAM')
        free=available_host_bytes(fraction)/2**30
        if free>=minimum:return free
        elapsed=time.monotonic()-started
        if elapsed>=timeout:
            raise MemoryPressureError(f'Host RAM headroom {free:.2f}GiB is below configured {minimum}GiB after {elapsed:.1f}s; committed results retained')
        if last_log is None or elapsed-last_log>=10:
            print(json.dumps(dict(event='memory_wait',available_gib=round(free,2),minimum_gib=minimum,
                elapsed_seconds=round(elapsed,1),timeout_seconds=timeout)),flush=True)
            last_log=elapsed
        deadline=time.monotonic()+min(poll,timeout-elapsed)
        while time.monotonic()<deadline:
            if stop_requested():raise InterruptedError('Stopped while waiting for host RAM')
            time.sleep(min(.1,max(0,deadline-time.monotonic())))

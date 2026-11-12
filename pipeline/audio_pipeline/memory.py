"""Conservative host RAM headroom, respecting container limits and file reclaim."""
from pathlib import Path


def available_host_bytes(cache_reclaim_fraction=.5):
    if not 0<=cache_reclaim_fraction<=1:raise ValueError('cache_reclaim_fraction must be in [0,1]')
    fields={line.split(':',1)[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line and line.split()[1].isdigit()}
    available=fields['MemAvailable']
    root=Path('/sys/fs/cgroup')
    try:
        limit=(root/'memory.max').read_text().strip()
        if limit!='max':
            current=int((root/'memory.current').read_text())
            try:stats=dict(line.split() for line in (root/'memory.stat').read_text().splitlines())
            except (OSError,ValueError):stats={}
            # file includes tmpfs/shmem and dirty pages. Exclude those and
            # unevictable pages; list counters alone omit clean active file cache.
            excluded=sum(int(stats.get(key,0)) for key in ('shmem','file_dirty','file_writeback','unevictable'))
            clean=max(0,int(stats.get('file',0))-excluded)
            reclaimable=int(clean*cache_reclaim_fraction)
            container=min(int(limit),max(0,int(limit)-current+reclaimable))
            available=min(available,container)
    except (OSError,ValueError):pass
    return available

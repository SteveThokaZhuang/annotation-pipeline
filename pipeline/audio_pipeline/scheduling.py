"""Deterministic duration-balanced shards without a distributed queue service."""
import heapq


def balanced_shards(tasks,workers,by_recording=False):
    if workers<1:return []
    shards=[[] for _ in range(workers)]
    loads=[(0.,index) for index in range(workers)]
    heapq.heapify(loads)
    def cost(task):
        item=task['item'];chunk=item.get('chunk',item)
        return max(.05,float(chunk.get('duration',1)))+len(item.get('text',''))*.002
    if by_recording:
        recordings={}
        for task in tasks:
            item=task['item'];chunk=item.get('chunk',item)
            key=chunk.get('recording_id',chunk.get('audio_path',chunk.get('path',chunk.get('id'))))
            recordings.setdefault(key,[]).append(task)
        # Assign whole recordings to a worker. Within each recording, nearby
        # durations share batches without scattering reads across the dataset.
        groups=sorted(recordings.values(),key=lambda group:sum(map(cost,group)),reverse=True)
        for group in groups:
            load,index=heapq.heappop(loads)
            shards[index].extend(sorted(group,key=cost,reverse=True))
            heapq.heappush(loads,(load+sum(map(cost,group)),index))
        return shards
    for task in sorted(tasks,key=cost,reverse=True):
        load,index=heapq.heappop(loads)
        shards[index].append(task)
        heapq.heappush(loads,(load+cost(task),index))
    return shards

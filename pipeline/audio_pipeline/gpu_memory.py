"""Runtime memory feedback, separate from inference/cache signatures."""


def observe_cuda_memory(policy,cuda,external=False):
    """Reclaim idle allocator blocks before interpreting low driver-free memory.

    Cached reservations can be reused by PyTorch. Treating them as live memory
    repeatedly halves the batch limit without freeing a single byte. Reclaim
    only under pressure, then keep the existing peak-memory and OOM safeguards.
    """
    free,total=cuda.mem_get_info()
    reclaimed=False
    if free<policy.min_free_gib*2**30 and cuda.memory_reserved()>cuda.memory_allocated():
        cuda.empty_cache()
        free,total=cuda.mem_get_info()
        reclaimed=True
    peak=cuda.max_memory_allocated()
    before=policy.limit
    if external:
        if policy.adaptive:
            if free<policy.min_free_gib*2**30:policy.limit=max(1,policy.limit//2)
            else:policy.limit=min(policy.maximum,policy.limit+max(1,policy.limit//8))
    else:policy.observe(peak,total,free)
    return dict(feedback_source='driver_free_memory_and_oom' if external else 'torch',driver_used_gib=(total-free)/2**30,free_gib=free/2**30,allocated_gib=cuda.memory_allocated()/2**30,
                reserved_gib=cuda.memory_reserved()/2**30,peak_allocated_gib=peak/2**30,
                reclaimed_allocator_cache=reclaimed,previous_batch_limit=before,
                next_batch_limit=policy.limit)

"""Per-row EOS metadata and bounded recovery without rerunning finished peers."""


def monitor_generation(model):
    original=model.generate
    stats=[]
    def generate(*args,**kwargs):
        stats.clear()
        output=original(*args,**kwargs)
        sequences=output.sequences if hasattr(output,'sequences') else output
        if getattr(output,'past_key_values',None) is not None:output.past_key_values=None
        prompt=kwargs.get('input_ids')
        offset=prompt.shape[1] if prompt is not None else 0
        eos=kwargs.get('eos_token_id',model.generation_config.eos_token_id)
        eos={eos} if isinstance(eos,int) else set(eos or [])
        if not eos:raise RuntimeError('Cannot verify completion: model has no EOS token ID')
        for row in sequences[:,offset:].tolist():
            end=next((i for i,token in enumerate(row) if token in eos),None)
            stats.append(dict(generated_tokens=len(row) if end is None else end+1,
                              finish_reason='length' if end is None else 'eos'))
        return output
    model.generate=generate
    return stats


def recover_unfinished(items,predict,metrics):
    """predict returns formatted rows plus EOS stats; only unfinished rows recur."""
    results,stats=predict(items)
    if len(results)!=len(items) or len(stats)!=len(items):
        raise RuntimeError('Incomplete ASR output/generation metadata')
    pending=[i for i,stat in enumerate(stats) if stat['finish_reason']!='eos']
    if not pending:return results
    metrics['token_limited_rows']=metrics.get('token_limited_rows',0)+len(pending)
    if len(items)==1:
        # Truncated text is never exposed as a valid completed hypothesis.
        results[0].update(text='',transcription_error='token_limit_without_eos',fallback_requested=True)
        return results
    metrics['retained_completed_rows']=metrics.get('retained_completed_rows',0)+len(items)-len(pending)
    size=max(1,len(items)//2)
    for offset in range(0,len(pending),size):
        indices=pending[offset:offset+size]
        metrics['completion_retry_calls']=metrics.get('completion_retry_calls',0)+1
        values=recover_unfinished([items[i] for i in indices],predict,metrics)
        for index,value in zip(indices,values):
            value.setdefault('completion_recovery',[]).insert(0,dict(previous_batch_items=len(items),
                generated_tokens=stats[index]['generated_tokens'],finish_reason=stats[index]['finish_reason']))
            results[index]=value
    return results

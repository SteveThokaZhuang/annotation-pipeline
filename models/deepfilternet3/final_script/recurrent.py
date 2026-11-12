"""Bound long CUDA GRU calls while carrying hidden state across all frames.

Only the recurrent layer is partitioned. STFT, feature normalization,
convolutions and synthesis retain their original whole-recording behavior.
"""
def bound_gru_sequences(model, max_frames):
    import torch
    if max_frames<1:raise ValueError('gru_chunk_frames must be positive')
    count=0
    for layer in model.modules():
        if not isinstance(layer,torch.nn.GRU):continue
        if layer.bidirectional:
            raise ValueError('State-carrying GRU partitioning requires a unidirectional model')
        original=layer.forward
        axis=1 if layer.batch_first else 0
        def forward(inputs,hidden=None,_original=original,_axis=axis):
            if not isinstance(inputs,torch.Tensor) or inputs.ndim!=3 or inputs.shape[_axis]<=max_frames:
                return _original(inputs,hidden)
            outputs=[]
            for block in inputs.split(max_frames,dim=_axis):
                value,hidden=_original(block.contiguous(),hidden)
                outputs.append(value)
            return torch.cat(outputs,dim=_axis),hidden
        layer.forward=forward
        count+=1
    return count

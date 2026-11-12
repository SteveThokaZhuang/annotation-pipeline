"""Independent per-recording CPU STFT state and exact native synthesis."""
import time


def prepare(item,raw_prepare,max_gib,parameters,nb_df):
    # Charge spectra, normalized features, decoded arrays and working headroom.
    estimate=item['duration']*item['channels']*(item['sample_rate']+48000)*4*4
    if estimate>max_gib*2**30:return None
    audio=raw_prepare(item,max_gib)
    if audio is None:return None
    import torch
    from torch.nn import functional as F
    from libdf import DF
    from df.enhance import df_features
    state=DF(**parameters);tick=time.perf_counter()
    waveform=torch.from_numpy(audio.T.copy());orig_len=waveform.shape[-1]
    waveform=F.pad(waveform,(0,state.fft_size()))
    features=df_features(waveform,state,nb_df,device=None)
    return dict(spectral_features=True,state=state,features=features,orig_len=orig_len,
                feature_seconds=time.perf_counter()-tick)


def enhance(model,prepared,channel,count):
    import torch
    from df.utils import as_complex
    model.eval()
    count=min(count,prepared['features'][0].shape[0]-channel)
    device=next(model.parameters()).device
    if hasattr(model,'reset_h0'):model.reset_h0(batch_size=count,device=device)
    tick=time.perf_counter()
    from audio_pipeline.transfers import to_device
    spec,erb_feat,spec_feat=[to_device(value[channel:channel+count],device) for value in prepared['features']]
    enhanced=model(spec.clone(),erb_feat,spec_feat)[0].cpu()
    model_seconds=time.perf_counter()-tick;tick=time.perf_counter()
    enhanced=as_complex(enhanced.squeeze(1))
    state=prepared['state'];audio=torch.as_tensor(state.synthesis(enhanced.numpy()))
    delay=state.fft_size()-state.hop_size();audio=audio[:,delay:prepared['orig_len']+delay]
    return audio.numpy(),dict(model_and_transfer_seconds=model_seconds,synthesis_seconds=time.perf_counter()-tick)

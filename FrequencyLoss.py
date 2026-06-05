import csv
import torch
import torchaudio

def frequency_filter(wav_diff, sampling_rate, n_fft=2048, hop_length=512, csv_path=''):
    """
    wav_diff: shape [B, T], 
    : shape [B]
    """
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft, hop_length=hop_length
    ).cuda()

    #  [B, freq_bins, frames]
    diff_spec = spectrogram(wav_diff)

    #  CSV  diff_spec  freq_bins 
    # ** batch  sum/mean
    #  freq_bins, frames  sum
    # 
    B, freq_bins, time_frames = diff_spec.shape

    #  CSV ...
    xs, ys = [], []
    # ... csv load & scaled ...
    
    # 
    for i in range(freq_bins):
        bin_freq = sampling_rate / 2 / freq_bins
        probe_freq = (i + 0.5) * bin_freq
        #  CSV 
        for j in range(len(xs) - 1):
            if xs[j] <= probe_freq <= xs[j + 1]:
                weight_freq = ys[j] + ((probe_freq - xs[j]) * (ys[j + 1] - ys[j])) / (xs[j + 1] - xs[j])
                #  batch, time_frames 
                diff_spec[:, i, :] *= weight_freq
                break

    #  diff_spec  [B, freq_bins, time_frames]
    #  freq_bins, time_frames  => shape [B]
    loss_per_sample = diff_spec.sum(dim=[1,2]) / (freq_bins * time_frames)
    #  loss_per_sample  shape  [B]

    return loss_per_sample



# 
if __name__ == "__main__":
    # 
    sampling_rate = 16000
    T = sampling_rate * 2  # 2 
    wav_diff = torch.randn(1, T).cuda()

    # 
    try:
        loss = frequency_filter(wav_diff, sampling_rate, csv_path='./points.csv')
        print(f"{loss.item()}")
    except Exception as e:
        print(f"{e}")
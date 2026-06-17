import numpy as np
import os
import random
import json
import torch
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import scipy
from preprocess import MAX_WAV_VALUE, normalize
from hyperparameter import hp

device = torch.device(hp.device)


def parse_metadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, is_training=True):
        super().__init__()
        self.data_root = Path(hp.project_root)
        self.params = hp
        self.metadata = parse_metadata(hp.metadata)
        self.is_training = is_training
        self.hop_samples = hp.hop_length

        self.use_prior = hp.use_prior if hasattr(hp, 'use_prior') else False
        self.max_energy_override = hp.max_energy_override if hasattr(hp, 'max_energy_override') else None

        if self.is_training:
            self.compute_stats()

        if self.use_prior:
            # build frame energy data for priorgrad
            self.energy_max = float(np.load(str(self.data_root.joinpath('stats_priorgrad', 'energy_max_train.npy')),
                                            allow_pickle=True))
            self.energy_min = float(np.load(str(self.data_root.joinpath('stats_priorgrad', 'energy_min_train.npy')),
                                            allow_pickle=True))
            print("INFO: loaded frame-level waveform stats : max {} min {}".format(self.energy_max, self.energy_min))
            if self.max_energy_override is not None:
                print("overriding max energy to {}".format(self.max_energy_override))
                self.energy_max = self.max_energy_override
            self.std_min = hp.std_min if hasattr(hp, 'std_min') else 0.0

    def compute_stats(self):
        if os.path.exists(self.data_root.joinpath("stats_priorgrad/energy_max_train.npy")) and \
                os.path.exists(self.data_root.joinpath("stats_priorgrad/energy_min_train.npy")):
            return
        # compute audio stats from the dataset
        # goal: pre-calculate variance of the frame-level part of the waveform
        # which will be used for the modified Gaussian base distribution for PriorGrad model

        energy_list = []
        print("INFO: computing training set waveform statistics for PriorGrad training...")
        for i in tqdm(range(len(self.metadata))):
            _, _, wav_path, _ = self.metadata[i]
            sr, audio = scipy.io.wavfile.read(wav_path)
            if self.params.sample_rate != sr:
                raise ValueError(f'Invalid sample rate {sr}.')
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            # match audio length to self.hop_size * n for evaluation
            if (audio.shape[0] % self.params.hop_length) != 0:
                audio = audio[:-(audio.shape[0] % self.params.hop_length)]
            audio = torch.FloatTensor(audio)
            spectrogram_path = self.metadata[i][0]
            spectrogram = torch.load(spectrogram_path)
            energy = (spectrogram.exp()).sum(1).sqrt()
            energy_list.append(energy.squeeze(0))

        energy_list = torch.cat(energy_list)
        energy_max = energy_list.max().numpy()
        energy_min = energy_list.min().numpy()

        self.data_root.joinpath("stats_priorgrad").mkdir(exist_ok=True)
        print("INFO: stats computed: max energy {} min energy {}".format(energy_max, energy_min))
        np.save(str(self.data_root.joinpath("stats_priorgrad/energy_max_train.npy")), energy_max)
        np.save(str(self.data_root.joinpath("stats_priorgrad/energy_min_train.npy")), energy_min)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        mel_path, gender, wav_path, std_path = self.metadata[idx]

        sr, audio = scipy.io.wavfile.read(wav_path)
        if self.params.sample_rate != sr:
            raise ValueError(f'Invalid sample rate {sr}.')
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95
        # match audio length to self.hop_size * n for evaluation
        if (audio.shape[0] % self.params.hop_length) != 0:
            audio = audio[:-(audio.shape[0] % self.params.hop_length)]
        audio = torch.FloatTensor(audio)

        if self.is_training:
            # get segment of audio
            start = random.randint(0, audio.shape[0] - (self.params.seg_len_mel * self.params.hop_length))
            end = start + (self.params.seg_len_mel * self.params.hop_length)
            audio = audio[start:end]

        spectrogram = torch.load(mel_path)
        target_std = torch.load(std_path)

        return {
            'audio': audio,  # [T_time]
            'spectrogram': spectrogram.T,  # [T_mel, 80]
            'target_std': target_std  # [T_mel]
        }


class Collator:
    def __init__(self, params, is_training=True):
        self.params = params
        self.is_training = is_training

    def collate(self, minibatch):
        samples_per_frame = self.params.hop_length
        for record in minibatch:
            # Filter out records that aren't long enough.
            if len(record['spectrogram']) < self.params.seg_len_mel:
                del record['spectrogram']
                del record['audio']
                continue

            record['spectrogram'] = record['spectrogram'].T
            record['target_std'] = record['target_std']
            record['target_std'] = torch.repeat_interleave(record['target_std'], samples_per_frame)
            record['audio'] = record['audio']

            assert record['audio'].shape == record['target_std'].shape

        audio = torch.stack([record['audio'] for record in minibatch if 'audio' in record])
        spectrogram = torch.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
        target_std = torch.stack([record['target_std'] for record in minibatch if 'target_std' in record])
        return {
            'audio': audio,
            'spectrogram': spectrogram,
            'target_std': target_std
        }


def make_training_data_loader(data_root, params, is_distributed=False):
    dataset = NumpyDataset(is_training=True)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.bs,
        collate_fn=Collator(params, is_training=True).collate,
        shuffle=not is_distributed,
        num_workers=1,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=False,
        drop_last=True)


def make_validation_data_loader(data_root, params, is_distributed=False):
    dataset = NumpyDataset(is_training=False)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        collate_fn=Collator(params, is_training=False).collate,
        shuffle=False,
        num_workers=1,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=False,
        drop_last=False)
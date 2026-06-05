import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F  #  functional  padding
from pathlib import Path
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import from_path as dataset_from_path
from dataset import from_path_valid as dataset_from_path_valid
from model import PriorGrad
from target import vc_model
from preprocess import get_mel
from data.utils import get_melgan_mel
from data.utils import get_melgan_mel_tensor
from hyperparameter import hp 
from params import params
from FrequencyLoss  import frequency_filter
from scale_attention import ScaleAttention


def _nested_map(struct, map_fn):
    if isinstance(struct, tuple):
        return tuple(_nested_map(x, map_fn) for x in struct)
    if isinstance(struct, list):
        return [_nested_map(x, map_fn) for x in struct]
    if isinstance(struct, dict):
        return {k: _nested_map(v, map_fn) for k, v in struct.items()}
    return map_fn(struct)

def scaled_mse_loss(decoder_output, target, target_std):
    # inverse of diagonal matrix is 1/x for each element
    sigma_inv = torch.reciprocal(target_std)
    mse_loss = (((decoder_output - target) * sigma_inv) ** 2)
    mse_loss = (mse_loss).sum() / torch.numel(decoder_output)
    return mse_loss


class PriorGradLearner:
    def __init__(self, model_dir, model, dataset, dataset_val, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.vc_model = vc_model()
        self.dataset = dataset
        self.dataset_val = dataset_val
        self.optimizer = optimizer
        self.params = params
        self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
        self.step = 0
        self.hp = hp
        self.is_master = True
        self.use_frequency_loss = params.use_frequency_loss
        self.lambda_frequency = params.lambda_frequency
        
        # [ STFT ]
        self.stft_configs = [(1024, 256)]

        # []
        device = next(model.parameters()).device
        self.scale_att = ScaleAttention(num_scales=len(self.stft_configs)).to(device)

        # [ ScaleAttention]
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.scale_att.parameters()),
            lr=params.learning_rate
        )

        self.use_l2loss = params.use_l2loss

        self.use_prior = params.use_prior
        self.condition_prior = params.condition_prior
        self.condition_prior_global = params.condition_prior_global

        assert not (self.condition_prior and self.condition_prior_global),\
            "use only one of the following parameter: condition_prior or condition_prior_global"

        beta = np.array(self.params.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

        self.summary_writer = None

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'step': self.step,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in
                          self.optimizer.state_dict().items()},
            'params': dict(self.params),
            'scaler': self.scaler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scaler.load_state_dict(state_dict['scaler'])
        self.step = state_dict['step']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.step}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'

        torch.save(self.state_dict(), save_name)
        print(f"[INFO] Saved checkpoint => {save_name}")

        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name) or os.path.isfile(link_name):
                os.remove(link_name)
            elif os.path.isdir(link_name):
                import shutil
                shutil.rmtree(link_name)
            
            try:
                os.symlink(save_basename, link_name)
            except FileExistsError:
                if os.path.islink(link_name) or os.path.isfile(link_name):
                    os.remove(link_name)
                os.symlink(save_basename, link_name)

    def restore_from_checkpoint(self, filename='weights', custom_checkpoint=''):
        try:
            checkpoint_path = custom_checkpoint if custom_checkpoint else f'{self.model_dir}/{filename}.pt'
            checkpoint = torch.load(checkpoint_path)
            
            if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
                self.model.module.load_state_dict(checkpoint['model'])
            else:
                self.model.load_state_dict(checkpoint['model'])
            
            self.step = checkpoint['step']
            print(f" {checkpoint_path} ")
            return True
        except FileNotFoundError:
            print(f" {checkpoint_path}")
            return False

    def train(self, max_steps=None):
        device = next(self.model.parameters()).device
        while True:
            iterator = tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset
            for features in iterator:
                if max_steps is not None and self.step > max_steps:
                    return
                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
                
                #  train_step  padding  try-except 
                # 
                loss, predicted = self.train_step(features)

                if torch.isnan(loss).any():
                    raise RuntimeError(f'Detected NaN loss at step {self.step}.')
                if self.is_master:
                    if self.step % 50 == 0:
                        self._write_summary(self.step, features, loss)
                    if self.step % 100 == 0:
                        self.run_valid_loop()
                    if self.step % 500 == 0:
                        print("INFO: saving checkpoint at step {}".format(self.step))
                        self.save_to_checkpoint()
                self.step += 1

    def train_step(self, features):
        for param in self.model.parameters():
            param.grad = None

        audio = features['audio']
        spectrogram = features['spectrogram']
        target_std = features['target_std']

        if self.condition_prior:
            target_std_specdim = target_std[:, ::self.params.hop_samples].unsqueeze(1)
            spectrogram = torch.cat([spectrogram, target_std_specdim], dim=1)
            global_cond = None
        elif self.condition_prior_global:
            target_std_specdim = target_std[:, ::self.params.hop_samples].unsqueeze(1)
            global_cond = target_std_specdim
        else:
            global_cond = None

        N, T = audio.shape
        device = audio.device
        self.noise_level = self.noise_level.to(device)

        with self.autocast:
            t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
            noise_scale = self.noise_level[t].unsqueeze(1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(audio)
            noise = noise * target_std
            noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

            # 
            predicted = self.model(noisy_audio, spectrogram, t, global_cond)

            # 
            adv_audio = (noisy_audio - ((1.0 - noise_scale) ** 0.5) * predicted.squeeze(1)) / noise_scale_sqrt
            adv_audio = torch.clamp(adv_audio, -1.0, 1.0)

            # 
            if self.use_frequency_loss:
                wav_diff = adv_audio - audio
                per_scale_losses = []
                for (n_fft, hop) in self.stft_configs:
                    single_loss = frequency_filter(
                        wav_diff,
                        self.params.sample_rate,
                        n_fft=n_fft,
                        hop_length=hop
                    )
                    per_scale_losses.append(single_loss)

                per_scale = torch.stack(per_scale_losses, dim=1)
                alphas = self.scale_att(per_scale)
                freq_loss_per_batch = (alphas * per_scale).sum(dim=1)
                frequency_loss = freq_loss_per_batch.mean()
            else:
                frequency_loss = 0.0

            # 
            x_mel = get_melgan_mel_tensor(audio, is_transposed=False)
            adv_mel = get_melgan_mel_tensor(adv_audio, is_transposed=False)
            src_mel = x_mel.clone().detach()

            if x_mel.dim() == 2: x_mel = x_mel.unsqueeze(0)
            if adv_mel.dim() == 2: adv_mel = adv_mel.unsqueeze(0)
            if src_mel.dim() == 2: src_mel = src_mel.unsqueeze(0)

            # ==========================================================
            # [] 
            # ==========================================================
            min_required_len = 64  # 
            current_len = x_mel.shape[-1]
            
            if current_len < min_required_len:
                pad_amount = min_required_len - current_len
                #  replicate 
                x_mel = F.pad(x_mel, (0, pad_amount), mode='replicate')
                adv_mel = F.pad(adv_mel, (0, pad_amount), mode='replicate')
                src_mel = F.pad(src_mel, (0, pad_amount), mode='replicate')
            # ==========================================================

            #  adv_l2_loss ()
            adv_l2_loss = self.vc_model.adv_loss(adv_mel, x_mel, src_mel)

            if self.use_prior:
                if self.use_l2loss:
                    mse_loss = hp.lambda_quality * scaled_mse_loss(predicted.squeeze(1), noise, target_std)
                    loss = mse_loss - hp.lambda_adv_l2 * adv_l2_loss + self.lambda_frequency * frequency_loss 
                else:
                    raise NotImplementedError
            else:
                if self.use_l2loss:
                    mse_loss = nn.MSELoss()(noise, predicted.squeeze(1))
                    loss = mse_loss - hp.lambda_adv_l2 * adv_l2_loss + self.lambda_frequency * frequency_loss 
                else:
                    loss = nn.L1Loss()(noise, predicted.squeeze(1)) - hp.lambda_adv_l2 * adv_l2_loss + self.lambda_frequency * frequency_loss 

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss, predicted

    def run_valid_loop(self):
        with torch.no_grad():
            device = next(self.model.parameters()).device
            losses = []
            losses_l1 = []
            audio_preds = []

            iterator = tqdm(self.dataset_val, desc=f'Valid {len(self.dataset_val)}') if self.is_master else self.dataset_val
            for features in iterator:
                features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

                audio = features['audio']
                spectrogram = features['spectrogram']
                target_std = features['target_std']

                if self.condition_prior:
                    target_std_specdim = target_std[:, ::self.params.hop_samples].unsqueeze(1)
                    spectrogram = torch.cat([spectrogram, target_std_specdim], dim=1)
                    global_cond = None
                elif self.condition_prior_global:
                    target_std_specdim = target_std[:, ::self.params.hop_samples].unsqueeze(1)
                    global_cond = target_std_specdim
                else:
                    global_cond = None

                N, T = audio.shape
                device = audio.device
                self.noise_level = self.noise_level.to(device)

                t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
                noise_scale = self.noise_level[t].unsqueeze(1)
                noise_scale_sqrt = noise_scale ** 0.5
                noise = torch.randn_like(audio) * target_std
                noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale) ** 0.5 * noise

                if hasattr(self.model, 'module'):
                    predicted = self.model.module(noisy_audio, spectrogram, t, global_cond)
                else:
                    predicted = self.model(noisy_audio, spectrogram, t, global_cond)

                adv_audio = (noisy_audio - ((1.0 - noise_scale) ** 0.5) * predicted.squeeze(1)) / noise_scale_sqrt
                adv_audio = torch.clamp(adv_audio, -1.0, 1.0)

                # 
                if self.use_frequency_loss:
                    wav_diff = adv_audio - audio
                    per_scale_losses = []
                    for (n_fft, hop) in self.stft_configs:
                        single_loss = frequency_filter(wav_diff, self.params.sample_rate, n_fft, hop)
                        per_scale_losses.append(single_loss)
                    per_scale = torch.stack(per_scale_losses, dim=1)
                    alphas = self.scale_att(per_scale)
                    frequency_loss = (alphas * per_scale).sum(dim=1).mean()
                else:
                    frequency_loss = 0.0

                x_mel = get_melgan_mel_tensor(audio, is_transposed=False)
                adv_mel = get_melgan_mel_tensor(adv_audio, is_transposed=False)
                src_mel = x_mel.clone().detach()

                if x_mel.dim() == 2: x_mel = x_mel.unsqueeze(0)
                if adv_mel.dim() == 2: adv_mel = adv_mel.unsqueeze(0)
                if src_mel.dim() == 2: src_mel = src_mel.unsqueeze(0)
                
                # [] Padding
                min_required_len = 64
                current_len = x_mel.shape[-1]
                if current_len < min_required_len:
                    pad_amount = min_required_len - current_len
                    x_mel = F.pad(x_mel, (0, pad_amount), mode='replicate')
                    adv_mel = F.pad(adv_mel, (0, pad_amount), mode='replicate')
                    src_mel = F.pad(src_mel, (0, pad_amount), mode='replicate')

                adv_l2_loss = self.vc_model.adv_loss(adv_mel, x_mel, src_mel)

                if self.use_prior:
                    if self.use_l2loss:
                        mse_loss = hp.lambda_quality * scaled_mse_loss(predicted.squeeze(1), noise, target_std)
                        loss = mse_loss - hp.lambda_adv_l2 * adv_l2_loss + self.lambda_frequency * frequency_loss 
                    else:
                        raise NotImplementedError
                else:
                    if self.use_l2loss:
                        mse_loss = nn.MSELoss()(noise, predicted.squeeze(1))
                        loss = mse_loss - hp.lambda_adv_l2 * adv_l2_loss + self.lambda_frequency * frequency_loss 
                    else:
                        loss = nn.L1Loss()(noise, predicted.squeeze(1)) - hp.lambda_adv_l2 * adv_l2_loss + self.lambda_frequency * frequency_loss 

                losses.append(loss.cpu().numpy())
                audio_pred = adv_audio
                audio_preds.append(audio_pred.cpu().numpy())

                predicted_mel = get_melgan_mel_tensor(audio_pred.squeeze(0), is_transposed=False)
                #  L1 Loss (Pad L1loss  padding  shape )
                #  spectrogram  padding
                if spectrogram.shape[-1] < min_required_len:
                     spectrogram = F.pad(spectrogram, (0, min_required_len - spectrogram.shape[-1]), mode='replicate')
                
                l1_loss_val = torch.nn.L1Loss()(predicted_mel, spectrogram).item()
                losses_l1.append(l1_loss_val)

            loss_valid = np.mean(losses)
            loss_l1 = np.mean(losses_l1)
            self._write_summary_valid(self.step, loss_valid, loss_l1, audio_preds)

    def predict(self, spectrogram, target_std, global_cond=None):
        print(f"Is grad enabled in predict(): {torch.is_grad_enabled()}")
        device = next(self.model.parameters()).device
        training_noise_schedule = np.array(self.params.noise_schedule)
        inference_noise_schedule = np.array(self.params.inference_noise_schedule)

        talpha = 1 - training_noise_schedule
        talpha_cum = np.cumprod(talpha)

        beta = inference_noise_schedule
        alpha = 1 - beta
        alpha_cum = np.cumprod(alpha)

        T = []
        for s in range(len(inference_noise_schedule)):
            for t in range(len(training_noise_schedule) - 1):
                if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                    twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                                talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                    T.append(t + twiddle)
                    break
        T = np.array(T, dtype=np.float32)

        if len(spectrogram.shape) == 2:
            spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.to(device)

        audio = torch.randn(spectrogram.shape[0], self.params.hop_samples * spectrogram.shape[-1],
                            device=device) * target_std
        noise_scale = torch.from_numpy(alpha_cum ** 0.5).float().unsqueeze(1).to(device)

        for n in range(len(alpha) - 1, -1, -1):
            c1 = 1 / alpha[n] ** 0.5
            c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
            if hasattr(self.model, 'module'):
                audio = c1 * (audio - c2 * self.model.module(audio, spectrogram, torch.tensor([T[n]], device=audio.device),
                                                            global_cond).squeeze(1))
            else:
                audio = c1 * (audio - c2 * self.model(audio, spectrogram, torch.tensor([T[n]], device=audio.device),
                                                        global_cond).squeeze(1))
            if n > 0:
                noise = torch.randn_like(audio) * target_std
                sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                audio += sigma * noise
            audio = torch.clamp(audio, -1.0, 1.0)

        return audio

    def _write_summary(self, step, features, loss):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
        writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
        writer.add_scalar('train/loss', loss, step)
        writer.add_scalar('train/grad_norm', self.grad_norm, step)
        writer.flush()
        self.summary_writer = writer

    def _write_summary_valid(self, step, loss, loss_l1, audio_preds):
        writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
        for i in range(len(audio_preds)):
            writer.add_audio('valid/audio_pred_{}'.format(i), audio_preds[i], step, sample_rate=self.params.sample_rate)
        writer.add_scalar('valid/loss', loss, step)
        writer.add_scalar('valid/loss_lsmae', loss_l1, step)
        writer.flush()
        self.summary_writer = writer


def _train_impl(replica_id, model, dataset, dataset_val, args, params):
    torch.backends.cudnn.benchmark = True
    opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

    learner = PriorGradLearner(args.model_dir, model, dataset, dataset_val, opt, params, fp16=args.fp16)
    learner.is_master = (replica_id == 0)
    learner.restore_from_checkpoint()
    learner.train(max_steps=args.max_steps)


def train(args, params, custom_checkpoint=''):
    dataset = dataset_from_path(args.data_root, args.filelist, params)
    dataset_val = dataset_from_path_valid(args.data_root, os.path.join(Path(args.filelist).parent, "valid.txt"), params)
    model = PriorGrad(params).cuda()

    learner = PriorGradLearner(args.model_dir, model, dataset, dataset_val, torch.optim.Adam(model.parameters(), lr=params.learning_rate), params, fp16=args.fp16)
    
    learner.restore_from_checkpoint(custom_checkpoint=custom_checkpoint)
    learner.train(max_steps=args.max_steps)


def train_distributed(replica_id, replica_count, port, args, params, custom_checkpoint=''
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = PriorGrad(params).to(device)
    model = DistributedDataParallel(model, device_ids=[replica_id])
    
    dataset = dataset_from_path(args.data_root, args.filelist, params, is_distributed=True)
    if replica_id == 0:
        dataset_val = dataset_from_path_valid(args.data_root, os.path.join(Path(args.filelist).parent, "valid.txt"), params, is_distributed=False)
    else:
        dataset_val = None
    
    learner = PriorGradLearner(args.model_dir, model, dataset, dataset_val, torch.optim.Adam(model.parameters(), lr=params.learning_rate), params, fp16=args.fp16)
    learner.restore_from_checkpoint(custom_checkpoint=custom_checkpoint)
    
    learner.train(max_steps=args.max_steps)
import os
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils import tensorboard
from torch.distributions import Normal
from torch import nn
import torchaudio
import torchaudio.transforms as T
import time

class ResidualBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            n_inputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs,
            n_outputs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(ConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ResidualBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=dilation_size,  # (kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()

def init(module):
    if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class a2g(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.audio_enc = Audio_Enc(args)
        self.motion_enc = Motion_Enc(args)
        self.motion_dec = Motion_Dec(args)
        self.mapping_net = MappingNet(args)
        self.spectrogram_transform = T.MelSpectrogram(sample_rate=16000,hop_length=1067,n_fft=400,f_min=125,f_max=7500,n_mels=64,center=False)
        self.mean = torch.zeros((10, 256, 16), device="cuda:0") + 0.1
        self.var = torch.zeros((10, 256, 16), device="cuda:0") + 0.1

        self.mean = torch.tensor(np.load('../../data/beat_cache/beat_4english_15_141/weights/mean.npy')).cuda()
        self.var = torch.tensor(np.load('../../data/beat_cache/beat_4english_15_141/weights/var.npy')).cuda()

    def sampling(self, size=None, mean=None, var=None):
        
        normal = Normal(mean, var)
        z_x = normal.sample((size,)) 
        z_x = z_x.permute(1, 0, 2)
        # else:
        # z_x = torch.randn(size, device="cuda:0")
        z_x = self.mapping_net(z_x)

        return z_x

    def forward(self,in_audio: torch.Tensor, pre_seq=None, in_pose=None, in_facial=None, in_text=None, in_id=None, in_emo=None, is_test=False, flag=False):

        if not self.training:
            audios = self.spectrogram_transform(in_audio)
            z_audio_share = self.audio_enc(audios[:, :])
            if in_pose is None:
                idx = random.randint(0, self.mean.shape[0] - 1)
                z_motion_spec = self.sampling(size=z_audio_share.shape[1], mean=self.mean[idx], var=self.var[idx])
            else:
                _, z_motion_spec = self.motion_enc(in_pose[:, :])
            pred_motions = self.motion_dec(z_audio_share, z_motion_spec)

            if flag:
                np.save('mean.npy', self.mean.detach().cpu().numpy())
                np.save('var.npy', self.var.detach().cpu().numpy())
                self.mean = None
                self.var = None

            return pred_motions

        audios = self.spectrogram_transform(in_audio)
        motions = in_pose
        self.z_audio_share = self.audio_enc(audios)
        (self.z_motion_share, self.z_motion_specific,) = self.motion_enc(
            motions
        )
        recon_m = self.motion_dec(self.z_motion_share, self.z_motion_specific)
        a2m = self.motion_dec(self.z_audio_share, self.z_motion_specific)
        self.z_x = self.sampling(
            size=self.z_motion_specific.shape[1],
            mean=self.z_motion_specific.mean(dim=(1,)),
            var=self.z_motion_specific.std(dim=(1,)),
        )
        if self.mean is None:
            self.mean = self.z_motion_specific.mean(dim=(1,)).unsqueeze(0)
        else:
            self.mean = torch.cat((self.mean, self.z_motion_specific.mean(dim=(1,)).unsqueeze(0)), dim=0)

        if self.var is None:
            self.var = self.z_motion_specific.std(dim=(1,)).unsqueeze(0)
        else:
            self.var = torch.cat((self.var, self.z_motion_specific.std(dim=(1,)).unsqueeze(0)), dim=0)

        a2x = self.motion_dec(self.z_audio_share, self.z_x)

        (self.z_a2x_share, self.z_a2x_spec) = self.motion_enc(a2x)
        return recon_m, a2m, a2x

class VAE(nn.Module):
    def __init__(self, args) -> None:
        super(VAE, self).__init__()
        self.global_step = 0

    def reparameterize(cls, mu, logvar):
        eps = torch.randn_like(logvar)
        std = torch.exp(0.5 * logvar)
        return mu + eps * std

    def kl_scheduler(self):
        return max((self.global_step // 10) % 10000 * 0.0001, 0.0001)

    def kl_divergence(cls, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1,))

    def step(self):
        self.global_step += 1

class Audio_Enc(VAE):
    def __init__(self, args):
        super(Audio_Enc, self).__init__(args)
        self.args = args

        self.TCN = ConvNet(
            64, [128, 128, 96, 96, 64], dropout=0
        )
        self.share_mean = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16),
        )
        self.share_var = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16),
        )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        output = self.TCN(inputs).permute(0, 2, 1)
        z_share = self.share_mean(output)

        return z_share

class Motion_Enc(VAE):
    def __init__(self, args):
        super(Motion_Enc, self).__init__(args)
        self.args = args
        input_channel = 47*3
        self.TCN = ConvNet(
            input_channel, [256, 256, 128, 128, 64], dropout=0,
        )
        self.share_linear = nn.Linear(64, 32)
        self.spec_linear = nn.Linear(64, 32)
        self.share_mean = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16),
        )
        self.share_var = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16),
        )
        self.spec_mean = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16),
        )
        self.spec_var = nn.Sequential(
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16),
        )

    def forward(self, inputs: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        output = self.TCN(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        
        share_output = self.share_linear(output)
        spec_output = self.spec_linear(output)

        z_share = self.share_mean(share_output)
        z_specific = self.spec_mean(spec_output)

        return z_share, z_specific

class Motion_Dec(VAE):
    def __init__(self, args):
        super(Motion_Dec, self).__init__(args)
        self.args = args
        output_dim = 3 * 47

        self.TCN = ConvNet(
            32, [64, 128, 128, 256, 256,], dropout=0,
        )
        self.pose_g = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, output_dim),
        )

    def forward(self, share_feature: torch.Tensor, spec_feature: torch.Tensor):
        """
        Args:
            inputs: input tensor of shape: (B, T, C)
        """
        idx = random.randint(0, spec_feature.shape[0] - 1)
        output = torch.cat((share_feature, spec_feature[idx].unsqueeze(0)), dim=2)
        output = self.TCN(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.pose_g(output)
        return output

class MappingNet(VAE):
    def __init__(self, args):
        super(MappingNet, self).__init__(args)
        self.args = args
        hidden_size = 16
        self.net = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
        )
        self.spec_mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.spec_var = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, inputs: torch.Tensor):
        output = self.net(inputs.permute(0, 2, 1)).permute(0, 2, 1)

        self.z_spec_mu = self.spec_mean(output)
        self.z_spec_var = self.spec_var(output)
        z_specific = self.reparameterize(self.z_spec_mu, self.z_spec_var)

        # else:
        #     z_specific = self.spec_mean(output)
        return z_specific

class ConvDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.pose_dims

        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(self.input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            nn.Conv1d(8, 8, 3),
        )

        self.LSTM = nn.LSTM(8, hidden_size=self.hidden_size, num_layers=4, bidirectional=True,
                          dropout=0.3, batch_first=True)
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(34-6, 1)
       
        self.do_flatten_parameters = False
        if torch.cuda.device_count() > 1:
            self.do_flatten_parameters = True

    def forward(self, poses):
        if self.do_flatten_parameters:
            self.LSTM.flatten_parameters()
        poses = poses.transpose(1, 2)
        feat = self.pre_conv(poses)
        feat = feat.transpose(1, 2)
        output, _ = self.LSTM(feat)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  
        batch_size = poses.shape[0]
        output = output.contiguous().view(-1, output.shape[2])
        output = self.out(output)  # apply linear to every output
        output = output.view(batch_size, -1)
        output = self.out2(output)
        output = torch.sigmoid(output)
        return output
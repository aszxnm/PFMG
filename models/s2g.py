import torch
import torch.nn as nn
import os
import pickle
import numpy as np
from torch.nn.utils import weight_norm
from .utils.build_vocab import Vocab
from torch.autograd import Variable
import torch.nn.functional as F
import math

class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    from https://github.com/mlperf/inference/blob/482f6a3beb7af2fb0bd2d91d6185d5e71c22c55f/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class Conv1d_tf(nn.Conv1d):
    """
    Conv1d with the padding behavior from TF
    modified from https://github.com/mlperf/inference/blob/482f6a3beb7af2fb0bd2d91d6185d5e71c22c55f/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py
    """

    def __init__(self, *args, **kwargs):
        super(Conv1d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        if rows_odd:
            input = F.pad(input, [0, rows_odd])

        return F.conv1d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


def ConvNormRelu(in_channels, out_channels, type='1d', downsample=False, k=None, s=None, padding='valid'):
    if k is None and s is None:
        if not downsample:
            k = 3
            s = 1
        else:
            k = 4
            s = 2

    if type == '1d':
        conv_block = Conv1d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
        norm_block = nn.BatchNorm1d(out_channels)
    elif type == '2d':
        conv_block = Conv2d_tf(in_channels, out_channels, kernel_size=k, stride=s, padding=padding)
        norm_block = nn.BatchNorm2d(out_channels)
    else:
        assert False

    return nn.Sequential(
        conv_block,
        norm_block,
        nn.LeakyReLU(0.2, True)
    )


class UnetUp(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UnetUp, self).__init__()
        self.conv = ConvNormRelu(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = torch.repeat_interleave(x1, 2, dim=2)
        x1 = x1[:, :, :x2.shape[2]]  # to match dim
        x = x1 + x2  # it is different to the original UNET, but I stick to speech2gesture implementation
        x = self.conv(x)
        return x

class AudioEncoder(nn.Module):
    def __init__(self, n_frames):
        super().__init__()
        self.n_frames = n_frames
        self.first_net = nn.Sequential(
            ConvNormRelu(1, 64, '2d', False),
            ConvNormRelu(64, 64, '2d', True),
            ConvNormRelu(64, 128, '2d', False),
            ConvNormRelu(128, 128, '2d', True),
            ConvNormRelu(128, 256, '2d', False),
            ConvNormRelu(256, 256, '2d', True),
            ConvNormRelu(256, 256, '2d', False),
            ConvNormRelu(256, 256, '2d', False, padding='valid')
        )

        # self.make_1d = torch.nn.Upsample((n_frames, 1), mode='bilinear', align_corners=False)

        self.down1 = nn.Sequential(
            ConvNormRelu(256, 256, '1d', False),
            ConvNormRelu(256, 256, '1d', False)
        )
        self.down2 = ConvNormRelu(256, 256, '1d', True)
        self.down3 = ConvNormRelu(256, 256, '1d', True)
        self.down4 = ConvNormRelu(256, 256, '1d', True)
        self.down5 = ConvNormRelu(256, 256, '1d', True)
        self.down6 = ConvNormRelu(256, 256, '1d', True)
        self.up1 = UnetUp(256, 256)
        self.up2 = UnetUp(256, 256)
        self.up3 = UnetUp(256, 256)
        self.up4 = UnetUp(256, 256)
        self.up5 = UnetUp(256, 256)

    def forward(self, spectrogram):
        spectrogram = spectrogram.unsqueeze(1)  # add channel dim
        # print(spectrogram.shape)
        spectrogram = spectrogram.float()
        out = self.first_net(spectrogram)
        # out = self.make_1d(out)
        out = F.interpolate(out, size=(int(spectrogram.shape[3]/1066.6), 1), mode ='bilinear')
        x1 = out.squeeze(2)
        x1 = x1.squeeze(3)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)

        return x

class s2g(nn.Module):
    def __init__(self, args, n_poses=34):
        super().__init__()
        self.gen_length = n_poses
        self.pose_dims = args.pose_dims
        self.n_pre_poses = args.pose_length
        self.audio_encoder = AudioEncoder(n_poses)
        self.pre_pose_encoder = nn.Sequential(
            nn.Linear(self.n_pre_poses * (self.pose_dims+1), 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 16)
        )

        self.decoder = nn.Sequential(
            ConvNormRelu(256 + 16, 256),
            ConvNormRelu(256, 256),
            ConvNormRelu(256, 256),
            ConvNormRelu(256, 256)
        )
        self.final_out = nn.Conv1d(256, self.pose_dims, 1, 1)

    def forward(self, pre_seq, in_audio=None, in_facial=None, in_text=None, in_id=None, in_emo=None):
        in_audio = in_audio.unsqueeze(1)
        audio_feat_seq = self.audio_encoder(in_audio)  # output (bs, feat_size, n_frames)
        pre_seq = pre_seq[:, :self.n_pre_poses, :]
        pre_seq = pre_seq.reshape(pre_seq.shape[0], -1)
        # print(pre_seq.shape)
        pre_pose_feat = self.pre_pose_encoder(pre_seq)  # output (bs, 16)
        pre_length = int(in_audio.shape[2] / 1066.6)
        pre_pose_feat = pre_pose_feat.unsqueeze(2).repeat(1, 1, pre_length)

        feat = torch.cat((audio_feat_seq, pre_pose_feat), dim=1)
        out = self.decoder(feat)
        out = self.final_out(out)
        out = out.transpose(1, 2)  # to (batch, seq, dim)

        return out
    
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
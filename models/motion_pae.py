import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from .utils import Utility as utility

class PaeModel(nn.Module):
    def __init__(self, args):
        super(PaeModel, self).__init__()
        self.input_channels = args.pose_dims
        # self.input_channels = 27
        self.embedding_channels = int(args.embedding_channels)
        self.time_range = args.pose_length
        self.window = args.window

        self.tpi = Parameter(torch.from_numpy(np.array([2.0*np.pi], dtype=np.float32)), requires_grad=False)
        self.args = Parameter(torch.from_numpy(np.linspace(-self.window/2, self.window/2, self.time_range, dtype=np.float32)), requires_grad=False)
        self.freqs = Parameter(torch.fft.rfftfreq(self.time_range)[1:] * (self.time_range) / self.window, requires_grad=False) #Remove DC frequency
        
        self.pad = nn.ConstantPad1d((0, 1), 0)

        intermediate_channels = int(self.input_channels/3)
        self.conv1 = nn.Conv1d(self.input_channels, intermediate_channels, self.time_range, stride=1, padding=int((self.time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.norm1 = utility.LN_v2(self.time_range)
        self.conv2 = nn.Conv1d(intermediate_channels, int(self.embedding_channels), self.time_range, stride=1, padding=int((self.time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.conv3 = nn.Conv1d(intermediate_channels, int(self.embedding_channels), self.time_range, stride=1, padding=int((self.time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

        self.fc = torch.nn.ModuleList()
        for i in range(self.embedding_channels):
            self.fc.append(nn.Linear(self.time_range, 2))
        self.deconv1 = nn.Conv1d(self.embedding_channels, intermediate_channels, self.time_range, stride=1, padding=int((self.time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.denorm1 = utility.LN_v2(self.time_range)
        self.deconv2 = nn.Conv1d(intermediate_channels, self.input_channels, self.time_range, stride=1, padding=int((self.time_range - 1) / 2), dilation=1, groups=1, bias=True, padding_mode='zeros')

    #Returns the frequency for a function over a time window in s
    def FFT(self, function, dim):
        rfft = torch.fft.rfft(function, dim=dim)
        magnitudes = rfft.abs()
        spectrum = magnitudes[:,:,1:] #Spectrum without DC component
        power = spectrum**2

        #Frequency
        freq = torch.sum(self.freqs * power, dim=dim) / torch.sum(power, dim=dim)

        #Amplitude
        amp = 2 * torch.sqrt(torch.sum(power, dim=dim)) / self.time_range

        #Offset
        offset = rfft.real[:,:,0] / self.time_range #DC component

        return freq, amp, offset

    def forward(self, x):

        y = torch.transpose(x, 1, 2)
        #Signal Embedding
        y = y.reshape(y.shape[0], self.input_channels, self.time_range)
        y = self.pad(y)
        y = self.conv1(y)
        y = self.norm1(y)
        y = F.elu(y)

        y = self.pad(y)
        z = self.conv3(y)
        y = self.conv2(y)
        latent = y #Save latent for returning

        #Frequency, Amplitude, Offset
        f, a, b = self.FFT(y, dim=2)

        #Phase
        p = torch.empty((y.shape[0], self.embedding_channels), dtype=torch.float32, device=y.device)
        for i in range(self.embedding_channels):
            v = self.fc[i](y[:,i,:])
            p[:,i] = torch.atan2(v[:,1], v[:,0]) / self.tpi

        #Parameters    
        p = p.unsqueeze(2)
        f = f.unsqueeze(2)
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
        params = [p, f, a, b] #Save parameters for returning

        #Latent Reconstruction
        y = a * torch.sin(self.tpi * (f * self.args + p)) + b
        signal = y #Save signal for returning

        #Signal Reconstruction
        y = y + z
        y = self.pad(y)
        y = self.deconv1(y)
        y = self.denorm1(y)
        y = F.elu(y)

        y = self.pad(y)
        y = self.deconv2(y)

        y = y.reshape(y.shape[0], self.time_range, self.input_channels)

        return y, latent, signal, params
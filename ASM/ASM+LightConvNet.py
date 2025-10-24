import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import torch.fft as fft

class LogVarLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out = torch.clamp(x.var(dim=self.dim), 1e-6, 1e6)
        return torch.log(out)


class LightweightConv1d(nn.Module):
    """
    Args:
        input_size: # of channels of the input and output
        kernel_size: convolution channels
        padding: padding
        num_heads: number of heads used. The weight is of shape
            `(num_heads, 1, kernel_size)`
        weight_softmax: normalize the weight with softmax before the convolution
    Shape:
        Input: BxCxT, i.e. (batch_size, input_size, timesteps)
        Output: BxCxT, i.e. (batch_size, input_size, timesteps)
    Attributes:
        weight: the learnable weights of the module of shape
            `(num_heads, 1, kernel_size)`
        bias: the learnable bias of the module of shape `(input_size)`
    """

    def __init__(self, input_size, kernel_size=1, padding=0, num_heads=1, weight_softmax=False, bias=False):
        super().__init__()
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(torch.Tensor(num_heads, 1, kernel_size))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input):
        B, C, T = input.size()  # （150，64，4）
        H = self.num_heads  # 8头

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        input = input.view(-1, H, T)  # （b，8，4）
        output = F.conv1d(input, weight, padding=self.padding, groups=self.num_heads)
        output = output.view(B, C, -1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)

        return output


class block_model(nn.Module):
    """
    Decomposition-Linear
    """

    def __init__(self, input_channels, input_len, out_len):
        super(block_model, self).__init__()
        self.channels = input_channels
        self.input_len = input_len
        self.out_len = out_len

        self.Linear_channel = nn.Linear(self.input_len, self.out_len)
        self.ln = nn.LayerNorm(out_len)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # (B,C,N,T) --> (B,C,N,T)
        output = self.Linear_channel(x)
        return output


class FrequencyMaskingModel(nn.Module):
    def __init__(self):
        super(FrequencyMaskingModel, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.alpha.requires_grad = True

        self.conv1 = nn.Conv2d(9, 9, kernel_size=(22, 1), padding=(0, 0))

        self.attention = nn.MultiheadAttention(embed_dim=1000, num_heads=10, batch_first=True)

        self.fc1 = nn.Linear(1000, 750)
        self.fc2 = nn.Linear(750, 501)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('running_mean', torch.zeros(9, 501))

        self.register_buffer('running_amps', torch.zeros(9, 501))

        self.num_batches_tracked = 0

        self.all_mask_spectrums = []

    def forward(self, data):
        b, freq_bands, channels, length = data.shape

        x = self.relu(self.conv1(data))

        x = x.view(b, 9, 1000)

        attn_output, _ = self.attention(x, x, x)

        x = self.fc1(attn_output)
        x = self.fc2(x)

        x = self.sigmoid(x) * 10
        x_mean = x.mean(dim=0)

        # 更新运行时均值
        self.num_batches_tracked += 1
        eaf = 1.0 / self.num_batches_tracked

        new_running_mean = self.running_mean * (1 - eaf) + eaf * x_mean.detach().clone()
        self.running_mean.copy_(new_running_mean)

        masked_data = torch.zeros_like(data)
        self.all_mask_spectrums = []

        current_batch_amps = []

        for i in range(freq_bands):
            dataset = data[:, i, :, :]

            fft_results = fft.rfft(dataset, dim=2)
            fft_magnitudes = torch.abs(fft_results)

            amps = fft_magnitudes.mean(dim=0).mean(dim=0)  # shape: (501,)
            current_batch_amps.append(amps.detach().clone())


            if self.num_batches_tracked == 1:
                amps_to_use = amps
            else:
                amps_to_use = self.running_amps[i, :]

            conv_output = self.running_mean[i, :]
            scaled_amps = amps_to_use * conv_output

            mask_spectrum = torch.relu(scaled_amps - self.alpha * 1500)
            mask_spectrum = mask_spectrum / (mask_spectrum + 1e-10)

            self.all_mask_spectrums.append(mask_spectrum.clone().detach())

            masked_fft_results = fft_results * mask_spectrum

            ifft_results = fft.irfft(masked_fft_results, n=length, dim=2)

            masked_data[:, i, :, :] = ifft_results

        for i in range(freq_bands):
            new_running_amps = self.running_amps[i, :] * (1 - eaf) + eaf * current_batch_amps[i]
            self.running_amps[i, :].copy_(new_running_amps)

        return masked_data

    def reset_running_stats(self):
        """重置运行时统计量"""
        self.running_mean.zero_()
        self.running_amps.zero_()
        self.num_batches_tracked = 0

    def get_running_amps(self):
        """获取当前的跨试次平均幅度"""
        return self.running_amps.clone()


# Data shape: batch * filterBand * chan * time
class LightConvNet(nn.Module):
    def __init__(self, num_classes=4, num_samples=1000, num_channels=22, num_bands=9, embed_dim=32,
                 win_len=250, num_heads=4, weight_softmax=True, bias=False):
        super().__init__()

        self.win_len = win_len

        self.frequency_masking = FrequencyMaskingModel()

        self.spacial_block = nn.Sequential(nn.Conv2d(num_bands, embed_dim, (num_channels, 1)),
                                           nn.BatchNorm2d(embed_dim),
                                           nn.ELU())

        self.temporal_block = LogVarLayer(dim=3)

        self.conv = LightweightConv1d(embed_dim, (num_samples // win_len), num_heads=num_heads,
                                      weight_softmax=weight_softmax, bias=bias)

        self.classify = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
       # print("shape before masking:", x.shape)
        x = self.frequency_masking(x)
       # print("shape after masking:", x.shape)

        out = self.spacial_block(x)
        out = out.reshape([*out.shape[0:2], -1, self.win_len])
        out = self.temporal_block(out)

        out = self.conv(out)

        out = out.view(out.size(0), -1)
        out = self.classify(out)

        return out

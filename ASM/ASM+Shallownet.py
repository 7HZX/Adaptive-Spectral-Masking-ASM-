from einops.layers.torch import Rearrange
import torch
from torch import nn
import torch.fft as fft

from .classification_module import ClassificationModule
from channel_attention.utils.weight_initialization import glorot_weight_zero_bias
class FrequencyMaskingModel(nn.Module):
    def __init__(self, max_freq_bands=1):
        super(FrequencyMaskingModel, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.alpha.requires_grad = True
        self.max_freq_bands = max_freq_bands

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(22, 1), padding=(0, 0))
        self.attention = nn.MultiheadAttention(embed_dim=875, num_heads=7, batch_first=True)
        self.fc1 = nn.Linear(875, 650)
        self.fc2 = nn.Linear(650, 438)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('running_mean', torch.zeros(1, 438))
        self.register_buffer('running_amps', torch.zeros(max_freq_bands, 438))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.use_cross_batch_stats = True

    def forward(self, data):
        b, freq_bands, channels, length = data.shape
        x = self.relu(self.conv1(data))
        x = x.view(b, 1, 875)
        attn_output, _ = self.attention(x, x, x)
        x = self.fc1(attn_output)
        x = self.fc2(x)
        x = self.sigmoid(x) * 10
        x_mean = x.mean(dim=0)

        self.num_batches_tracked += 1
        eaf = 1.0 / self.num_batches_tracked.float()
        new_running_mean = self.running_mean * (1 - eaf) + eaf * x_mean.detach().clone()
        self.running_mean.copy_(new_running_mean)

        batch_amps = torch.zeros(freq_bands, 438, device=data.device)

        for i in range(freq_bands):
            dataset = data[:, i, :, :]
            fft_results = fft.rfft(dataset, dim=2)
            fft_magnitudes = torch.abs(fft_results)

            current_batch_amps = fft_magnitudes.mean(dim=0).mean(dim=0)
            batch_amps[i, :] = current_batch_amps

        new_running_amps = self.running_amps * (1 - eaf) + eaf * batch_amps.detach().clone()
        self.running_amps.copy_(new_running_amps)

        masked_data = torch.zeros_like(data)

        for i in range(freq_bands):
            dataset = data[:, i, :, :]
            fft_results = fft.rfft(dataset, dim=2)

            if self.use_cross_batch_stats and self.num_batches_tracked > 1:

                amps = self.running_amps[i, :]
            else:

                fft_magnitudes = torch.abs(fft_results)
                amps = fft_magnitudes.mean(dim=0).mean(dim=0)


            conv_output = self.running_mean[0, :]
            scaled_amps = amps * conv_output

            mask_spectrum = torch.relu(scaled_amps - self.alpha * 2000)

            mask_spectrum = mask_spectrum / (mask_spectrum + 1e-10)

            mask_spectrum = mask_spectrum.view(1, 1, -1)

            masked_fft_results = fft_results * mask_spectrum

            ifft_results = fft.irfft(masked_fft_results, n=length, dim=2)
            masked_data[:, i, :, :] = ifft_results

        return masked_data

class ShallowNetModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            n_classes: int,
            input_window_samples: int,
            n_filters_time: int = 40,
            filter_time_length: int = 25,
            pool_time_length: int = 75,
            pool_time_stride: int = 15,
            drop_prob: float = 0.5,
    ):
        super(ShallowNetModule, self).__init__()
        self.rearrange_input = Rearrange("b c t -> b 1 t c")
        self.frequency_masking = FrequencyMaskingModel()
        self.conv_time = nn.Conv2d(1, n_filters_time, (filter_time_length, 1),
                                   bias=True)
        self.conv_spat = nn.Conv2d(n_filters_time, n_filters_time, (1, in_channels),
                                   bias=False)
        self.bnorm = nn.BatchNorm2d(n_filters_time)

        self.pool = nn.AvgPool2d((pool_time_length, 1), (pool_time_stride, 1))
        self.dropout = nn.Dropout(drop_prob)
        out = input_window_samples - filter_time_length + 1
        out = int((out - pool_time_length) / pool_time_stride + 1)

        self.classifier = nn.Sequential(nn.Conv2d(n_filters_time, n_classes, (out, 1)),
                                        Rearrange("b c 1 1 -> b c"))
        glorot_weight_zero_bias(self)

    def forward(self, x):

        x = self.rearrange_input(x)

        x = x.permute(0, 1, 3, 2)
        x = self.frequency_masking(x)
        x = x.permute(0, 1, 3, 2)
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bnorm(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class ShallowNet(ClassificationModule):
    def __init__(
            self,
            in_channels: int,
            n_classes: int,
            input_window_samples: int,
            n_filters_time: int = 40,
            filter_time_length: int = 25,
            pool_time_length: int = 75,
            pool_time_stride: int = 15,
            drop_prob: float = 0.5,
            **kwargs
    ):
        model = ShallowNetModule(
            in_channels=in_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            n_filters_time=n_filters_time,
            filter_time_length=filter_time_length,
            pool_time_length=pool_time_length,
            pool_time_stride=pool_time_stride,
            drop_prob=drop_prob
        )
        super(ShallowNet, self).__init__(model, n_classes, **kwargs)

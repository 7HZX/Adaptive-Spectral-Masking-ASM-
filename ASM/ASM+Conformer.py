"""
EEG Conformer

Convolutional Transformer for EEG decoding

Couple CNN and Transformer in a concise manner with amazing results
"""
# remember to change paths

import argparse
import os
import numpy as np
import math
import glob
import random
import itertools
import datetime
import time
import sys
import scipy.io
import torch.fft as fft

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchsummary import summary
import torch.autograd as autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class FrequencyMaskingModel(nn.Module):
    def __init__(self, max_freq_bands=1):
        super(FrequencyMaskingModel, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.alpha.requires_grad = True
        self.max_freq_bands = max_freq_bands

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(22, 1), padding=(0, 0))
        self.attention = nn.MultiheadAttention(embed_dim=1000, num_heads=10, batch_first=True)
        self.fc1 = nn.Linear(1000, 750)
        self.fc2 = nn.Linear(750, 501)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('running_mean', torch.zeros(1, 501))
        self.register_buffer('running_amps', torch.zeros(max_freq_bands, 501))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self.use_cross_batch_stats = True

    def forward(self, data):
        b, freq_bands, channels, length = data.shape
        x = self.relu(self.conv1(data))
        x = x.view(b, 1, 1000)
        attn_output, _ = self.attention(x, x, x)
        x = self.fc1(attn_output)
        x = self.fc2(x)
        x = self.sigmoid(x) * 10
        x_mean = x.mean(dim=0)

        self.num_batches_tracked += 1
        eaf = 1.0 / self.num_batches_tracked.float()
        new_running_mean = self.running_mean * (1 - eaf) + eaf * x_mean.detach().clone()
        self.running_mean.copy_(new_running_mean)

        batch_amps = torch.zeros(freq_bands, 501, device=data.device)

        for i in range(freq_bands):
            dataset = data[:, i, :, :]
            fft_results = fft.rfft(dataset, dim=2)
            fft_magnitudes = torch.abs(fft_results)

            current_batch_amps = fft_magnitudes.mean(dim=0).mean(dim=0)  # (501,)
            batch_amps[i, :] = current_batch_amps

        new_running_amps = self.running_amps * (1 - eaf) + eaf * batch_amps.detach().clone()
        self.running_amps.copy_(new_running_amps)

        masked_data = torch.zeros_like(data)

        for i in range(freq_bands):
            dataset = data[:, i, :, :]
            fft_results = fft.rfft(dataset, dim=2)

            if self.use_cross_batch_stats and self.num_batches_tracked > 1:

                amps = self.running_amps[i, :]  # (501,)
            else:

                fft_magnitudes = torch.abs(fft_results)
                amps = fft_magnitudes.mean(dim=0).mean(dim=0)  # (501,)


            conv_output = self.running_mean[0, :]
            scaled_amps = amps * conv_output

            mask_spectrum = torch.relu(scaled_amps - self.alpha * 2000)

            mask_spectrum = mask_spectrum / (mask_spectrum + 1e-10)

            mask_spectrum = mask_spectrum.view(1, 1, -1)

            masked_fft_results = fft_results * mask_spectrum

            ifft_results = fft.irfft(masked_fft_results, n=length, dim=2)
            masked_data[:, i, :, :] = ifft_results

        return masked_data


# Convolution module
# use conv to capture local features, instead of postion embedding.
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),  # (288, 40, 22, 976)
            nn.Conv2d(40, 40, (22, 1), (1, 1)),  # (288, 40, 1, 976)
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # (288, 40, 1, 64)。
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.frequency_masking = FrequencyMaskingModel()

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # (288, emb_size=40, 1, 64)。
            # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),  # (288, 64（64 * 1）, emb_size=40)
        )

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape  # b=288
        x = self.frequency_masking(x)
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)  # （b，64，40）->（b，10，64，4）
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)  # （b，64，40）->（b，10，64，4）
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)  # （b，64，40）->（b，h=10，64，dim=4）
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # （b,10,64,4)*(b,10,64,4)->(b,10,64,64)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()

        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(2440, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4, **kwargs):
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )


class ExP():
    def __init__(self, nsub):
        super(ExP, self).__init__()
        self.batch_size = 72
        self.n_epochs = 2000 #2000
        self.c_dim = 4
        self.lr = 0.0002
        self.b1 = 0.5
        self.b2 = 0.999
        self.dimension = (190, 50)
        self.nSub = nsub

        self.start_epoch = 0
        self.root = "./Data/strict_TE/"

        self.log_write = open("./results-new2037/log_subject%d.txt" % self.nSub, "w")

        self.Tensor = torch.cuda.FloatTensor
        self.LongTensor = torch.cuda.LongTensor

        self.criterion_l1 = torch.nn.L1Loss().cuda()
        self.criterion_l2 = torch.nn.MSELoss().cuda()
        self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()

        self.model = Conformer().cuda()
        # Note: gpus variable needs to be defined or this line should be modified
        # self.model = nn.DataParallel(self.model, device_ids=[i for i in range(len(gpus))])
        self.model = nn.DataParallel(self.model)
        self.model = self.model.cuda()

    # Segmentation and Reconstruction (S&R) data augmentation
    def interaug(self, timg, label):
        aug_data = []
        aug_label = []
        for cls4aug in range(4):
            cls_idx = np.where(label == cls4aug + 1)
            tmp_data = timg[cls_idx]
            tmp_label = label[cls_idx]

            tmp_aug_data = np.zeros((int(self.batch_size / 4), 1, 22, 1000))
            for ri in range(int(self.batch_size / 4)):
                for rj in range(8):
                    rand_idx = np.random.randint(0, tmp_data.shape[0], 8)
                    tmp_aug_data[ri, :, :, rj * 125:(rj + 1) * 125] = tmp_data[rand_idx[rj], :, :,
                                                                      rj * 125:(rj + 1) * 125]

            aug_data.append(tmp_aug_data)
            aug_label.append(tmp_label[:int(self.batch_size / 4)])
        aug_data = np.concatenate(aug_data)
        aug_label = np.concatenate(aug_label)
        aug_shuffle = np.random.permutation(len(aug_data))
        aug_data = aug_data[aug_shuffle, :, :]
        aug_label = aug_label[aug_shuffle]

        aug_data = torch.from_numpy(aug_data).cuda()
        aug_data = aug_data.float()
        aug_label = torch.from_numpy(aug_label - 1).cuda()
        aug_label = aug_label.long()
        return aug_data, aug_label

    def get_source_data(self):

        # train data
        self.total_data = scipy.io.loadmat(self.root + 'A0%dT.mat' % self.nSub)
        self.train_data = self.total_data['data']
        self.train_label = self.total_data['label']

        self.train_data = np.transpose(self.train_data, (2, 1, 0))
        self.train_data = np.expand_dims(self.train_data, axis=1)
        self.train_label = np.transpose(self.train_label)

        self.allData = self.train_data
        self.allLabel = self.train_label[0]

        shuffle_num = np.random.permutation(len(self.allData))
        self.allData = self.allData[shuffle_num, :, :, :]
        self.allLabel = self.allLabel[shuffle_num]

        # test data
        self.test_tmp = scipy.io.loadmat(self.root + 'A0%dE.mat' % self.nSub)
        self.test_data = self.test_tmp['data']
        self.test_label = self.test_tmp['label']

        self.test_data = np.transpose(self.test_data, (2, 1, 0))
        self.test_data = np.expand_dims(self.test_data, axis=1)
        self.test_label = np.transpose(self.test_label)

        self.testData = self.test_data
        self.testLabel = self.test_label[0]

        # standardize
        target_mean = np.mean(self.allData)
        target_std = np.std(self.allData)
        self.allData = (self.allData - target_mean) / target_std
        self.testData = (self.testData - target_mean) / target_std

        # data shape: (trial, conv channel, electrode channel, time samples)
        return self.allData, self.allLabel, self.testData, self.testLabel

    def train(self):
        img, label, test_data, test_label = self.get_source_data()

        img = torch.from_numpy(img)
        label = torch.from_numpy(label - 1)

        dataset = torch.utils.data.TensorDataset(img, label)
        self.dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        test_data = torch.from_numpy(test_data)
        test_label = torch.from_numpy(test_label - 1)
        test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                                           shuffle=True)

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        test_data = Variable(test_data.type(self.Tensor))
        test_label = Variable(test_label.type(self.LongTensor))

        bestAcc = 0
        averAcc = 0
        num = 0
        Y_true = 0
        Y_pred = 0

        # Train the model
        total_step = len(self.dataloader)
        curr_lr = self.lr

        for e in range(self.n_epochs):
            self.model.train()
            for i, (img, label) in enumerate(self.dataloader):
                img = Variable(img.cuda().type(self.Tensor))
                label = Variable(label.cuda().type(self.LongTensor))

                # Data augmentation
                aug_data, aug_label = self.interaug(self.allData, self.allLabel)
                img = torch.cat((img, aug_data))
                label = torch.cat((label, aug_label))

                tok, outputs = self.model(img)

                loss = self.criterion_cls(outputs, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Test process
            if (e + 1) % 1 == 0:
                self.model.eval()
                Tok, Cls = self.model(test_data)

                loss_test = self.criterion_cls(Cls, test_label)
                y_pred = torch.max(Cls, 1)[1]
                acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))
                train_pred = torch.max(outputs, 1)[1]
                train_acc = float((train_pred == label).cpu().numpy().astype(int).sum()) / float(label.size(0))

                print('Epoch:', e,
                      '  Train loss: %.6f' % loss.detach().cpu().numpy(),
                      '  Test loss: %.6f' % loss_test.detach().cpu().numpy(),
                      '  Train accuracy %.6f' % train_acc,
                      '  Test accuracy is %.6f' % acc)

                self.log_write.write(str(e) + "    " + str(acc) + "\n")
                num = num + 1
                averAcc = averAcc + acc
                if acc > bestAcc:
                    bestAcc = acc
                    Y_true = test_label
                    Y_pred = y_pred

        if isinstance(self.model, torch.nn.DataParallel):
            torch.save(self.model.module.state_dict(), 'model.pth')
        else:
            torch.save(self.model.state_dict(), 'model.pth')

        averAcc = averAcc / num
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        self.log_write.write('The average accuracy is: ' + str(averAcc) + "\n")
        self.log_write.write('The best accuracy is: ' + str(bestAcc) + "\n")

        return bestAcc, averAcc, Y_true, Y_pred

    def save_mask_spectrum(self, model):
        with open('final_mask_spectrum_values.txt', 'w') as f:
            f.write('Final Mask Spectrum Values:\n')
            for i in range(model.module.frequency_masking.running_mean.shape[1]):
                mask_spectrum = model.module.frequency_masking.running_mean[0, i].detach().cpu().numpy()
                f.write(f'Frequency Band {i + 1}:\n')
                np.savetxt(f, mask_spectrum, fmt='%.6f')
                f.write('\n')


def main():
    best = 0
    aver = 0
    result_write = open("./results-new2037/sub_result.txt", "w")

    for i in range(9):
        starttime = datetime.datetime.now()

        seed_n = 2037
        print('seed is ' + str(seed_n))
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        torch.cuda.manual_seed(seed_n)
        torch.cuda.manual_seed_all(seed_n)

        print('Subject %d' % (i + 1))
        exp = ExP(i + 1)

        bestAcc, averAcc, Y_true, Y_pred = exp.train()
        print('THE BEST ACCURACY IS ' + str(bestAcc))
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'Seed is: ' + str(seed_n) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The best accuracy is: ' + str(bestAcc) + "\n")
        result_write.write('Subject ' + str(i + 1) + ' : ' + 'The average accuracy is: ' + str(averAcc) + "\n")

        endtime = datetime.datetime.now()
        print('subject %d duration: ' % (i + 1) + str(endtime - starttime))
        best = best + bestAcc
        aver = aver + averAcc
        if i == 0:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))

    best = best / 9
    aver = aver / 9

    result_write.write('**The average Best accuracy is: ' + str(best) + "\n")
    result_write.write('The average Aver accuracy is: ' + str(aver) + "\n")
    result_write.close()

if __name__ == "__main__":
    print(time.asctime(time.localtime(time.time())))
    main()
    print(time.asctime(time.localtime(time.time())))
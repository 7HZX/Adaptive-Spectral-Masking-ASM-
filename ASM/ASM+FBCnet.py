#!/usr/bin/env python
# coding: utf-8
"""
All network architectures: FBCNet, EEGNet, DeepConvNet
@author: Ravikiran Mane
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
import torch.fft as fft
current_module = sys.modules[__name__]

debug = False

#%% Deep convnet - Baseline 1
class deepConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize,  *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias= False, max_norm = 2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1,3), stride = (1,3))
            )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(1,outF, kernalSize, padding = 0, max_norm = 2, *args, **kwargs),
                Conv2dWithConstraint(25, 25, (nChan, 1), padding = 0, bias= False, max_norm = 2),
                nn.BatchNorm2d(outF),
                nn.ELU(),
                nn.MaxPool2d((1,3), stride = (1,3))
                )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                Conv2dWithConstraint(inF, outF, kernalSize, max_norm = 0.5,*args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 2, dropoutP = 0.25, *args, **kwargs):
        super(deepConvNet, self).__init__()

        kernalSize = (1,10)
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
            for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

    def forward(self, x):

        x = self.allButLastLayers(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x

#%% EEGNet Baseline 2
class eegNet(nn.Module):
    def initialBlocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding = (0, self.C1 // 2 ), bias =False),
                nn.BatchNorm2d(self.F1),
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                     padding = 0, bias = False, max_norm = 1,
                                     groups=self.F1),
                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1,4), stride = 4),
                nn.Dropout(p = dropoutP))
        block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 22),
                                     padding = (0, 22//2) , bias = False,
                                     groups=self.F1* self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1,1),
                          stride =1, bias = False, padding = 0),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1,8), stride = 8),
                nn.Dropout(p = dropoutP)
                )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                nn.Conv2d(inF, outF, kernalSize, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1,1,nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass = 2,
                 dropoutP = 0.25, F1=8, D = 2,
                 C1 = 125, *args, **kwargs):
        super(eegNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nClass = nClass
        self.nChan = nChan
        self.C1 = C1

        self.firstBlocks = self.initialBlocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)
        self.lastLayer = self.lastBlock(self.F2, nClass, (1, self.fSize[1]))

    def forward(self, x):
        x = self.firstBlocks(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)
    
class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm = True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)

#%% Support classes for FBNet Implementation
class VarLayer(nn.Module):
    '''
    The variance layer: calculates the variance of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(VarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.var(dim = self.dim, keepdim= True)

class StdLayer(nn.Module):
    '''
    The standard deviation layer: calculates the std of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(StdLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.std(dim = self.dim, keepdim=True)

class LogVarLayer(nn.Module):
    '''
    The log variance layer: calculates the log variance of the data along given 'dim'
    (natural logarithm)
    '''
    def __init__(self, dim):
        super(LogVarLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.log(torch.clamp(x.var(dim = self.dim, keepdim= True), 1e-6, 1e6))

class MeanLayer(nn.Module):
    '''
    The mean layer: calculates the mean of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MeanLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.mean(dim = self.dim, keepdim=True)

class MaxLayer(nn.Module):
    '''
    The max layer: calculates the max of the data along given 'dim'
    '''
    def __init__(self, dim):
        super(MaxLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        ma ,ima = x.max(dim = self.dim, keepdim=True)
        return ma

class swish(nn.Module):
    '''
    The swish layer: implements the swish activation function
    '''
    def __init__(self):
        super(swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


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

class FBCNet(nn.Module):
    # just a FBCSP like structure : chan conv and then variance along the time axis
    '''
        FBNet with seperate variance for every 1s. 
        The data input is in a form of batch x 1 x chan x time x filterBand
    '''
    def SCB(self, m, nChan, nBands, doWeightNorm=True, *args, **kwargs):
        '''
        The spatial convolution block
        m : number of sptatial filters.
        nBands: number of bands in the data
        '''
        return nn.Sequential(
                Conv2dWithConstraint(nBands, m*nBands, (nChan, 1), groups= nBands,
                                     max_norm = 2 , doWeightNorm = doWeightNorm,padding = 0),
                nn.BatchNorm2d(m*nBands),
                swish()
                )

    def LastBlock(self, inF, outF, doWeightNorm=True, *args, **kwargs):
        return nn.Sequential(
                LinearWithConstraint(inF, outF, max_norm = 0.5, doWeightNorm = doWeightNorm, *args, **kwargs),
                nn.LogSoftmax(dim = 1))

    def __init__(self, nChan, nTime, nClass = 2, nBands = 9, m = 32,
                 temporalLayer = 'LogVarLayer', strideFactor= 4, doWeightNorm = True, *args, **kwargs):
        super(FBCNet, self).__init__()

        self.nBands = nBands
        self.m = m
        self.strideFactor = strideFactor

        self.frequency_masking = FrequencyMaskingModel()

        # create all the parrallel SCBc
        self.scb = self.SCB(m, nChan, self.nBands, doWeightNorm = doWeightNorm)
        
        # Formulate the temporal agreegator
        self.temporalLayer = current_module.__dict__[temporalLayer](dim = 3)

        # The final fully connected layer
        self.lastLayer = self.LastBlock(self.m*self.nBands*self.strideFactor, nClass, doWeightNorm = doWeightNorm)

    def forward(self, x):

        x = torch.squeeze(x.permute((0,4,2,3,1)), dim = 4)
        #print("xshape:", x.shape)
        x = self.frequency_masking(x)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3]/self.strideFactor)])
        x = self.temporalLayer(x)
        x = torch.flatten(x, start_dim= 1)
        x = self.lastLayer(x)
        return x

    def extract_features(self, x):

        x = torch.squeeze(x.permute((0, 4, 2, 3, 1)), dim=4)
        # print("xshape:", x.shape)
        x = self.frequency_masking(x)
        x = self.scb(x)
        x = x.reshape([*x.shape[0:2], self.strideFactor, int(x.shape[3] / self.strideFactor)])
        x = self.temporalLayer(x)
        out = torch.flatten(x, start_dim=1)

        return out
    
    





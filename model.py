#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py
@desc This file is extracted from the project stream and contains the complete model and training parameters.

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/6/14 11:29   lintean      1.0         None
'''
from scipy.signal import hilbert
import numpy as np
import scipy.io as scio
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as func
from torch.autograd import Function
from torch.autograd import Variable
import math
import random
import sys
import os
from modules.transformer import TransformerEncoder

# data input
# ConType is the acoustic environment of the selected data
# names are the subject data used in this training
ConType = ["No"]
names = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12", "S13", "S14", "S15", "S16", "S17",
         "S18"]

# Training parameters
overlap = 0.5   # Data overlap rate of sliding window
window_length = 140 # The length of the sliding window(The sampling rate is 70Hz)
batch_size = 1
max_epoch = 200 # The maximum number of iterations
isEarlyStop = False # Whether to early stop

# Data parameter
people_number = 18  # Number of participants
eeg_channel = 16
audio_channel = 1
channel_number = eeg_channel + audio_channel * 2    # Number of channels
trail_number = 20   # Number of trials
cell_number = 3500  # Number of data points in the trail
test_percent = 0.1  # Test set ratio
vali_percent = 0.1  # Validation set ratio

conv_eeg_audio_number = 4


class CNN(nn.Module):
    def __init__(self):
        """
        The CMAA model.
        """
        super(CNN, self).__init__()

        self.channel = [16, 16, 16, 16]
        self.ofc_channel = window_length

        self.output_fc = nn.Sequential(
            nn.Linear(self.ofc_channel * 2, self.ofc_channel), nn.ReLU(),
            nn.Linear(self.ofc_channel, 2), nn.Sigmoid()
        )

        self.fc = nn.ModuleList([nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        ) for i in range(2)])

        self.proj_audio = nn.Conv1d(audio_channel, 16, 1, bias=False)
        self.proj_audio2 = nn.Conv1d(audio_channel, 16, 1, bias=False)

        self.cm_attn = nn.ModuleList([TransformerEncoder(
            embed_dim=16,
            num_heads=1,
            layers=5,
            attn_dropout=0,
            relu_dropout=0,
            res_dropout=0,
            embed_dropout=0,
            attn_mask=False
        ) for i in range(conv_eeg_audio_number)])

    def dot(self, a, b):
        """
        Cosine similarity
        @param a: (channel, window_length). Change the number of channels in the front (to 16)
        @param b: (channel, window_length). Change the number of channels in the front (to 16)
        @return: (140)
        """
        temp = torch.tensordot(a, b, dims=[[0], [0]])
        temp = torch.diag(temp)
        norm_a = torch.norm(a, p=2, dim=0)
        norm_b = torch.norm(b, p=2, dim=0)
        temp = temp / (norm_a * norm_b)
        return temp

    def forward(self, x):
        """
        @param x: shape (1, 1, channel(18), window_length)
            The batch size defaults to 1
            18 channels include 2 wav channels and 16 eeg channels via csp
        @return: output shape (1, 2)
        """
        wavA = x[0, 0, 0:1, :]
        wavA = torch.t(wavA).unsqueeze(0)
        eeg = x[0, 0, 1:-1, :]
        eeg = torch.t(eeg).unsqueeze(0)
        wavB = x[0, 0, -1:, :]
        wavB = torch.t(wavB).unsqueeze(0)

        wavA = wavA.transpose(1, 2)
        eeg = eeg.transpose(1, 2)
        wavB = wavB.transpose(1, 2)

        # wav Channel 1 to 16
        wavA = self.proj_audio(wavA)
        wavB = self.proj_audio2(wavB)

        # 4CMA
        # multihead_attention Input shape: (Time, Batch, Channel)
        # wav and eeg shape: (Batch, Channel, Time)
        data = [wavA, eeg, eeg, wavB]
        kv = [eeg, wavA, wavB, eeg]
        # weight = [0 for i in range(conv_eeg_audio_number)]
        for i in range(conv_eeg_audio_number):
            data[i] = data[i].permute(2, 0, 1)
            kv[i] = kv[i].permute(2, 0, 1)
            data[i] = self.cm_attn[i](data[i], kv[i], kv[i])
            data[i] = data[i].permute(1, 2, 0)

        # dot
        # wav and eeg shape: (Batch, Channel, Time)
        data_dot = None
        for i in range(2):
            temp1 = self.dot(data[i * 3].squeeze(0), data[i + 1].squeeze(0))
            temp1 = self.fc[i](temp1.unsqueeze(1))
            data_dot = temp1.squeeze(1) if data_dot is None else torch.cat([data_dot, temp1.squeeze(1)], dim=0)
        output = self.output_fc(data_dot).unsqueeze(0)

        return output


# Model parameters and initialization
myNet = CNN()
lr = 1e-4
optimizer = torch.optim.Adam(myNet.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=5,
#     min_lr=0, eps=0.001)
loss_func = nn.CrossEntropyLoss()

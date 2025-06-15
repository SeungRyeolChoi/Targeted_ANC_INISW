"""
출처: https://github.com/Lu-Baihh/WaveNet-VNNs-for-ANC/blob/main/WaveNet_VNNs/networks.py
"""

import torch.nn.functional as F
import os
import logging
import numpy as np
import pandas as pd
import utils as utils
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm as tqdm

class Causal_Conv1d(nn.Module):
    """
    인과적 1D 합성곱 계층
    입력 신호 과거만을 참조하도록 padding을 적용
    """
    def __init__(self,in_channels,out_channels,kernel_size,stride,bias,dilation=1):
        super(Causal_Conv1d,self).__init__()
        self.kernel_size=kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
            dilation=dilation
        )
    def forward(self,x):
        # 인과적 패딩 적용 (미래 정보 차단)
        x = F.pad(x, (self.dilation*(self.kernel_size-1), 0), mode='constant', value=0)
        return self.conv(x)

class VNN2(nn.Module):
    """
    2차 Volterra Neural Network(VNN) 블록
    선형항(linear term) + 2차 쌍곱항(quadratic term)을 동시에 학습
    복잡한 비선형 신호 특성을 모델링
    """
    def __init__(self, config):
        super(VNN2, self).__init__()
        self.config = config
        self.Q2 = config['VNN2']['Q2']
        self.out_channel = config['VNN2']['conv1d']['out'][0]
        self.conv1 = Causal_Conv1d(
            config['VNN2']['conv1d']['input'][0],
            self.out_channel,
            config['VNN2']['conv1d']['kernel'][0],
            stride=1,
            bias=False
        )
        self.conv2 = Causal_Conv1d(
            config['VNN2']['conv1d']['input'][1],
            2*self.Q2*self.config['VNN2']['conv1d']['out'][1],
            config['VNN2']['conv1d']['kernel'][1],
            stride=1,
            bias=False
        )

    def forward(self, x):
        # 선형항 계산
        linear_term = self.conv1(x)

        x2 = self.conv2(x)
        # 곱셈쌍(pair)으로 분리
        q = self.Q2 * self.out_channel
        a = x2[:, :q, :]
        b = x2[:, q:2*q, :]
        x2_mul = a * b

        # 2차항 누적 합산
        quad_term = torch.zeros_like(linear_term)
        for i in range(self.Q2):
            start = i * self.out_channel
            end = start + self.out_channel
            quad_term += x2_mul[:, start:end, :]

        return (linear_term + quad_term).squeeze()

class dilated_residual_block(nn.Module):
    """
    dilated(확장) 잔차 블록
    WaveNet의 핵심 모듈로, dilation을 조절해 넓은 리셉티브 필드로 과거 정보를 포착
    """
    def __init__(self, dilation, config):
        super().__init__()
        self.config = config
        self.conv1 = Causal_Conv1d(
            config['WaveNet']['Resblock']['conv1d']['res'],
            2*config['WaveNet']['Resblock']['conv1d']['res'],
            kernel_size=config['WaveNet']['Resblock']['conv1d']['kernel'][0],
            stride=1,
            bias=False,
            dilation=dilation
        )
        self.conv2 = Causal_Conv1d(
            config['WaveNet']['Resblock']['conv1d']['res'],
            config['WaveNet']['Resblock']['conv1d']['res'] + config['WaveNet']['Resblock']['conv1d']['skip'],
            kernel_size=config['WaveNet']['Resblock']['conv1d']['kernel'][1],
            stride=1,
            bias=False
        )

    def forward(self, data_x):
        original_x = data_x
        data_out = self.conv1(data_x)
        # 잔차(res)와 게이트(gate)로 분리
        res = self.config['WaveNet']['Resblock']['conv1d']['res']
        x1 = utils.slicing(data_out, slice(0, res), 1)
        x2 = utils.slicing(data_out, slice(res, 2*res), 1)
        # 게이트 활성화 (gated activation)
        data_x = torch.tanh(x1) * torch.sigmoid(x2)
        data_x = self.conv2(data_x)

        res_x  = utils.slicing(data_x, slice(0, res), 1)
        skip_x = utils.slicing(
            data_x,
            slice(res, res + self.config['WaveNet']['Resblock']['conv1d']['skip']),
            1
        )
        # res_x는 잔차 연결, skip_x는 skip 연결로 활용
        return (res_x + original_x, skip_x)

class WaveNet_VNNs(nn.Module):
    """
    WaveNet + Volterra(2차) 블록을 조합한 메인 저감/분류 신경망
    입력 신호를 여러 단계 인과 conv → 여러 dilation residual block → nonlinear post-processing → VNN2(비선형 보상) 구조로 처리
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_stacks = config['WaveNet']['num_stacks']
        dils = config['WaveNet']['dilations']
        if isinstance(dils, int):
            self.dilations = [2**i for i in range(dils+1)]
        else:
            self.dilations = dils

        self.conv1 = Causal_Conv1d(
            config['WaveNet']['conv']['input'][0],
            config['WaveNet']['conv']['out'][0],
            config['WaveNet']['conv']['kernel'][0],
            stride=1,
            bias=False
        )
        self.conv2 = Causal_Conv1d(
            config['WaveNet']['conv']['input'][1],
            config['WaveNet']['conv']['out'][1],
            config['WaveNet']['conv']['kernel'][1],
            stride=1,
            bias=False
        )
        self.conv3 = Causal_Conv1d(
            config['WaveNet']['conv']['input'][2],
            config['WaveNet']['conv']['out'][2],
            config['WaveNet']['conv']['kernel'][2],
            stride=1,
            bias=False
        )
        self.conv4 = Causal_Conv1d(
            config['WaveNet']['conv']['input'][3],
            config['WaveNet']['conv']['out'][3],
            config['WaveNet']['conv']['kernel'][3],
            stride=1,
            bias=False
        )
        self.dilated_layers = nn.ModuleList([
            dilated_residual_block(d, config)
            for d in self.dilations
        ])
        self.VNN = VNN2(config)

    def forward(self, x):
        # 초기 인과 conv
        data_out = self.conv1(x)
        skip_connections = []
        # 여러 스택의 dilated block
        for _ in range(self.num_stacks):
            for layer in self.dilated_layers:
                res, skip = layer(data_out)
                data_out = res
                skip_connections.append(skip)

        # 모든 skip 연결 합산 (메모리 효율 방식)
        out_sum = skip_connections[0].clone()
        for skip in skip_connections[1:]:
            out_sum += skip
        data_out = out_sum

        # 비선형 후처리 및 conv 연속 적용
        data_out = torch.tanh(data_out)
        data_out = self.conv2(data_out); data_out = torch.tanh(data_out)
        data_out = self.conv3(data_out); data_out = torch.tanh(data_out)
        data_out = self.conv4(data_out); data_out = torch.tanh(data_out)
        # 마지막 VNN2(2차 Volterra 블록) 적용
        return self.VNN(data_out).squeeze()

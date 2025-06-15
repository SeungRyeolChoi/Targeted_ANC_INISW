"""!
@brief 인과적이고 단순화된 SuDO-RM-RF 모델

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import torch
import torch.nn as nn
import math

class ScaledWSConv1d(nn.Conv1d):
    """스케일드 가중치 정규화가 적용된 1D 합성곱 계층
    1D 합성곱 계층(Conv1d)에 Causal Masking(인과성 보장)과 가중치 정규화를 추가한 계층
    미래 입력 정보를 보지 않게 가중치(커널)의 일부를 0으로 마스킹"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0,
                 dilation=1, groups=1, bias=True, gain=False,
                 eps=1e-8):
        nn.Conv1d.__init__(self, in_channels, out_channels,
                           kernel_size, stride, padding, dilation,
                           groups, bias)
        self.causal_mask = torch.ones_like(self.weight)
        if kernel_size >= 3:
            future_samples = kernel_size // 2
            self.causal_mask[..., -future_samples:] = 0.

    def get_weight(self):
        return self.weight * self.causal_mask.to(self.weight.device)

    def forward(self, x):
        return nn.functional.conv1d(
            x, self.get_weight(), self.bias,
            self.stride, self.padding, self.dilation, self.groups)

class ConvAct(nn.Module):
    '''
    이 클래스는 노멀라이즈된 출력을 가지는 dilation conv를 정의
    1D 합성곱(ScaledWSConv1d) + PReLU 활성화 함수로 구성된 작은 블록
    실질적으로 특성 추출 및 비선형성(활성화) 부여
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: 입력 채널 수
        :param nOut: 출력 채널 수
        :param kSize: 커널 크기
        :param stride: 다운샘플링을 위한 선택적 스트라이드
        :param d: 선택적 dilation 계수
        '''
        super().__init__()
        self.conv = ScaledWSConv1d(nIn, nOut, kSize, stride=stride,
                                   padding=((kSize - 1) // 2), groups=groups)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        return self.act(output)

class UConvBlock(nn.Module):
    '''
    이 클래스는 입력 특징을 다양한 해상도에서 분석할 수 있도록
    다운샘플링과 업샘플링을 연속적으로 수행하는 블록을 정의.
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=5,
                 alpha=1.,
                 beta=1.,):
        super().__init__()
        self.beta, self.alpha = beta, alpha
        self.skipinit_gain = nn.Parameter(torch.zeros(()))
        self.proj_1x1 = ConvAct(out_channels, in_channels, 1,
                                stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(ConvAct(in_channels, in_channels, kSize=21,
                                   stride=1, groups=in_channels))

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(ConvAct(in_channels, in_channels,
                                       kSize=21,
                                       stride=stride,
                                       groups=in_channels))
        if upsampling_depth > 1:
            self.upsampler = torch.nn.Upsample(scale_factor=2)
        self.res_conv = ScaledWSConv1d(in_channels, out_channels, 1)

    def forward(self, x):
        '''
        :param x: 입력 feature map
        :return: 변환된 feature map
        '''
        residual = x.clone()
        # Reduce --> 고차원 feature map을 저차원 공간으로 투영
        output1 = self.proj_1x1(x / self.beta)
        output = [self.spp_dw[0](output1)]

        # 이전 레벨에서 다운샘플링 수행
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # 역순으로 결합 (업샘플링)
        for _ in range(self.depth-1):
            resampled_out_k = self.upsampler(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k

        return self.res_conv(output[-1]) * self.skipinit_gain * self.alpha + residual

class CausalSuDORMRF(nn.Module):
    """
    전체 인과적 SuDORMRF 네트워크(오디오 분리 모델)
    Encoder(전처리) → Bottleneck → 여러 개의 UConvBlock(Separation) → Mask Estimation → Decoder(후처리)
    입력(혼합음)을 여러 개의 분리 신호(음원)로 변환
    """
    def __init__(self,
                 in_audio_channels=1,
                 out_channels=128,
                 in_channels=256,
                 num_blocks=16,
                 upsampling_depth=5,
                 enc_kernel_size=21,
                 enc_num_basis=256,
                 num_sources=2):
        super(CausalSuDORMRF, self).__init__()

        # 생성할 소스의 수
        self.in_audio_channels = in_audio_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # 임의의 길이 입력에 적합한 패딩 필요
        assert self.enc_kernel_size % 2, (
            '신호처리를 고려해 필터 크기는 홀수로 지정하세요. '
            'hop size는 짝수가 될 것이기 때문입니다.')
        self.n_least_samples_req = self.enc_kernel_size // 2 * 2 ** self.upsampling_depth

        # 프론트엔드
        self.encoder = ScaledWSConv1d(in_channels=in_audio_channels,
                                      out_channels=enc_num_basis,
                                      kernel_size=enc_kernel_size * 2 - 1,
                                      stride=enc_kernel_size // 2,
                                      padding=(enc_kernel_size * 2 - 1) // 2,
                                      bias=False)
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # 나머지 계층 전, 한번 더 dense layer 적용
        self.bottleneck = ScaledWSConv1d(
            in_channels=enc_num_basis,
            out_channels=out_channels,
            kernel_size=1)

        # 분리 모듈
        uconv_layers = []
        expected_var = 1.0
        alpha = 1.
        for _ in range(num_blocks):
            beta = expected_var ** 0.5
            uconv_layers.append(
                UConvBlock(out_channels=out_channels,
                           in_channels=in_channels,
                           upsampling_depth=upsampling_depth,
                           alpha=alpha,
                           beta=beta))
        self.sm = nn.Sequential(*uconv_layers)

        mask_conv = ScaledWSConv1d(
            out_channels, num_sources * enc_num_basis * in_audio_channels, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # 백엔드
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources * in_audio_channels,
            out_channels=num_sources * in_audio_channels,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=1, bias=False)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.PReLU()
    # 순전파
    def forward(self, input_wav):

        # 프론트엔드
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # 분리 모듈
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0],
                   self.num_sources * self.in_audio_channels,
                   self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        
        # 백엔드
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        return self.remove_trailing_zeros(estimated_waveforms, input_wav)

    def pad_to_appropriate_length(self, x):
        input_length = x.shape[-1]
        if input_length < self.n_least_samples_req:
            values_to_pad = self.n_least_samples_req
        else:
            res = 1 if input_length % self.n_least_samples_req else 0
            least_number_of_pads = input_length // self.n_least_samples_req
            values_to_pad = (least_number_of_pads + res) * self.n_least_samples_req

        padded_x = torch.zeros(list(x.shape[:-1]) + [values_to_pad], dtype=torch.float32)
        padded_x[..., :x.shape[-1]] = x
        return padded_x.to(x.device)

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]

if __name__ == "__main__":
    in_audio_channels = 2
    model = CausalSuDORMRF(in_audio_channels=in_audio_channels,
                           out_channels=256,
                           in_channels=384,
                           num_blocks=16,
                           upsampling_depth=5,
                           enc_kernel_size=21,
                           enc_num_basis=256,
                           num_sources=2)

    fs = 44100
    timelength = 1.
    timesamples = int(fs * timelength)
    batch_size = 1
    dummy_input = torch.rand(batch_size, in_audio_channels, timesamples)

    estimated_sources = model(dummy_input)
    print(estimated_sources.shape)
    assert estimated_sources.shape[-1] == dummy_input.shape[-1]

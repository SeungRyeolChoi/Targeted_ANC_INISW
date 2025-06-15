"""
출처: https://github.com/Lu-Baihh/WaveNet-VNNs-for-ANC/blob/main/WaveNet_VNNs/loss_function.py
"""

import torch
import math
import torch.nn as nn

def pwelch(x, fs=3125, nperseg=512, noverlap=341, nfft=512, window='hamming'):
    """Welch 방법을 이용한 파워 스펙트럼 밀도(PSD) 추정 함수"""
    
    step = nperseg - noverlap
    device = x.device
    # 윈도우 생성
    if window == 'hann':
        window = torch.hann_window(nperseg, periodic=True, dtype=x.dtype, device=device)
    elif window == 'hamming':
        window = torch.hamming_window(nperseg, periodic=True, dtype=x.dtype, device=device)
    # 오버랩된 세그먼트로 분할
    segments = x.unfold(0, nperseg, step)
    # 각 세그먼트에 윈도우 적용
    segments = segments * window
    # FFT 및 파워 스펙트럼 계산
    fft_segments = torch.fft.fft(segments, nfft, dim=-1)
    psd = torch.abs(fft_segments[:, :nfft // 2 + 1]) ** 2
    # MATLAB 방식과 동일하게 PSD 스케일링
    win_power = (window**2).sum()
    psd /= win_power * fs
    psd[:, 1:-1] *= 2  # 단측 스펙트럼 보정
    # 모든 세그먼트에 대해 평균
    psd_mean = psd.mean(dim=0)
    # 주파수 벡터 생성
    freqs = torch.fft.fftfreq(nfft, 1 / fs)[:nfft // 2 + 1]
    return freqs, psd_mean

def compute_a_weighting(frequencies):
    """
    A-가중치(A-weighting) 스펙트럼 계산 함수
    사람 귀의 민감도를 반영해 주파수별로 가중치를 부여
    """
    f1 = 20.6
    f2 = 107.7
    f3 = 737.9
    f4 = 12194.0
    A1000 = -2.0
    num = (f4 ** 2) * (frequencies ** 4)
    den = (frequencies ** 2 + f1 ** 2) * torch.sqrt(frequencies ** 2 + f2 ** 2) * torch.sqrt(frequencies ** 2 + f3 ** 2) * (frequencies ** 2 + f4 ** 2)
    A = 20.0 * torch.log10(num / den) - A1000
    return A

class dBA_Loss(nn.Module):
    """dBA 손실 함수
    오디오 신호의 파워 스펙트럼에 대해 A-weighting을 적용
    사람이 듣는 청감 특성을 반영한 전체 에너지를 계산
    """
    def __init__(self, fs, nfft, f_up, f_low=1):
        super(dBA_Loss, self).__init__()
        self.fs = fs
        self.nfft = nfft
        self.f_up = f_up
        self.f_low = f_low

    def forward(self, x):
        batch_size = x.shape[0]
        loss = 0.0
        for i in range(batch_size):
            f, pxx = pwelch(x[i, :], self.fs, nperseg=512, noverlap=341, nfft=512, window='hamming')
            A_weighting = compute_a_weighting(f)
            f_resolution = self.fs / self.nfft
            Lev_f_up = math.floor(self.f_up / f_resolution + 1)
            Lev_f_low = math.floor(self.f_low / f_resolution + 1)
            pxy_dBA = 10 * torch.log10(pxx) + A_weighting.to(x.device)
            level_A = 10 * torch.log10(torch.sum(10 ** (pxy_dBA[Lev_f_low-1:Lev_f_up] / 10), dim=0))
            loss += torch.sum(level_A)
        return loss / batch_size  

class NMSE_Loss(nn.Module):
    """NMSE 손실 함수
    예측(enhanced)과 정답(denoised) 신호의 파워(제곱합) 비율을 dB 단위로 비교"""
    def __init__(self):
        super(NMSE_Loss, self).__init__()

    def forward(self, en, dn):
        return 10 * torch.log10(torch.sum((en.squeeze())**2) / torch.sum((dn.squeeze())**2))

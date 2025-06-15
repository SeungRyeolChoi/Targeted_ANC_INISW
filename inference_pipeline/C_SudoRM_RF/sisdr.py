"""!
@brief Torch에서 SISNR을 매우 효율적으로 계산하는 코드. asteroid에서 값 검증을 위해 일부 코드가 차용됨.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import torch
import torch.nn as nn
import itertools
from torch.nn.modules.loss import _Loss

def _sdr(y, z, SI=False):
    # 두 신호 y(정답), z(추정치)의 SDR 또는 SI-SDR을 계산하는 내부 함수.
    if SI:
        a = ((z*y).mean(-1) / (y*y).mean(-1)).unsqueeze(-1) * y
        return 10*torch.log10( (a**2).mean(-1) / ((a-z)**2).mean(-1))
    else:
        return 10*torch.log10( (y*y).mean(-1) / ((y-z)**2).mean(-1))

def sdri_loss(y, z, of=0):
    # SDRi (Signal-to-Distortion Ratio improvement) 손실 함수.
    # SDR 점수의 음수(-SDR)로 반환(최소화 목적).
    if len(y.shape) < 3:
        y = y.unsqueeze(0)
    if len(z.shape) < 3:
        z = z.unsqueeze(0)
    s = _sdr(y, z, SI=False) - of
    return -s.mean()

def sisdr_loss(y, z, of=0):
    # SI-SDRi (Scale-Invariant SDR improvement) 손실 함수.
    # SI-SDR 점수의 음수(-SI-SDR)로 반환(최소화 목적).
    if len(y.shape) < 3:
        y = y.unsqueeze(0)
    if len(z.shape) < 3:
        z = z.unsqueeze(0)
    s = _sdr(y, z, SI=True) - of
    return -s.mean()

def pit_loss(y, z, of=0, SI=False):
    """
    Permutation Invariant Training(PIT) 손실.
    두 소스(정답, 추정치) 간의 모든 permutation에서 SDR 또는 SI-SDR을 계산,
    가장 높은 SDR permutation의 음수(-SDR)를 반환.
    """
    if len(y.shape) < 3:
        y = y.unsqueeze(0)
    if len(z.shape) < 3:
        z = z.unsqueeze(0)
    p = list(itertools.permutations(range(y.shape[-2])))
    s = torch.stack([_sdr(y[:, j, :], z, SI) for j in p], dim=2)
    s = s.mean(1)
    i = s.argmax(-1)
    j = torch.arange(s.shape[0], dtype=torch.long, device=i.device)
    return -s[j, i].mean()

class PermInvariantSISDR(nn.Module):
    """!
    PIT 기반 SI-SDR 계산 클래스.
    복수 소스 추정 시 각 permutation을 자동으로 고려해 best SI-SDR을 선택.
    """

    def __init__(self,
                 batch_size=None,
                 zero_mean=False,
                 n_sources=None,
                 backward_loss=True,
                 improvement=False,
                 return_individual_results=False):
        """
        결과 및 이후에 사용할 torch tensor들을 위한 초기화
        :param batch_size: 각 batch의 샘플 개수
        :param zero_mean: SI-SDR 계산 전 신호의 time 축에 대해 zero-mean 수행 여부
        """
        super().__init__()
        self.bs = batch_size
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.permutations = list(itertools.permutations(
            torch.arange(n_sources)))
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.improvement = improvement
        self.n_sources = n_sources
        self.return_individual_results = return_individual_results

    def normalize_input(self, pr_batch, t_batch, initial_mixtures=None):
        # 입력 신호 길이 맞춤, (옵션) zero-mean 적용
        min_len = min(pr_batch.shape[-1], t_batch.shape[-1])
        if initial_mixtures is not None:
            min_len = min(min_len, initial_mixtures.shape[-1])
            initial_mixtures = initial_mixtures[:, :, :min_len]
        pr_batch = pr_batch[:, :, :min_len]
        t_batch = t_batch[:, :, :min_len]

        if self.perform_zero_mean:
            pr_batch = pr_batch - torch.mean(pr_batch, dim=-1, keepdim=True)
            t_batch = t_batch - torch.mean(t_batch, dim=-1, keepdim=True)
            if initial_mixtures is not None:
                initial_mixtures = initial_mixtures - torch.mean(
                    initial_mixtures, dim=-1, keepdim=True)
        return pr_batch, t_batch, initial_mixtures

    @staticmethod
    def dot(x, y):
        # 두 신호의 내적
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_permuted_sisnrs(self,
                                permuted_pr_batch,
                                t_batch,
                                t_t_diag, eps=10e-8):
        # 각 permutation별 SI-SNR 계산 (projection 기반)
        s_t = (self.dot(permuted_pr_batch, t_batch) /
               (t_t_diag + eps) * t_batch)
        e_t = permuted_pr_batch - s_t
        sisnrs = 10 * torch.log10(self.dot(s_t, s_t) /
                                  (self.dot(e_t, e_t) + eps))
        return sisnrs

    def compute_sisnr(self,
                      pr_batch,
                      t_batch,
                      initial_mixtures=None,
                      eps=10e-8):
        # 전체 permutation SI-SNR 계산 후, best permutation 선택
        t_t_diag = self.dot(t_batch, t_batch)

        sisnr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            sisnr = self.compute_permuted_sisnrs(permuted_pr_batch,
                                                 t_batch,
                                                 t_t_diag, eps=eps)
            sisnr_l.append(sisnr)
        all_sisnrs = torch.cat(sisnr_l, -1)
        best_sisdr, best_perm_ind = torch.max(all_sisnrs.mean(-2), -1)

        if self.improvement:
            initial_mix = initial_mixtures.repeat(1, self.n_sources, 1)
            base_sisdr = self.compute_permuted_sisnrs(initial_mix,
                                                      t_batch,
                                                      t_t_diag, eps=eps)
            best_sisdr -= base_sisdr.mean()

        if not self.return_individual_results:
            best_sisdr = best_sisdr.mean()

        if self.backward_loss:
            return -best_sisdr, best_perm_ind
        return best_sisdr, best_perm_ind

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-9,
                initial_mixtures=None,
                return_best_permutation=False):
        """!
        :param pr_batch: 추정된 wavs (batch_size x self.n_sources x 길이)
        :param t_batch: 정답 wavs (batch_size x self.n_sources x 길이)
        :param eps: 수치 안정성을 위한 상수
        :param initial_mixtures: SISDRi 계산을 위한 초기 mixture (batch_size x 1 x 길이)
        :returns: 결과 tensor (batch_size x 1), best permutation index 등
        """
        pr_batch, t_batch, initial_mixtures = self.normalize_input(
            pr_batch, t_batch, initial_mixtures=initial_mixtures)

        sisnr_l, best_perm_ind = self.compute_sisnr(
            pr_batch, t_batch, eps=eps,
            initial_mixtures=initial_mixtures)

        if return_best_permutation:
            best_permutations = self.permutations_tensor[best_perm_ind]
            return sisnr_l, best_permutations
        else:
            return sisnr_l

# 이하 asteroid 라이브러리에서 복사된 PITLossWrapper 클래스
# 내부 docstring만 유지, 나머지는 별도 번역 생략
class PITLossWrapper(nn.Module):
    """ Permutation invariant loss wrapper. """
    def __init__(self, loss_func, pit_from='pw_mtx', perm_reduce=None):
        super().__init__()
        self.loss_func = loss_func
        self.pit_from = pit_from
        self.perm_reduce = perm_reduce
        if self.pit_from not in ['pw_mtx', 'pw_pt', 'perm_avg']:
            raise ValueError('Unsupported loss function type for now. Expected'
                             'one of [`pw_mtx`, `pw_pt`, `perm_avg`]')

    def forward(self, est_targets, targets, return_est=False,
                reduce_kwargs=None, **kwargs):
        n_src = targets.shape[1]
        assert n_src < 10, f"Expected source axis along dim 1, found {n_src}"
        if self.pit_from == 'pw_mtx':
            pw_losses = self.loss_func(est_targets, targets, **kwargs)
        elif self.pit_from == 'pw_pt':
            pw_losses = self.get_pw_losses(self.loss_func, est_targets,
                                           targets, **kwargs)
        elif self.pit_from == 'perm_avg':
            min_loss, min_loss_idx = self.best_perm_from_perm_avg_loss(
                self.loss_func, est_targets, targets, **kwargs
            )
            mean_loss = torch.mean(min_loss)
            if not return_est:
                return mean_loss
            reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
            return mean_loss, reordered
        else:
            return

        assert pw_losses.ndim == 3, ("Something went wrong with the loss "
                                     "function, please read the docs.")
        assert (pw_losses.shape[0] ==
                targets.shape[0]), "PIT loss needs same batch dim as input"

        reduce_kwargs = reduce_kwargs if reduce_kwargs is not None else dict()
        min_loss, min_loss_idx = self.find_best_perm(
            pw_losses, n_src, perm_reduce=self.perm_reduce, **reduce_kwargs
        )
        mean_loss = torch.mean(min_loss)
        if not return_est:
            return mean_loss
        reordered = self.reorder_source(est_targets, n_src, min_loss_idx)
        return mean_loss, reordered

    @staticmethod
    def get_pw_losses(loss_func, est_targets, targets, **kwargs):
        batch_size, n_src, *_ = targets.shape
        pair_wise_losses = targets.new_empty(batch_size, n_src, n_src)
        for est_idx, est_src in enumerate(est_targets.transpose(0, 1)):
            for target_idx, target_src in enumerate(targets.transpose(0, 1)):
                pair_wise_losses[:, est_idx, target_idx] = loss_func(
                    est_src, target_src, **kwargs)
        return pair_wise_losses

    @staticmethod
    def find_best_perm(pair_wise_losses, n_src, perm_reduce=None, **kwargs):
        pwl = pair_wise_losses.transpose(-1, -2)
        perms = pwl.new_tensor(list(itertools.permutations(range(n_src))),
                               dtype=torch.long)
        idx = torch.unsqueeze(perms, 2)
        if perm_reduce is None:
            perms_one_hot = pwl.new_zeros((*perms.size(), n_src)).scatter_(2,
                                                                           idx,
                                                                           1)
            loss_set = torch.einsum('bij,pij->bp', [pwl, perms_one_hot])
            loss_set /= n_src
        else:
            batch = pwl.shape[0]
            n_perm = idx.shape[0]
            pwl_set = pwl[:, torch.arange(n_src), idx.squeeze(-1)]
            loss_set = perm_reduce(pwl_set, **kwargs)
        min_loss_idx = torch.argmin(loss_set, dim=1)
        min_loss, _ = torch.min(loss_set, dim=1, keepdim=True)
        return min_loss, min_loss_idx

class PairwiseNegSDR(_Loss):
    """
    배치 단위로 SI-SDR, SD-SDR, SNR 등 음질 평가를 위한 pairwise 음수 손실 함수 base class.
    입력: (batch, n_src, time) 크기의 정답/추정치
    출력: (batch, n_src, n_src) pairwise loss matrix
    """
    def __init__(self, sdr_type, zero_mean=True, take_log=True):
        super(PairwiseNegSDR, self).__init__()
        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log

    def forward(self, est_targets, targets):
        assert targets.size() == est_targets.size()
        # 1단계: zero-mean 정규화
        if self.zero_mean:
            mean_source = torch.mean(targets, dim=2, keepdim=True)
            mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
            targets = targets - mean_source
            est_targets = est_targets - mean_estimate
        # 2단계: pairwise SI-SDR
        s_target = torch.unsqueeze(targets, dim=1)
        s_estimate = torch.unsqueeze(est_targets, dim=2)
        if self.sdr_type in ["sisdr", "sdsdr"]:
            pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)
            s_target_energy = torch.sum(s_target**2, dim=3, keepdim=True) + 1e-8
            pair_wise_proj = pair_wise_dot * s_target / s_target_energy
        else:
            pair_wise_proj = s_target.repeat(1, s_target.shape[2], 1, 1)
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = s_estimate - s_target
        else:
            e_noise = s_estimate - pair_wise_proj
        pair_wise_sdr = torch.sum(pair_wise_proj ** 2, dim=3) / (
                torch.sum(e_noise ** 2, dim=3) + 1e-8)
        if self.take_log:
            pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + 1e-8)
        return - pair_wise_sdr

class StabilizedPermInvSISDRMetric(nn.Module):
    """
    복수 소스, permutation 상황에서 numerically stable SISDR metric 계산 클래스.
    주로 음원 분리/재구성 성능 평가에 사용.
    """

    def __init__(self,
                 zero_mean=False,
                 single_source=False,
                 n_estimated_sources=None,
                 n_actual_sources=None,
                 backward_loss=True,
                 improvement=False,
                 return_individual_results=False):
        """
        결과와 이후에 사용될 torch tensor들을 위한 초기화
        :param zero_mean: SDR 계산 전 zero-mean 수행 여부
        :param single_source: 단일 소스 평가용 여부
        :param n_estimated_sources: 추정 소스 개수
        :param n_actual_sources: 실제 소스 개수
        """
        super().__init__()
        self.perform_zero_mean = zero_mean
        self.backward_loss = backward_loss
        self.improvement = improvement
        self.n_estimated_sources = n_estimated_sources
        self.n_actual_sources = n_actual_sources
        assert self.n_estimated_sources >= self.n_actual_sources, (
            '추정 소스 개수는 실제 소스 개수 이상이어야 합니다: {} but got: {}'.format(
                self.n_actual_sources, self.n_estimated_sources))
        self.permutations = list(itertools.permutations(
            torch.arange(self.n_estimated_sources),
            r=self.n_actual_sources))
        self.permutations_tensor = torch.LongTensor(self.permutations)
        self.return_individual_results = return_individual_results
        self.single_source = single_source
        if self.single_source:
            assert self.n_actual_sources == 1

    def normalize_input(self, input_tensor):
        # 입력 신호를 zero-mean 여부에 따라 정규화
        if self.perform_zero_mean:
            return input_tensor - torch.mean(input_tensor, dim=-1, keepdim=True)
        else:
            return input_tensor

    @staticmethod
    def dot(x, y):
        # 두 신호의 내적
        return torch.sum(x * y, dim=-1, keepdim=True)

    def compute_stabilized_sisnr(self,
                                 permuted_pr_batch,
                                 t_batch,
                                 t_signal_powers, eps=1e-8):
        # numerically stable 버전의 SI-SNR 계산
        pr_signal_powers = self.dot(permuted_pr_batch, permuted_pr_batch)
        inner_prod_sq = self.dot(permuted_pr_batch, t_batch) ** 2
        rho_sq = inner_prod_sq / (pr_signal_powers * t_signal_powers + eps)
        return 10 * torch.log10((rho_sq + eps) / (1. - rho_sq + eps))

    def compute_sisnr(self,
                      pr_batch,
                      t_batch,
                      eps=1e-8):
        assert t_batch.shape[-2] == self.n_actual_sources
        t_signal_powers = self.dot(t_batch, t_batch)

        sisnr_l = []
        for perm in self.permutations:
            permuted_pr_batch = pr_batch[:, perm, :]
            sisnr = self.compute_stabilized_sisnr(
                permuted_pr_batch, t_batch, t_signal_powers, eps=eps)
            sisnr_l.append(sisnr)
        all_sisnrs = torch.cat(sisnr_l, -1)
        best_sisdr, best_perm_ind = torch.max(all_sisnrs.mean(-2), -1)

        if self.improvement:
            initial_mixture = torch.sum(t_batch, -2, keepdim=True)
            initial_mixture = self.normalize_input(initial_mixture)
            initial_mix = initial_mixture.repeat(1, self.n_actual_sources, 1)
            base_sisdr = self.compute_stabilized_sisnr(
                initial_mix, t_batch, t_signal_powers, eps=eps)
            best_sisdr -= base_sisdr.mean()

        if not self.return_individual_results:
            best_sisdr = best_sisdr.mean()

        if self.backward_loss:
            return -best_sisdr, best_perm_ind
        return best_sisdr, best_perm_ind

    def forward(self,
                pr_batch,
                t_batch,
                eps=1e-9,
                return_best_permutation=False):
        """!
        :param pr_batch: 추정된 wavs (batch_size x self.n_sources x 길이)
        :param t_batch: 정답 wavs (batch_size x self.n_sources x 길이)
        :param eps: 수치 안정성을 위한 상수
        :returns: 결과 tensor (batch_size x 1), best permutation index 등
        """
        if self.single_source:
            pr_batch = torch.sum(pr_batch, -2, keepdim=True)

        pr_batch = self.normalize_input(pr_batch)
        t_batch = self.normalize_input(t_batch)

        sisnr_l, best_perm_ind = self.compute_sisnr(
            pr_batch, t_batch, eps=eps)
        if return_best_permutation:
            best_permutations = self.permutations_tensor[best_perm_ind]
            return sisnr_l, best_permutations
        else:
            return sisnr_l

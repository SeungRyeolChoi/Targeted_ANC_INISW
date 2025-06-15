"""
End-to-End 안내방송/소음 분리·저감 파이프라인 추론 스크립트
- C-SuDoRM-RF++ 기반 분리모델, ASC, WaveNet-VNNs ANC모델을 조합하여
  혼합 오디오에서 안내방송 유지+소음 저감 최종 파형을 생성
- 각 단계별 분리/분류/저감 결과 및 주요 지표(dBA, NMSE, ms/sec) 계산/저장

@author
"""

import os
import sys
import argparse
import time
import glob
import json

import torch
import numpy as np
import soundfile as sf
from scipy.io import loadmat

# ─── 경로 설정 ─────────────────────────────────────────────────────────
# WaveNet-VNNs 모듈 경로
sys.path.append('/content/drive/MyDrive/inference_pipeline/WaveNet_VNNs')
# sudo_rm_rf 분리 모델 경로
# sys.path.append('/content/drive/MyDrive/sudo_rm_rf')

from C_SudoRM_RF.causal_improved_sudormrf_v3 import CausalSuDORMRF
import C_SudoRM_RF.mixture_consistency as mixture_consistency
from networks import VNN2, Causal_Conv1d, WaveNet_VNNs
from utils    import fir_filter, SEF
from loss_function import dBA_Loss
from ASC.ASC import ASC

#ANC 공통 설정(코랩 기준으로 경로 설정)
PRI_PATH = "/content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/channel/pri_channel.mat"
SEC_PATH = "/content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/channel/sec_channel.mat"
PRI      = torch.tensor(loadmat(PRI_PATH)["pri_channel"].squeeze(), dtype=torch.float)
SEC      = torch.tensor(loadmat(SEC_PATH)["sec_channel"].squeeze(), dtype=torch.float)

ETA, SR  = 0.1, 16000
SEG_SEC  = 10
SEG_LEN  = SEG_SEC * SR

def load_separation_model(ckpt_path, hp):
    """
    C-SuDORM-RF++ 분리 모델 생성 및 체크포인트 로드
    """
    model = CausalSuDORMRF(
        in_audio_channels=1,
        out_channels=hp['out_channels'],
        in_channels=hp['in_channels'],
        num_blocks=hp['num_blocks'],
        upsampling_depth=hp['upsampling_depth'],
        enc_kernel_size=hp['enc_kernel_size'],
        enc_num_basis=hp['enc_num_basis'],
        num_sources=hp['max_num_sources']
    ).to(hp['device'])
    sd = torch.load(ckpt_path, map_location=hp['device'])
    model.load_state_dict(sd)
    model.eval()
    return model

def load_noise_model(config_path, model_path, device):
    """
    WaveNet-VNNs ANC 모델 로드 (config+파라미터)
    """
    cfg   = json.load(open(config_path, "r"))
    model = WaveNet_VNNs(cfg).to(device).eval()
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def load_asc(model_path, device):
    """
    asc(안내방송/그 외 소음 이진 분류기) 로드
    """
    model = ASC().to(device)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model

def compute_sigmoid_mask(sig: np.ndarray, classifier: torch.nn.Module, device) -> np.ndarray:
    """
    분리된 각 채널(1초 단위)별로 분류기 통과 → 확률값 마스킹
    """
    window_len = SR
    L = len(sig)
    masks = np.zeros(L, dtype=np.float32)

    for start in range(0, L, window_len):
        end = min(start + window_len, L)
        seg = sig[start:end]
        if seg.shape[0] < window_len:
            seg = np.pad(seg, (0, window_len - seg.shape[0]), mode="constant")

        x = torch.from_numpy(seg.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = classifier(x)
            prob_chunk = torch.sigmoid(logit).item()
        masks[start:end] = prob_chunk

    return masks

def reduce_noise(noise_model, wav_np, device):
    """
    ANC 모델로 전체 소음(wav_np) 입력 → 저감/안티노이즈 분리 출력 (segment 단위 처리)
    """
    N    = len(wav_np)
    outs_en, outs_dn = [], []

    with torch.no_grad():
        for start in range(0, N, SEG_LEN):
            seg = wav_np[start:start+SEG_LEN]
            T   = len(seg)
            wav = torch.tensor(seg, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

            tgt_gpu = fir_filter(PRI.to(device), wav)
            out = noise_model(wav)
            if out.dim() == 3:
                out_flat = out.squeeze(1)
            elif out.dim() == 2:
                out_flat = out
            else:
                out_flat = out.unsqueeze(0)

            nl = SEF(out_flat, ETA)
            dn = fir_filter(SEC.to(device), nl)
            en = tgt_gpu + dn

            outs_en.append(en.squeeze().cpu().numpy()[:T])
            outs_dn.append(dn.squeeze().cpu().numpy()[:T])

    return np.concatenate(outs_en, axis=0), np.concatenate(outs_dn, axis=0)

def preprocess_sep_chunk(wav_chunk, hp):
    """
    Separation 입력: 정규화/패딩/길이 보정 후 tensor 반환
    """
    T = wav_chunk.shape[0]
    L = hp['chunk_len']
    if T < L:
        wav_chunk = np.pad(wav_chunk, (0, L - T), mode='constant')
    else:
        wav_chunk = wav_chunk[:L]

    t      = torch.from_numpy(wav_chunk).float().to(hp['device']).unsqueeze(0).unsqueeze(0)
    m_mean = t.mean(-1, True)
    m_std  = t.std(-1, True)
    t      = (t - m_mean) / (m_std + 1e-9)
    return t, m_mean, m_std, T

def postprocess_sep_chunk(est, m_mean, m_std, valid_len):
    """
    Separation 출력: 역정규화 및 유효길이만 반환
    """
    out = (est * (m_std + 1e-9) + m_mean).cpu().numpy()[0]
    return out[:, :valid_len]

def inference_end2end(args):
    """
    End-to-End 분리-분류-저감 전체 추론 파이프라인
    입력 폴더 내 모든 wav에 대해 각 단계별 결과/지표 산출
    """
    # 결과 저장 폴더 준비
    os.makedirs(args.sep_out,     exist_ok=True)
    os.makedirs(args.noise_out,   exist_ok=True)
    os.makedirs(args.anti_out,    exist_ok=True)
    os.makedirs(args.denoise_out, exist_ok=True)
    os.makedirs(args.final_out,   exist_ok=True)

    # 하이퍼파라미터 (모델마다 다름, 실험 config에 따라 변경)
    hp = {
        'out_channels':     256,
        'in_channels':      512,
        'num_blocks':       18,
        'upsampling_depth': 5,
        'enc_kernel_size':  21,
        'enc_num_basis':    256,
        'max_num_sources':  2,
        'fs':               SR,
        'device':           torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'chunk_len':        int(4.0 * SR),
    }

    # 모델 로드
    sep_model   = load_separation_model(args.sep_ckpt, hp)
    noise_model = load_noise_model(args.noise_cfg,   args.noise_ckpt, hp['device'])
    bcd_model   = load_asc(args.bcd_ckpt, hp['device'])
    dBA_fn      = dBA_Loss(fs=SR, nfft=512, f_up=SR/2).to(hp['device'])

    # 입력 데이터 미리 로드
    wav_list = []
    for path in sorted(glob.glob(os.path.join(args.input_dir, "*.wav"))):
        fname = os.path.splitext(os.path.basename(path))[0]
        wav, fs = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        assert fs == SR
        wav_list.append((fname, wav))

    # 각 파일별 속도/지표 리스트
    ms_per_s_list = []
    dba_list      = []
    nmse_list     = []

    for fname, wav in wav_list:
        t0 = time.time()
        L       = len(wav)
        sep_acc = np.zeros((hp['max_num_sources'], L), dtype=np.float32)
        sep_start = time.time()

        # 분리: chunk 단위 분리/정규화
        for start in range(0, L, hp['chunk_len']):
            end   = min(start + hp['chunk_len'], L)
            chunk = wav[start:end]
            mix_t, m_mean, m_std, valid = preprocess_sep_chunk(chunk, hp)
            with torch.no_grad():
                est = sep_model(mix_t)
                est = mixture_consistency.apply(est, mix_t)
            sep_acc[:, start:start+valid] = postprocess_sep_chunk(est, m_mean, m_std, valid)
        sep_end = time.time()

        # 분리 2채널 각각 방송/소음 여부 판별 (마스크 확률값 평균)
        mask_chan0 = compute_sigmoid_mask(sep_acc[0], bcd_model, hp['device'])
        mask_chan1 = compute_sigmoid_mask(sep_acc[1], bcd_model, hp['device'])
        if mask_chan0.mean() >= mask_chan1.mean():
            sep_broadcast = sep_acc[0]
            sep_noise     = sep_acc[1]
        else:
            sep_broadcast = sep_acc[1]
            sep_noise     = sep_acc[0]

        # 분리된 소음 채널(sep_noise) 저장
        sf.write(os.path.join(args.noise_out, f"{fname}_sep_noise.wav"), sep_noise, SR)

        # ANC(저감) 단계
        to_denoise_part  = sep_noise.copy()
        anc_start = time.time()
        enhanced_part, anti_part = reduce_noise(noise_model, to_denoise_part, hp['device'])
        anc_end = time.time()

        final_noise = enhanced_part
        final_mix   = sep_broadcast + final_noise

        t1 = time.time()
        total_ms = (t1 - t0) * 1000.0
        sep_ms   = (sep_end - sep_start) * 1000.0
        anc_ms   = (anc_end - anc_start) * 1000.0
        audio_sec = L / SR
        rtf       = (total_ms / 1000.0) / audio_sec
        ms_per_s  = total_ms / audio_sec

        # dBA, NMSE 지표 계산
        enh_t  = torch.tensor(final_noise, device=hp['device']).unsqueeze(0)
        anti_t = torch.tensor(anti_part,   device=hp['device']).unsqueeze(0)
        tgt_t  = enh_t - anti_t

        dBA_val  = (dBA_fn(enh_t) - dBA_fn(tgt_t)).item()
        nmse_val = (
            10 * torch.log10(
                (enh_t ** 2).sum() /
                ((enh_t - anti_t) ** 2).sum()
            )
        ).item()

        # 결과 파일 저장
        sf.write(os.path.join(args.sep_out,     f"{fname}_broadcast.wav"), sep_broadcast, SR)
        sf.write(os.path.join(args.denoise_out, f"{fname}_denoised_noise.wav"), final_noise, SR)
        sf.write(os.path.join(args.anti_out,    f"{fname}_anti_noise.wav"),       anti_part,   SR)
        sf.write(os.path.join(args.final_out,   f"{fname}_final.wav"),            final_mix,   SR)

        print(
            f"[OK] {fname}: total {total_ms:.1f} ms | sep {sep_ms:.1f} ms | "
            f"anc {anc_ms:.1f} ms | RTF {rtf:.3f} | {ms_per_s:.1f} ms/sec • "
            f"dBA {dBA_val:+.2f} dB • NMSE {nmse_val:+.2f} dB"
        )

        ms_per_s_list.append(ms_per_s)
        dba_list.append(dBA_val)
        nmse_list.append(nmse_val)

    # 전체 평균(앞 2개 제외) 및 표준편차 출력
    ms_array   = np.array(ms_per_s_list, dtype=np.float32)
    dba_array  = np.array(dba_list, dtype=np.float32)
    nmse_array = np.array(nmse_list, dtype=np.float32)

    if len(ms_array) > 2:
        ms_trimmed   = ms_array[2:]
        dba_trimmed  = dba_array[2:]
        nmse_trimmed = nmse_array[2:]
    else:
        ms_trimmed   = ms_array
        dba_trimmed  = dba_array
        nmse_trimmed = nmse_array

    mean_ms    = ms_trimmed.mean()
    std_ms     = ms_trimmed.std(ddof=0)
    mean_dba   = dba_trimmed.mean()
    mean_nmse  = nmse_trimmed.mean()

    print(f"\n=== 전체 {len(ms_array)}개 파일 처리 완료 (처음 2개 제외: {len(ms_trimmed)}개) ===")
    print(f"평균 ms/sec: {mean_ms:.2f} ms/sec")
    print(f"표준편차   : {std_ms:.2f} ms/sec")
    print(f"평균 dBA   : {mean_dba:+.2f} dB")
    print(f"평균 NMSE  : {mean_nmse:+.2f} dB\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sep_ckpt",    "-s",  required=True, help="C-SuDoRM-RF++ 체크포인트(.pt)")
    p.add_argument("--noise_cfg",   "-nc", required=True, help="WaveNet-VNNs config.json")
    p.add_argument("--noise_ckpt",  "-n",  required=True, help="WaveNet-VNNs 모델(.pth)")
    p.add_argument("--bcd_ckpt",    "-b",  required=True, help="ASC 모델(.pth)")
    p.add_argument("--input_dir",   "-i",  required=True, help="혼합 wav 폴더")
    p.add_argument("--sep_out",     "-o1", required=True, help="분리된 broadcast 저장 폴더")
    p.add_argument("--noise_out",   "-o2", required=True, help="분리된 소음(sep_noise) 저장 폴더")
    p.add_argument("--anti_out",    "-o3", required=True, help="ANC가 생성한 Anti-Noise 저장 폴더")
    p.add_argument("--denoise_out", "-o4", required=True, help="ANC 저감된 소음(Enhanced Noise) 저장 폴더")
    p.add_argument("--final_out",   "-o5", required=True, help="최종 합성된 wav 저장 폴더")
    args = p.parse_args()

    inference_end2end(args)

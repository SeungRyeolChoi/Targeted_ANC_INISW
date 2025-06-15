"""
Causal-SuDORMRF++ 음원 분리 모델의 폴더 단위 오디오 inference 스크립트

- 학습 시와 동일한 구조의 모델을 생성하고, 지정한 체크포인트에서 가중치를 불러와 inference-only 모드로 동작
- 입력 폴더 내 모든 .wav 파일을 대상으로 chunk 단위로 분리 처리 후, 각 분리 소스를 개별 wav로 저장
- 모델 추론 과정에서 mixture consistency 보정도 적용
"""

import os
import argparse
import time
import torch
import soundfile as sf
import numpy as np
import sys

# ── 프로젝트 루트 경로를 PYTHONPATH에 추가 (import 경로 보장)
current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from causal_improved_sudormrf_v3 import CausalSuDORMRF
import mixture_consistency

def load_model(ckpt_path, hp):
    """
    모델 객체 생성 → 체크포인트 가중치 불러오기 → eval 모드 전환
    - ckpt_path: 저장된 .pt 파일 경로
    - hp: 모델 구조와 디바이스 정보(dict)
    - 반환: PyTorch 모델 객체
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

def preprocess_chunk(wav_chunk, hp):
    """
    입력 오디오 청크(1D numpy) → 0-mean, unit-std 정규화 및 길이 보정
    - 길이 < target_len: zero padding
    - 길이 > target_len: 자름
    - 텐서 (1,1,T)로 변환
    - 정규화 평균/표준편차 반환(후처리 복원용)
    """
    T = wav_chunk.shape[0]
    target_len = hp['chunk_len']
    if T < target_len:
        wav_chunk = np.pad(wav_chunk, (0, target_len - T), mode='constant')
    else:
        wav_chunk = wav_chunk[:target_len]
    t = torch.from_numpy(wav_chunk).float().to(hp['device'])
    t = t.unsqueeze(0).unsqueeze(0)  # (1,1,T)
    m_mean, m_std = t.mean(-1, True), t.std(-1, True)
    t = (t - m_mean) / (m_std + 1e-9)
    return t, m_mean, m_std

def postprocess_chunk(est, m_mean, m_std, valid_len):
    """
    분리 네트워크 출력(est)을 원래 scale로 복원 후 numpy로 변환
    - valid_len만큼만 자름 (패딩 구간 제거)
    - est: (1, Nsrc, chunk_len) 텐서
    - 반환: (Nsrc, valid_len) numpy array
    """
    out = (est * (m_std + 1e-9) + m_mean).detach().cpu().numpy()[0]  # (Nsrc, T)
    return out[:, :valid_len]

def inference_folder(model, hp, input_dir, output_dir):
    """
    입력 폴더 내 모든 .wav 파일 반복 추론 → 각 분리 채널 wav 저장
    - 입력: model, 하이퍼파라미터(hp), 입력 폴더, 출력 폴더
    - chunk_len 단위로 청크 분할, 혼합 길이 전체 커버
    - stereo 파일은 mono로 변환
    - 각 소스별 파일을 {파일명}_s{1,2}.wav로 저장
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_len = hp['chunk_len']
    sr = hp['fs']

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith('.wav'):
            continue
        inpath = os.path.join(input_dir, fname)
        wav, fs = sf.read(inpath)

        # ── stereo → mono 변환
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        assert fs == sr, f"샘플링레이트 불일치: {fs} != {sr}"

        L = len(wav)
        n_src = hp['max_num_sources']
        outs_acc = np.zeros((n_src, L), dtype=np.float32)

        t0 = time.time()

        # ── chunk_len 단위로 non-overlap inference 반복
        for start in range(0, L, chunk_len):
            end = min(start + chunk_len, L)
            wav_chunk = wav[start:end]
            valid_len = end - start

            mix_t, m_mean, m_std = preprocess_chunk(wav_chunk, hp)

            with torch.no_grad():
                est = model(mix_t)                              # (1, Nsrc, chunk_len)
                est = mixture_consistency.apply(est, mix_t)

            out_chunk = postprocess_chunk(est, m_mean, m_std, valid_len)  # (Nsrc, valid_len)
            outs_acc[:, start:start+valid_len] = out_chunk

        total_time = time.time() - t0
        total_ms = total_time * 1000.0

        # ── 결과 소스별로 저장
        basename = os.path.splitext(fname)[0]
        for i in range(n_src):
            outpath = os.path.join(output_dir, f"{basename}_s{i+1}.wav")
            sf.write(outpath, outs_acc[i], sr)

        print(f"[OK] {fname} → {n_src} files, inference: {total_ms:.1f} ms "
              f"({total_time/L*sr*1000:.1f} ms/sec audio)")

if __name__ == "__main__":
    """
    커맨드라인 인자 파싱 및 하이퍼파라미터 설정, 모델 로드, 폴더 전체 추론 실행
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-ckpt", required=True,
                        help="훈련된 .pt 파일 경로")
    parser.add_argument("--input_dir",  "-i",   required=True,
                        help="혼합 wav 폴더")
    parser.add_argument("--output_dir", "-o",   required=True,
                        help="분리 결과 저장 폴더")
    args = parser.parse_args()

    # ── 학습/실행에 사용한 주요 하이퍼파라미터
    hp = {
        'out_channels':     256,
        'in_channels':      512,
        'num_blocks':       18,
        'upsampling_depth': 5,
        'enc_kernel_size':  21,
        'enc_num_basis':    256,
        'max_num_sources':  2,
        'audio_timelength': 4.0,
        'fs':               16000,
        'device':           torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'chunk_len':        int(4.0 * 16000),  # 4초 단위 청크
    }

    model = load_model(args.checkpoint, hp)
    inference_folder(model, hp, args.input_dir, args.output_dir)

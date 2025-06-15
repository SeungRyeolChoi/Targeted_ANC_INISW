"""
WaveNet-VNNs 추론 스크립트
- 학습된 모델로 test s2(noise) 파일에 대해 ANC(저감) 및 anti-noise 신호를 추론.
- dBA, NMSE, 처리속도(ms/sec) 등 성능 지표를 파일별·전체 평균으로 출력.
- pri/sec FIR 필터, SEF(비선형 함수), dBA loss 등 end-to-end 파이프라인 구성.
"""

import os
import json
import time
import glob
import argparse
import torch
import numpy as np
import soundfile as sf
from scipy.io import loadmat
from tqdm import tqdm
from networks import WaveNet_VNNs
from utils import fir_filter, SEF
from loss_function import dBA_Loss

def parse_args():
    """!
    커맨드라인 인자 파싱
    - 모델 경로, config, 입력/출력 폴더, eta, 샘플레이트, segment 길이 등 지정
    """
    parser = argparse.ArgumentParser(
        description="WaveNet-VNNs Inference Script"
    )
    parser.add_argument(
        "--model-path", "-m",
        required=True,
        help="Path to the trained model .pth file"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.json",
        help="Path to the network config JSON"
    )
    parser.add_argument(
        "--test-data-dir", "-t",
        required=True,
        help="Directory containing test WAV files ending with _s2.wav"
    )
    parser.add_argument(
        "--output-enh-dir", "-e",
        dest="output_enh_dir",
        required=True,
        help="Directory where enhanced WAVs will be saved"
    )
    parser.add_argument(
        "--output-anti-dir", "-a",
        dest="output_anti_dir",
        required=True,
        help="Directory where anti-noise WAVs will be saved"
    )
    parser.add_argument(
        "--eta", type=float, default=0.1,
        help="SEF nonlinearity parameter"
    )
    parser.add_argument(
        "--sr", type=int, default=16000,
        help="Sample rate"
    )
    parser.add_argument(
        "--seg-sec", type=int, default=10,
        help="Segment length in seconds"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    MODEL_PATH      = args.model_path
    CONFIG_PATH     = args.config
    TEST_DATA_DIR   = args.test_data_dir
    OUTPUT_ENH_DIR  = args.output_enh_dir
    OUTPUT_ANTI_DIR = args.output_anti_dir
    ETA             = args.eta
    SR              = args.sr
    SEG_SEC         = args.seg_sec
    SEG_LEN         = SEG_SEC * SR

    os.makedirs(OUTPUT_ENH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_ANTI_DIR, exist_ok=True)

    # 1) pri/sec FIR 필터 계수 로드
    pri = torch.tensor(loadmat("/content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/channel/pri_channel.mat")["pri_channel"].squeeze(), dtype=torch.float)
    sec = torch.tensor(loadmat("/content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/channel/sec_channel.mat")["sec_channel"].squeeze(), dtype=torch.float)


    # 2) WaveNet-VNNs 모델 초기화 및 가중치 로드
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = WaveNet_VNNs(config).to(device).eval()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # 3) dBA Loss
    dBA_fn_gpu = dBA_Loss(fs=SR, nfft=512, f_up=SR/2).to(device)

    # 4) 테스트용 s2(noise) 파일 리스트와 wav 메모리 프리로드
    # 저감 모델만을 테스트를 할 때 s2라 붙은 노이즈파일만을 가져다가 사용하였기에 추가하였음. 
    files = sorted(glob.glob(os.path.join(TEST_DATA_DIR, "*_s2.wav")))
    print(f"총 테스트 파일 수 (s2만): {len(files)}")

    wav_dataset = []
    for path in files:
        wav_np, sr = sf.read(path)
        assert sr == SR, f"샘플레이트 불일치: {sr} != {SR}"
        wav_dataset.append((path, wav_np))

    # 5) 파일별 지표 저장용 리스트
    all_dBA  = []
    all_NMSE = []
    all_ms   = []

    with torch.no_grad():
        for path, wav_np in tqdm(wav_dataset, desc="전체 파일 처리", unit="file"):
            fname = os.path.splitext(os.path.basename(path))[0]
            N     = len(wav_np)
            n_seg = (N + SEG_LEN - 1) // SEG_LEN

            outs_en, outs_dn = [], []
            seg_dBA, seg_NMSE, seg_time = [], [], []

            print(f"\n▶ `{fname}` 처리 시작: {n_seg} segments")
            for start in range(0, N, SEG_LEN):
                seg = wav_np[start:start+SEG_LEN]
                wav = torch.tensor(seg, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)

                t0 = time.time()
                # ANC 파이프라인: FIR(pri) → WaveNet → SEF → FIR(sec) → 합성
                tgt_gpu = fir_filter(pri.to(device), wav)
                out_gpu = model(wav)
                if out_gpu.dim() == 3:
                    out_flat = out_gpu.squeeze(1)
                elif out_gpu.dim() == 2:
                    out_flat = out_gpu
                else:
                    out_flat = out_gpu.unsqueeze(0)
                nonlin_gpu = SEF(out_flat, ETA)
                dn_gpu     = fir_filter(sec.to(device), nonlin_gpu)
                tgt_gpu.add_(dn_gpu)
                en_gpu = tgt_gpu
                t1 = time.time()

                # 성능지표 계산(dBA, NMSE)
                dBA_val  = (dBA_fn_gpu(en_gpu.squeeze(0)) - 
                            dBA_fn_gpu((en_gpu - dn_gpu).squeeze(0))).item()
                nmse_val = (10 * torch.log10((en_gpu**2).sum() /
                                            ((en_gpu - dn_gpu)**2).sum())).item()

                outs_en.append(en_gpu.squeeze().cpu().numpy())
                outs_dn.append(dn_gpu.squeeze().cpu().numpy())
                seg_dBA.append(dBA_val)
                seg_NMSE.append(nmse_val)
                seg_time.append(t1 - t0)

                del wav, out_gpu, out_flat, nonlin_gpu, dn_gpu, en_gpu
                torch.cuda.empty_cache()

            avg_dBA  = np.mean(seg_dBA)
            avg_NMSE = np.mean(seg_NMSE)
            ms_sec   = np.mean(seg_time) / SEG_SEC * 1000  # ms per second

            all_dBA.append(avg_dBA)
            all_NMSE.append(avg_NMSE)
            all_ms.append(ms_sec)

            # 추론 결과 wav 파일 저장 (enhanced/anti-noise)
            enh_path  = os.path.join(OUTPUT_ENH_DIR,  f"{fname}_enh.wav")
            anti_path = os.path.join(OUTPUT_ANTI_DIR, f"{fname}_anti.wav")
            sf.write(enh_path,  np.concatenate(outs_en).astype(np.float32), SR)
            sf.write(anti_path, np.concatenate(outs_dn).astype(np.float32), SR)

            print(f"✔ `{fname}` 완료 | "
                  f"평균 dBA 저감: {avg_dBA:+.2f} dB, "
                  f"평균 NMSE: {avg_NMSE:+.2f} dB, "
                  f"처리 속도: {ms_sec:.1f} ms/sec")

    # 6) 전체 평균 (앞의 2개 파일 제외, 워밍업/캐시 등 무시 목적)
    """
    초반의 첫 wav 파일 처리 시, 실제 파일 처리 시간이 아닌
    데이터 로더 초기화/병목 등의 준비 시간이 함께 포함되므로
    정확한 평균 속도를 반영하지 못할 수 있다고 판단.
    """ 
    if len(all_dBA) > 2:
        final_dBA  = np.mean(all_dBA[2:])
        final_NMSE = np.mean(all_NMSE[2:])
        final_ms   = np.mean(all_ms[2:])
        print("\n=== 최종 요약 (맨 처음 2개 파일 제외) ===")
        print(f"평균 dBA 저감 : {final_dBA:+.2f} dB")
        print(f"평균 NMSE    : {final_NMSE:+.2f} dB")
        print(f"평균 처리 속도: {final_ms:.1f} ms/sec")
    else:
        print("\n파일이 2개 이하라서 최종 평균을 계산할 수 없습니다.")

if __name__ == "__main__":
    main()

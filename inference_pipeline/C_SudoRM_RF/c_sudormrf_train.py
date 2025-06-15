"""
SudoRM-RF Causal 모델 학습/평가 메인 스크립트

- 프로젝트 내 PYTHONPATH 및 의존 모듈 세팅
- 커맨드라인 파서로 하이퍼파라미터 로드, 데이터 로더/모델/평가지표 등 생성
- 학습/검증/테스트 전체 loop 구현 (checkpoint resume, early stopping 포함)
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# ─── PYTHONPATH에 프로젝트 루트 추가 ─────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.insert(0, root_dir)

# ─── 프로젝트 모듈 import ─────────────────────────────────────
import improved_cmd_args_parser_v2 as parser
import mixture_consistency as mixture_consistency
import dataset_setup as dataset_setup
import sisdr as sisdr_lib
import causal_improved_sudormrf_v3 as causal_improved_sudormrf

def main():
    """
    커맨드라인 하이퍼파라미터 파싱 → 데이터/모델/평가지표 생성 → 학습/검증/테스트 전체 루프
    - 실험 reproducibility와 코드 유지보수성을 위해 모든 주요 하이퍼파라미터, 경로, 체크포인트 관리
    """
    args = parser.get_args()
    hparams = vars(args)

    # ─── 데이터로더 생성 ──────────────────────────────────────
    gens = dataset_setup.setup(hparams)
    assert hparams['n_channels'] == 1, 'Only mono-channel input supported'
    train_loader = gens['train']
    val_loader   = gens['val']
    test_loader  = gens['test']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ─── SI-SDR 평가 지표 객체 생성 ──────────────────────────
    sisdr_metric = sisdr_lib.StabilizedPermInvSISDRMetric(
        zero_mean=True,
        single_source=False,
        n_estimated_sources=hparams['max_num_sources'],
        n_actual_sources=hparams['max_num_sources'],
        backward_loss=False,
        improvement=True,
        return_individual_results=True
    ).to(device)
    sisdr_metric.permutations_tensor = sisdr_metric.permutations_tensor.to(device)

    # ─── 모델 생성 ──────────────────────────────────────────
    model = causal_improved_sudormrf.CausalSuDORMRF(
        in_audio_channels=1,
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        num_sources=hparams['max_num_sources']
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=1.0 / hparams['divide_lr_by'],
        patience=hparams['patience']
    )

    # ─── Checkpoint 설정 ───────────────────────────────────
    ckpt_dir = hparams['checkpoints_path']
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt_path = os.path.join(ckpt_dir, "causal_best.pt")
    last_ckpt_path = os.path.join(ckpt_dir, "causal_last.pt")

    start_epoch, best_val, no_improve = 1, -float('inf'), 0
    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', best_val)
        no_improve = ckpt.get('no_improve', no_improve)
        print(f"▶ Resume from checkpoint: {last_ckpt_path}")
        print(f"   Loaded epoch={ckpt['epoch']}, best_val={best_val:.2f}, no_improve={no_improve}")

    # ─── 온라인 증강 함수 정의 (에너지/순서 다양성 확보) ────────
    def online_augment(sources):
        B, n_src, T = sources.shape
        device = sources.device
        permuted = [sources[torch.randperm(B, device=device), i] for i in range(n_src)]
        wavs = torch.stack(permuted, dim=1)
        src_perm = torch.randperm(n_src, device=device)
        wavs = wavs[:, src_perm]
        scales = torch.rand(B, n_src, 1, device=device) + 0.5
        return wavs * scales

    es_patience = hparams.get('early_stop_patience', hparams['patience'])

    # ─── 메인 학습 루프 ──────────────────────────────────────
    for epoch in range(start_epoch, hparams['n_epochs'] + 1):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"[Epoch {epoch}/{hparams['n_epochs']}] Train", ncols=100)

        for i, (mix, clean) in enumerate(train_bar, 1):
            mix, clean = mix.to(device), clean.to(device)
            optimizer.zero_grad()

            aug = online_augment(clean)
            mix_in = aug.sum(dim=1, keepdim=True)
            mix_in = (mix_in - mix_in.mean(-1, True)) / (mix_in.std(-1, True) + 1e-9)

            if mix_in.dim() == 2:
                mix_in = mix_in.unsqueeze(1)
            elif mix_in.shape[1] != 1:
                mix_in = mix_in.mean(dim=1, keepdim=True)

            est = model(mix_in)
            est = mixture_consistency.apply(est, mix_in)

            loss_fn = sisdr_lib.PermInvariantSISDR(
                zero_mean=True,
                n_sources=hparams['max_num_sources'],
                backward_loss=True
            )
            loss = loss_fn(est, aug)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['clip_grad_norm'])
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(train_loss=f"{running_loss / i:.4f}")

        avg_train_loss = running_loss / len(train_loader)

        # ─── 검증 루프 ───────────────────────────────────────
        model.eval()
        val_scores = []
        with torch.no_grad():
            for _, clean in val_loader:
                clean = clean.to(device)
                mix_in = clean.sum(dim=1, keepdim=True)
                mix_in = (mix_in - mix_in.mean(-1, True)) / (mix_in.std(-1, True) + 1e-9)

                if mix_in.dim() == 2:
                    mix_in = mix_in.unsqueeze(1)
                elif mix_in.shape[1] != 1:
                    mix_in = mix_in.mean(dim=1, keepdim=True)

                est = model(mix_in)
                est = mixture_consistency.apply(est, mix_in)
                sdr, _ = sisdr_metric(est, clean, return_best_permutation=True)
                val_scores.append(sdr.tolist())

        mean_val = np.mean(val_scores)
        scheduler.step(mean_val)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val SISDRi: {mean_val:.2f} dB")
        print(f" Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if mean_val > best_val:
            best_val = mean_val
            no_improve = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ↑ New best model saved: {best_ckpt_path}")
        else:
            no_improve += 1
            print(f"  ↔ No improvement for {no_improve}/{es_patience} epochs")

        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val': best_val,
            'no_improve': no_improve
        }, last_ckpt_path)

        if no_improve >= es_patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement in {es_patience} epochs).")
            break

    print(f"\nTraining finished. Best Val SISDRi: {best_val:.2f} dB\n")

    # ─── 테스트 루프 (best model) ──────────────────────────
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    model.eval()
    test_scores = []
    with torch.no_grad():
        for _, clean in test_loader:
            clean = clean.to(device)
            mix_in = clean.sum(dim=1, keepdim=True)
            mix_in = (mix_in - mix_in.mean(-1, True)) / (mix_in.std(-1, True) + 1e-9)

            if mix_in.dim() == 2:
                mix_in = mix_in.unsqueeze(1)
            elif mix_in.shape[1] != 1:
                mix_in = mix_in.mean(dim=1, keepdim=True)

            est = model(mix_in)
            est = mixture_consistency.apply(est, mix_in)
            sdr, _ = sisdr_metric(est, clean, return_best_permutation=True)
            test_scores.append(sdr.tolist())

    print(f"Test SISDRi (best model): {np.mean(test_scores):.2f} dB")

if __name__ == '__main__':
    main()

"""
안내방송음/그 외 소음 이진 분류기 트레이닝 스크립트

- ASCDataset: 폴더 내 wav 파일을 1초 단위로 슬라이스하여 이진 라벨 데이터셋 구성 (안내방송/그 외 소음)
- ASC: 1D CNN 기반 이진 분류기 구조
- 데이터 imbalance 대응: WeightedRandomSampler 사용
- ThreadPoolExecutor로 wav 파일 duration 병렬 스캔(초기화 가속)
- 학습/검증/테스트 루프 및 다양한 분류 지표(정확도, F1, ROC-AUC 등) 계산 및 출력
- 체크포인트 저장 및 resume 지원
"""

import os
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import soundfile as sf
import random
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# 데이터셋 클래스: 폴더 내 모든 wav 파일을 1초 단위 세그먼트로 변환

class ASCDataset(Dataset):
    """
    지정된 폴더(안내방송음/그 외 소음) 내 wav 파일들을 모두 스캔하여,
    각 파일을 duration에 따라 1초(window_len) 단위로 슬라이스해서 세그먼트 생성.
    라벨: 안내방송음(1), 그 외 소음(0)
    병렬 duration 조회(ThreadPoolExecutor)로 속도 개선.
    """
    def __init__(self, broadcast_dir, noise_dir, sr=16000, window_len=16000, max_workers=8):
        self.sr = sr
        self.window_len = window_len
        self.segments = []
        # 방송/비방송 파일 목록 수집
        bfiles = glob.glob(os.path.join(broadcast_dir, '**', '*.wav'), recursive=True) if os.path.isdir(broadcast_dir) else []
        nfiles = glob.glob(os.path.join(noise_dir, '**', '*.wav'), recursive=True) if os.path.isdir(noise_dir) else []
        pairs = [(fp, 1) for fp in bfiles] + [(fp, 0) for fp in nfiles]
        print(f"[Info] Scanning {len(pairs)} files in {broadcast_dir} and {noise_dir}...")
        # 파일별 duration 병렬 조회 후, 1초 단위 세그먼트 정보 생성
        def get_duration_label(fp, label):
            info = sf.info(fp)
            return fp, info.frames / info.samplerate, label
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            futures = [exe.submit(get_duration_label, fp, lb) for fp, lb in pairs]
            for f in tqdm(as_completed(futures), total=len(futures), desc='Scanning'):
                fp, dur, lb = f.result()
                nseg = int(np.floor(dur))
                for i in range(nseg):
                    self.segments.append((fp, i * self.window_len, lb))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        fp, offset, lb = self.segments[idx]
        wav, _ = sf.read(fp, start=int(offset), frames=self.window_len, dtype='float32')
        if wav.shape[0] < self.window_len:
            wav = np.pad(wav, (0, self.window_len - wav.shape[0]), mode='constant')
        x = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        return x, torch.tensor(lb, dtype=torch.float32)


# 1D CNN 기반 단순 이진 분류기
class ASC(nn.Module):
    def __init__(self, window_len=16000):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 31, 2, 15)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, 31, 2, 15)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, 31, 2, 15)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(2)
        return self.fc(x)


# 트레이닝/검증/테스트 루프 + 평가 지표
def main(args):
    #데이터셋 생성
    train_ds = ASCDataset(args.train_s1_dir, args.train_s2_dir, sr=args.sr,
                                     window_len=args.window_len, max_workers=args.max_workers)
    val_ds   = ASCDataset(args.val_s1_dir, args.val_s2_dir, sr=args.sr,
                                     window_len=args.window_len, max_workers=args.max_workers)
    test_ds  = ASCDataset(args.test_s1_dir, args.test_s2_dir, sr=args.sr,
                                     window_len=args.window_len, max_workers=args.max_workers)
    #Weighted Sampler로 데이터 불균형 대응
    labels = [lb for _,_,lb in train_ds.segments]
    pos, neg = sum(labels), len(labels) - sum(labels)
    weights = [(1/pos if lb==1 else 1/neg) for lb in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    #모델/최적화/스케줄러 세팅
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ASC(window_len=args.window_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.BCEWithLogitsLoss()
    #체크포인트 경로 세팅
    resume = args.resume_checkpoint
    save   = args.save_path
    best   = save.replace('.pth', '_best.pth')
    last   = save.replace('.pth', '_last.pth')
    #체크포인트 resume
    start_epoch, best_val = 1, float('inf')
    if resume and os.path.isfile(resume):
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', best_val)
        print(f"Resumed from epoch {ckpt['epoch']} (best_val={best_val:.4f})")
    #에폭별 학습/검증 루프
    for ep in range(start_epoch, args.epochs+1):
        model.train(); train_loss = 0
        for xb, yb in tqdm(train_loader, desc=f"Train {ep}", leave=False):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(1), yb)
            loss.backward(); optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)
        #검증
        model.eval(); val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb).squeeze(1), yb).item() * xb.size(0)
        val_loss /= len(val_ds)
        print(f"Epoch {ep}: TrainLoss={train_loss:.4f}, ValLoss={val_loss:.4f}")
        #Best 모델 저장
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best)
            print(f" → New best model saved: {best}")
        #Resume용 체크포인트 저장
        ckpt = {'epoch': ep, 'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(), 'sched_state': scheduler.state_dict(),
                'best_val': best_val}
        torch.save(ckpt, save)
        scheduler.step(val_loss)
    #마지막 에폭 모델 저장
    torch.save(model.state_dict(), last)
    print(f" → Last model saved: {last}")
    #테스트 및 평가 지표 산출
    model.load_state_dict(torch.load(best, map_location=device))
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            test_loss += criterion(logits, yb).item() * xb.size(0)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    #지표 출력
    cm     = confusion_matrix(all_labels, all_preds)
    prec   = precision_score(all_labels, all_preds)
    rec    = recall_score(all_labels, all_preds)
    f1     = f1_score(all_labels, all_preds)
    roc_auc= roc_auc_score(all_labels, all_probs)
    pr_auc = average_precision_score(all_labels, all_probs)
    print(f"Test Loss={test_loss/len(test_ds):.4f}, Accuracy={correct/total:.4f} (Best model)")
    print(f"Precision={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}")
    print(f"ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for split in ['train', 'val', 'test']:
        parser.add_argument(f'--{split}_s1_dir', required=True)
        parser.add_argument(f'--{split}_s2_dir', required=True)
    parser.add_argument('--save_path', default='broadcast_classifier.pth', help='Path for resume checkpoint')
    parser.add_argument('--resume_checkpoint', default='', help='Checkpoint to resume')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--window_len', type=int, default=16000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)

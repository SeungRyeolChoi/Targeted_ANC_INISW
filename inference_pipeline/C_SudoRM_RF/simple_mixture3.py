"""
MixDataset 클래스 - SudoRM-RF/음원 분리 실험용 데이터셋 구현

- 각 split(train/val/test)에 대해 mixtures 및 개별 소스(spk1, spk2) 오디오 파일 경로를 자동으로 매핑
- 길이 보정, 정규화 등 실전 실험에서 반복적으로 쓰이는 오디오 전처리 로직을 통합
"""

import os
import glob
import torch
from torch.utils.data import Dataset as _TorchDataset
import librosa

class MixDataset(_TorchDataset):
    """
    데이터셋 옵션, 파일 경로, 데이터 로딩/전처리/정규화 등 실전 음원 분리 실험 전처리 일괄 관리
    """
    def __init__(
        self,
        root_dirpath,
        task,
        split,
        sample_rate,
        timelength,
        zero_pad,
        min_or_max,
        n_channels,
        augment,
        normalize_audio,
        n_samples,
        min_num_sources,
        max_num_sources
    ):
        """
        실험 환경/데이터셋 세팅 관련 모든 옵션 인스턴스 변수로 저장
        mixtures, spk1, spk2 오디오 파일 경로 자동 매핑
        """
        super().__init__()
        self.root_dirpath    = root_dirpath
        self.task            = task
        self.sample_rate     = sample_rate
        self.timelength      = timelength
        self.zero_pad        = zero_pad
        self.min_or_max      = min_or_max
        self.n_channels      = n_channels
        self.augment         = augment
        self.normalize_audio = normalize_audio
        self.n_samples       = n_samples
        self.min_num_sources = min_num_sources
        self.max_num_sources = max_num_sources

        # 데이터 split별 base 디렉토리 및 파일 경로 설정
        base      = os.path.join(root_dirpath, split)
        mix_dir   = os.path.join(base, 'mixtures')
        spk1_dir  = os.path.join(base, 'spk1')
        spk2_dir  = os.path.join(base, 'spk2')

        # mixture 파일 경로를 리스트로 저장
        self.mix_paths = sorted(glob.glob(os.path.join(mix_dir, '*.wav')))

        # spk1, spk2 오디오 id→경로 dict로 미리 매핑
        spk1_list = glob.glob(os.path.join(spk1_dir, '*_*.wav'))
        spk2_list = glob.glob(os.path.join(spk2_dir, '*_*.wav'))
        spk1_map  = {os.path.basename(p).split('_', 1)[1]: p for p in spk1_list}
        spk2_map  = {os.path.basename(p).split('_', 1)[1]: p for p in spk2_list}

        # mixture별 id 기반으로 개별 소스 파일 경로 매핑
        self.s1_paths = []
        self.s2_paths = []
        for mix_path in self.mix_paths:
            id_part = os.path.basename(mix_path).replace("mix_", "")
            try:
                self.s1_paths.append(spk1_map[id_part])
                self.s2_paths.append(spk2_map[id_part])
            except KeyError:
                raise FileNotFoundError(f"Cannot find matching sources for {mix_path}")

    def __len__(self):
        """ 전체 mixture(샘플) 개수 반환 """
        return len(self.mix_paths)

    def safe_load_audio(self, path):
        """
        오디오 파일 안전 로딩(librosa).
        에러 발생시 0으로 채운 wav 반환.
        """
        try:
            wav, _ = librosa.load(path, sr=self.sample_rate, mono=True)
            return torch.tensor(wav, dtype=torch.float32)
        except Exception as e:
            print(f"[WARNING] Failed to load {path}: {e}")
            return torch.zeros(int(self.timelength * self.sample_rate), dtype=torch.float32)

    def fix_length(self, wav):
        """
        오디오 길이(timelength * sample_rate)로 맞추기(pad or truncate).
        """
        target_len = int(self.timelength * self.sample_rate)
        T = wav.shape[-1]
        if T > target_len:
            start = torch.randint(0, T - target_len + 1, (1,)).item()
            return wav[..., start:start + target_len]
        elif T < target_len:
            pad_shape = list(wav.shape[:-1]) + [target_len - T]
            pad = torch.zeros(pad_shape, dtype=wav.dtype)
            return torch.cat([wav, pad], dim=-1)
        else:
            return wav

    def __getitem__(self, idx):
        """
        인덱스별 mixture/spk1/spk2 오디오 로드 및 길이 보정,
        shape 표준화, 옵션에 따라 정규화 적용하여 반환.
        반환: (mix_tensor, torch.stack([s1_tensor, s2_tensor]))
        """
        # 1. 오디오 로드
        mix_tensor = self.safe_load_audio(self.mix_paths[idx])
        s1_tensor  = self.safe_load_audio(self.s1_paths[idx])
        s2_tensor  = self.safe_load_audio(self.s2_paths[idx])

        # 2. 길이 보정
        mix_tensor = self.fix_length(mix_tensor)
        s1_tensor  = self.fix_length(s1_tensor)
        s2_tensor  = self.fix_length(s2_tensor)

        # 3. (1, T) shape로 강제
        if mix_tensor.ndim == 1:
            mix_tensor = mix_tensor.unsqueeze(0)
        elif mix_tensor.ndim == 2 and mix_tensor.shape[0] > 1:
            mix_tensor = mix_tensor.mean(0, keepdim=True)

        # 4. 정규화(옵션)
        if self.normalize_audio:
            mean = mix_tensor.mean()
            std  = mix_tensor.std(unbiased=False) + 1e-9
            mix_tensor = (mix_tensor - mean) / std
            s1_tensor  = (s1_tensor  - mean) / std
            s2_tensor  = (s2_tensor  - mean) / std

        return mix_tensor, torch.stack([s1_tensor, s2_tensor])

    def get_generator(self, batch_size, num_workers, shuffle=True):
        """
        실험에 바로 쓸 수 있는 PyTorch DataLoader 반환.
        - shuffle/num_workers/pin_memory/prefetch 등 고정 세팅
        """
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4
        )

# 외부 참조용 별칭
Dataset = MixDataset

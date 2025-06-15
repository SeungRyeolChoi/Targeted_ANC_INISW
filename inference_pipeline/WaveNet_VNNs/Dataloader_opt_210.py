"""
TrainDataset 및 TestDataset
- 음성 데이터를 지정한 길이와 간격으로 segment 단위로 잘라서
  학습/테스트용 Dataset 객체로 만듦
- 캐싱/중복 로드 방지, 파일 시스템 스캔, segment 슬라이딩 윈도우 처리 등 자동화
"""

import os
import torch
from torch.utils.data import Dataset
import torchaudio


# TrainDataset: 학습 데이터셋
# 지정한 폴더에서 모든 wav를 segment 단위로 잘라 리스트에 저장
# 캐시 파일이 있으면 재사용, 없으면 split 후 캐시 저장
class TrainDataset(Dataset):
    def __init__(self,
                 dimension=16000*3,     # segment 길이
                 stride=16000*3,        # segment 간 간격
                 data_path="/content/drive/MyDrive/WaveNet-VNNs-for-ANC/TrainData",
                 cache_file="/content/drive/MyDrive/WaveNet-VNNs-for-ANC/cache/prepare_train_data.pth_210"
                 ):
        super(TrainDataset, self).__init__()
        self.dimension = dimension
        self.stride = stride
        self.data_path = data_path
        self.cache_file = cache_file
        
        if os.path.exists(cache_file):
            # 캐시가 있으면 즉시 segment 리스트 불러옴
            self.wb_list = torch.load(cache_file)
            print("Loaded training cache")
        else:
            # 없으면 split()으로 segment 생성 후 캐시 저장
            self.wb_list = []
            self.split()
            torch.save(self.wb_list, cache_file)
            print("Saved training cache")
    
    def split(self):
        # 지정 폴더 내 모든 wav 파일을 순회하며 dimension, stride 단위로 segment 추출
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if file.lower().endswith('.wav'):
                    path = os.path.join(root, file)
                    wav, _ = torchaudio.load(path)
                    wav = wav.to(torch.float)
                    length = wav.size(1)
                    if length >= self.dimension:
                        start = 0
                        while start + self.dimension <= length:
                            segment = wav[:, start:start + self.dimension]
                            self.wb_list.append(segment)
                            start += self.stride

    def __len__(self):
        # 전체 segment 개수 반환
        return len(self.wb_list)

    def __getitem__(self, index):
        # segment tensor 반환([channel, dimension])
        return self.wb_list[index]


# TestDataset: 테스트 데이터셋
# TrainDataset과 유사하나, segment와 원본 파일 경로도 함께 반환

class TestDataset(Dataset):
    def __init__(self,
                 dimension=16000*10,
                 stride=16000*10,   
                 data_path="/content/drive/MyDrive/WaveNet-VNNs-for-ANC/test_data"  # test 데이터 경로
                 ):
        super(TestDataset, self).__init__()
        self.dimension = dimension
        self.stride = stride
        self.data_path = data_path
        self.samples = []
        self.split()

    def split(self):
        # 테스트 폴더 내 모든 wav 파일에서 dimension, stride 단위로 segment와 원본 파일경로 쌍 저장
        for root, _, files in os.walk(self.data_path):
            for file in files:
                if not file.lower().endswith('.wav'):
                    continue
                path = os.path.join(root, file)
                wav, _ = torchaudio.load(path)
                wav = wav.to(torch.float)
                length = wav.size(1)
                if length >= self.dimension:
                    start = 0
                    while start + self.dimension <= length:
                        segment = wav[:, start:start + self.dimension]
                        # (파일경로, segment tensor) 저장
                        self.samples.append((path, segment))
                        start += self.stride

    def __len__(self):
        # 전체 (파일경로, segment) 쌍 개수
        return len(self.samples)

    def __getitem__(self, index):
        # (파일경로, segment tensor) 튜플 반환
        return self.samples[index]

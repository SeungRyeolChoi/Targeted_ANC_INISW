"""
ANNOUNCENOISE 데이터셋 경로 및 주요 정보 정의(코랩 기준)
- ANNOUNCENOISE_ROOT_PATH: Google Drive 내 프로젝트 기준 데이터셋 루트 경로
- DATASETS: ANNOUNCENOISE 데이터셋의 각 split(train/val/test) 경로, 샘플링레이트, 소스수, 길이 등 주요 메타 정보 dict
"""

ANNOUNCENOISE_ROOT_PATH = '/content/drive/MyDrive/final_data'

DATASETS = {
    'ANNOUNCENOISE': {
        'train_dir': ANNOUNCENOISE_ROOT_PATH + '/train',
        'val_dir':   ANNOUNCENOISE_ROOT_PATH + '/val',
        'test_dir':  ANNOUNCENOISE_ROOT_PATH + '/test',
        'fs':        16000,   # 샘플링레이트(Hz)
        'n_src':     2,       # 소스 개수(예: 안내방송+소음)
        'audio_len': 4.0,     # 오디오 길이(초 단위)
    },
}

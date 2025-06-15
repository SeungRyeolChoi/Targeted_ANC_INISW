"""!
@brief Infer Dataset Specific parameters and return generators
@author Efthymios Tzinis
@copyright University of Illinois at Urbana-Champaign
"""

from __config__ import (
    ANNOUNCENOISE_ROOT_PATH,
)

import simple_mixture3 as simple_mixture_loader

def create_loader_for_simple_dataset(
    dataset_name=None,
    separation_task=None,
    data_split=None,
    sample_rate=None,
    min_or_max=None,
    zero_pad=None,
    timelegth=None,
    n_channels=None,
    normalize_audio=None,
    n_samples=None,
    min_num_sources=None,
    max_num_sources=None,
):
    """
    dataset_name, split 등에 따라 적절한 Dataset 객체를 생성해서 리턴.
    n_samples가 <=0 이면 None 으로 바꿔서 전체 사용.
    """
    # 1) 각 데이터셋별 loader, root_path, split 이름 결정
    if dataset_name == "ANNOUNCENOISE":
        loader = simple_mixture_loader
        root_path = ANNOUNCENOISE_ROOT_PATH
        translator = {"train": "train", "val": "val", "test": "test"}
        translated_split = translator[data_split]
    else:
        raise ValueError(f"Dataset `{dataset_name}` is not supported yet")

    # 2) n_samples가 None 또는 0 이하일 경우 None 으로 변경
    if n_samples is None or n_samples <= 0:
        n_samples_arg = None
    else:
        n_samples_arg = n_samples

    # 3) Dataset 생성
    data_loader = loader.Dataset(
        root_dirpath=root_path,
        task=separation_task,
        split=translated_split,
        sample_rate=sample_rate,
        timelength=timelegth,
        zero_pad=zero_pad,
        min_or_max=min_or_max,
        n_channels=n_channels,
        augment=("tr" in data_split),
        normalize_audio=normalize_audio,
        n_samples=n_samples_arg,
        min_num_sources=min_num_sources,
        max_num_sources=max_num_sources,
    )
    return data_loader


def setup(hparams):
    """
    hparams 의 train/val/test/train_val 각각에 대해
    create_loader_for_simple_dataset → get_generator 호출
    """
    generators = {}

    for data_split in ["train", "val", "test", "train_val"]:
        # 해당 split이 지정되지 않았으면 None
        if hparams.get(data_split) is None:
            generators[data_split] = None
            continue

        # 복수 데이터셋 미지원
        if len(hparams[data_split]) > 1:
            raise ValueError("현재는 여러 데이터셋 동시 사용을 지원하지 않습니다.")

        # n_samples 파라미터 뽑아서 <=0 이면 None
        raw_n = hparams.get(f"n_{data_split}", None)
        if raw_n is None or raw_n <= 0:
            n_samples_arg = None
        else:
            n_samples_arg = raw_n

        # Dataset 객체 생성
        loader = create_loader_for_simple_dataset(
            dataset_name=hparams[data_split][0],
            separation_task=hparams["separation_task"],
            data_split=data_split.split("_")[0],
            sample_rate=hparams["fs"],
            n_channels=hparams["n_channels"],
            min_or_max=hparams["min_or_max"],
            zero_pad=hparams["zero_pad_audio"],
            timelegth=hparams["audio_timelength"],
            normalize_audio=hparams["normalize_audio"],
            n_samples=n_samples_arg,
            min_num_sources=hparams["min_num_sources"],
            max_num_sources=hparams["max_num_sources"],
        )

        # DataLoader 생성
        generators[data_split] = loader.get_generator(
            batch_size=hparams["batch_size"],
            num_workers=hparams["n_jobs"],
        )

    return generators

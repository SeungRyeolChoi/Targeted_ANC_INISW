"""
실험 인자 파서 - SudoRmRf 및 다양한 데이터셋, 모델, 트레이닝 설정 지원
"""

import argparse

def get_args():
    """
    커맨드라인 인자 파서 생성
    """

    parser = argparse.ArgumentParser(
        description='실험 인자 파서 (Experiment Argument Parser)'
    )

    """
    ----- 데이터셋 관련 파라미터 -----
    """

    parser.add_argument("--train", type=str, nargs='+',
                        help="학습 데이터셋 선택", default=None,
                        choices=['WHAM', 'LIBRI2MIX', 'MUSDB', 'FUSS', 'WHAMR', 'ANNOUNCENOISE'])
    parser.add_argument("--val", type=str, nargs='+',
                        help="검증 데이터셋 선택", default=None,
                        choices=['WHAM', 'LIBRI2MIX', 'MUSDB', 'FUSS', 'WHAMR', 'ANNOUNCENOISE'])
    parser.add_argument("--test", type=str, nargs='+',
                        help="테스트 데이터셋 선택", default=None,
                        choices=['WHAM', 'LIBRI2MIX', 'MUSDB', 'FUSS', 'WHAMR', 'ANNOUNCENOISE'])

    parser.add_argument("--train_dir", type=str,
                        help="학습 데이터 경로", default=None)
    parser.add_argument("--val_dir", type=str,
                        help="검증 데이터 경로", default=None)
    parser.add_argument("--test_dir", type=str,
                        help="테스트 데이터 경로", default=None)

    parser.add_argument("--train_val", type=str, nargs='+',
                        help="train 데이터로 validation 수행 (optional)", default=None,
                        choices=['WHAM', 'LIBRI2MIX', 'WHAMR', 'ANNOUNCENOISE'])

    parser.add_argument("--n_train", type=int,
                        help="학습 샘플 개수 제한(미사용 시 0)", default=0)
    parser.add_argument("--n_val", type=int,
                        help="검증 샘플 개수 제한(미사용 시 0)", default=0)
    parser.add_argument("--n_test", type=int,
                        help="테스트 샘플 개수 제한(미사용 시 0)", default=0)
    parser.add_argument("--n_train_val", type=int,
                        help="train셋에서 validation용 샘플 개수 제한(미사용 시 0)", default=0)

    parser.add_argument("--audio_timelength", type=float,
                        help="오디오 로딩 길이(초 단위)", default=4.)

    parser.add_argument("--min_or_max", type=str,
                        help="길이 결정 기준(min: 가장 짧은 소스에 맞춰 자름, max: 가장 긴 소스에 0패딩)", 
                        default='min', choices=['min', 'max'])

    parser.add_argument("--zero_pad_audio", action='store_true',
                        help="필요한 timelength 맞추기 위해 zero padding 여부", default=False)

    parser.add_argument("--normalize_audio", action='store_true',
                        help="오디오별 평균/표준편차 정규화 수행", default=False)

    """
    ----- 분리 태스크 관련 파라미터 -----
    """

    parser.add_argument("--n_channels", type=int,
                        help="믹스 입력 채널 수(1/2)", default=1, choices=[1, 2])
    parser.add_argument("--min_num_sources", type=int,
                        help="믹스 내 최소 소스 개수", default=1)
    parser.add_argument("--max_num_sources", type=int,
                        help="믹스 내 최대 소스 개수", default=4)
    parser.add_argument("--separation_task", type=str,
                        help="수행할 분리 태스크", default=None,
                        choices=['enhance_single_white_noise',
                                 'enhance_single', 'enhance_both',
                                 'sep_clean', 'sep_noisy', 'noisy',
                                 'noisy_reverberant'])

    """
    ----- 트레이닝 하이퍼파라미터 -----
    """

    parser.add_argument("-bs", "--batch_size", type=int,
                        help="배치 크기", default=4)
    parser.add_argument("--n_epochs", type=int,
                        help="학습 epoch 수", default=500)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="초기 러닝레이트", default=1e-3)
    parser.add_argument("--divide_lr_by", type=float,
                        help="러닝레이트 감소 배수", default=3.)
    parser.add_argument("--patience", type=int,
                        help="patience (러닝레이트 감소)", default=20)
    parser.add_argument("--early_stop_patience", type=int, default=40,
                        help="early stopping patience")
    parser.add_argument("--resume", action='store_true',
                        help="이전 체크포인트에서 이어서 학습")
    parser.add_argument("--optimizer", type=str,
                        help="사용할 optimizer", default="adam",
                        choices=['adam', 'radam'])
    parser.add_argument("--clip_grad_norm", type=float,
                        help="gradient clipping max norm (0: 미사용)", default=5.)
    parser.add_argument("-fs", type=int,
                        help="오디오 샘플링레이트", default=8000)

    """
    ----- CometML 관련 실험정보 -----
    """

    parser.add_argument("-tags", "--cometml_tags", type=str,
                        nargs="+", help="CometML 실험 태그", default=[])
    parser.add_argument("--experiment_name", type=str,
                        help="실험명", default=None)
    parser.add_argument("--project_name", type=str,
                        help="프로젝트명", default="yolo_experiment")

    """
    ----- 디바이스/로깅 설정 -----
    """

    parser.add_argument("-cad", "--cuda_available_devices", type=str,
                        nargs="+", help="사용가능 CUDA 디바이스", default=['0'],
                        choices=['0', '1', '2', '3'])
    parser.add_argument("--n_jobs", type=int,
                        help="데이터로딩용 CPU 워커수", default=4)

    parser.add_argument("-elp", "--experiment_logs_path", type=str,
                        help="오디오 결과 로그 저장 경로", default=None)
    parser.add_argument("-mlp", "--metrics_logs_path", type=str,
                        help="메트릭 로그 저장 경로", default=None)
    parser.add_argument("-clp", "--checkpoints_path", type=str,
                        help="체크포인트 저장 경로", default=None)
    parser.add_argument("--save_checkpoint_every", type=int,
                        help="N epoch마다 모델 저장", default=0)

    """
    ----- SUDO-RM-RF 모델 주요 구조 파라미터 -----
    """

    parser.add_argument("--out_channels", type=int,
                        help="U-block 외부 internal 채널 수", default=128)
    parser.add_argument("--in_channels", type=int,
                        help="U-block 내부 internal 채널 수", default=512)
    parser.add_argument("--num_blocks", type=int,
                        help="U-block 반복 횟수", default=16)
    parser.add_argument("--upsampling_depth", type=int,
                        help="U-block 내 up/down-sampling 깊이", default=5)
    parser.add_argument("--group_size", type=int,
                        help="group comm module의 그룹 수", default=16)
    parser.add_argument("--enc_kernel_size", type=int,
                        help="인코더/디코더 커널 사이즈", default=21)
    parser.add_argument("--enc_num_basis", type=int,
                        help="인코더 basis 개수", default=512)

    """
    ----- Attention/모델 타입 -----
    """

    parser.add_argument("--att_dims", type=int,
                        help="attention 차원", default=256)
    parser.add_argument("--att_n_heads", type=int,
                        help="attention head 개수", default=4)
    parser.add_argument("--att_dropout", type=float,
                        help="attention dropout", default=0.1)
    parser.add_argument("--model_type", type=str,
                        help="모델 타입 선택", default='relu',
                        choices=['relu', 'softmax', 'groupcomm',
                                 'groupcomm_v2', 'causal',
                                 'attention', 'attention_v2',
                                 'attention_v3', 'sepformer'])

    return parser.parse_args()

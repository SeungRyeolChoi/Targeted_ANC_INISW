"""!
출처: https://github.com/etzinis/sudo_rm_rf/blob/master/sudo_rm_rf/dnn/experiments/utils/mixture_consistency.py
@brief 논문 기반 mixture consistency 적용 코드  
Scott Wisdom, John R Hershey, Kevin Wilson, Jeremy Thorpe, Michael
Chinen, Brian Patton, and Rif A Saurous.  
"Differentiable consistency constraints for improved deep speech enhancement", ICASSP 2019.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""


import torch

def apply(pr_batch, input_mixture, mix_weights_type='uniform'):
    """
    Mixture consistency(혼합 신호 일관성) 적용 함수
    분리된 소스들의 합이 원래 입력 혼합신호와 일치하도록 보정하는 함수
    
    :param pr_batch: 분리된 소스들(batch_size x 소스 수 x 파형 길이) 텐서
    :param input_mixture: 원본 혼합 신호(batch_size x 1 x 파형 길이) 텐서
    :param mix_weights_type: 소스별 correction 가중치 종류 ('uniform' 또는 'magsq')
    :return: mixture consistency가 적용된 소스 신호(batch_size x 소스 수 x 파형 길이)
    """
    num_sources = pr_batch.shape[1]
    pr_mixture = torch.sum(pr_batch, 1, keepdim=True)  # 예측 소스들의 합(혼합 추정값)

    if mix_weights_type == 'magsq':
        # 각 소스의 에너지(magnitude 제곱) 기반으로 가중치 산출
        mix_weights = torch.mean(pr_batch ** 2, -1, keepdim=True)
        mix_weights /= (torch.sum(mix_weights, 1, keepdim=True) + 1e-9)
    elif mix_weights_type == 'uniform':
        # 모든 소스에 동일 가중치 적용
        mix_weights = (1.0 / num_sources)
    else:
        raise ValueError('지원하지 않는 mixture consistency 가중치 타입입니다: {}'
                         ''.format(mix_weights_type))

    # 예측 혼합값(pr_mixture)과 실제 혼합(input_mixture)의 차이를 각 소스별로 보정
    source_correction = mix_weights * (input_mixture - pr_mixture)
    return pr_batch + source_correction

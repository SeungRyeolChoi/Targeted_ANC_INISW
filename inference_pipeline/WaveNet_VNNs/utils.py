"""
출처: https://github.com/Lu-Baihh/WaveNet-VNNs-for-ANC/blob/main/WaveNet_VNNs/utils.py
"""
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
import torch.distributed as dist

def is_dist_avail_and_initialized():
    """ 분산(distributed) 환경이 지원되는지 확인 """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    """ 전체 프로세스(노드) 수 반환 """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def reduce_value(value, average=True):
    """ 전체 노드 간 값 집계 (분산 환경 평균/합 계산) """
    world_size = get_world_size()
    if world_size < 2:  # 싱글 GPU일 때
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value

def slicing(x, slice_idx, axes):
    """ 텐서 특정 축 기준으로 slice 추출 """
    dimensionality = len(x.shape) 
    if dimensionality == 3:
       if axes == 1:
           return x[:,slice_idx,:]
       if axes == 2:
           return x[:,:,slice_idx]
    if dimensionality == 2:
       if axes == 0:
           return x[slice_idx,:]
       if axes == 1:
           return x[:,slice_idx]


def fir_filter(b, x):
    """ FIR 필터 연산 (Conv1d 기반 구현) """
    len = x.size()[2]
    M = x.size()[1]
    filter_len = b.size()[0]
    b = b.to(x.dtype)
    b = torch.flip(b, [0]).unsqueeze(0).unsqueeze(0)
    x = x
    y = F.conv1d(x, b, stride=1, padding=filter_len-1)[:,:,0:len] 
    return y

def SEF(y, etafang, num_points=1000):
    """
    SEF(Scaled Error Function) 비선형 함수  
    eta 파라미터를 통한 비선형 특성 반영
    """
    if y.dim() == 1:
        y = y.unsqueeze(0)  
    
    if etafang == 0:
        return y.unsqueeze(1)    # 선형 (비선형 없음)

    sign = torch.sign(y)
    y_abs = torch.abs(y)
    y_abs = torch.nan_to_num(y_abs, nan=0.0, posinf=1e6, neginf=-1e6)
    etafang = max(etafang, 1e-6)
    z = torch.linspace(0, 1, num_points, device=y.device, dtype=y.dtype).view(1, 1, -1)
    z = z * y_abs.unsqueeze(-1)
    integrand = torch.exp(-z**2 / (2 * etafang))

    if torch.isnan(integrand).any() or torch.isinf(integrand).any():
        print(f"NaN/Inf detected in integrand etafang={etafang}")

    dz = y_abs / (num_points - 1)
    dz = torch.where(dz == 0, torch.tensor(1e-6, device=dz.device, dtype=dz.dtype), dz)

    integral_abs = torch.sum((integrand[:, :, 1:] + integrand[:, :, :-1]) / 2, dim=2) * dz

    integral = sign * integral_abs
    return integral.unsqueeze(1)  

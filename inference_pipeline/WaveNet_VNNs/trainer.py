"""
WaveNet-VNNs Trainer 클래스
- 전체 학습(분산/단일), 체크포인트 관리, 로그/스케줄러, 손실 등 통합 처리
- AMP, Tensorboard, resume 기능 포함.

@author
"""

import os
import torch
import toml
from datetime import datetime
from tqdm import tqdm
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from utils import reduce_value
from utils import fir_filter, SEF
from torch.cuda.amp import autocast, GradScaler

class Trainer:
    """
    전체 학습 파이프라인 관리 클래스
    """
    def __init__(self, config, model, optimizer, loss_func1, loss_func2,
                 train_dataloader, pri_channel, sec_channel, train_sampler, args):
        # 주요 구성요소/파라미터 저장
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, [5, 10, 15, 20, 25], gamma=0.5, verbose=False)
        
        self.loss_func1 = loss_func1   # dBA 손실
        self.loss_func2 = loss_func2   # NMSE 손실

        self.train_dataloader = train_dataloader
        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size

        # 혼합정확도 향상 위한 AMP (자동 mixed precision)
        self.scaler = GradScaler()

        # 트레이너 관련 주요 설정값
        self.trainer_config = config['trainer']
        self.epochs = self.trainer_config['epochs']
        self.save_checkpoint_interval = self.trainer_config['save_checkpoint_interval']
        self.clip_grad_norm_value = self.trainer_config['clip_grad_norm_value']
        self.resume = self.trainer_config['resume']
        self.pri = pri_channel.to(self.device)
        self.sec = sec_channel.to(self.device)
        self.square_eta = self.trainer_config['eta']

        # 실험 로그/체크포인트/샘플 저장 경로 세팅
        if not self.resume:
            self.exp_path = self.trainer_config['exp_path'] + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
        else:
            self.exp_path = self.trainer_config['exp_path'] + '_' + self.trainer_config['resume_datetime']

        self.log_path = os.path.join(self.exp_path, 'logs')
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoints')
        self.sample_path = os.path.join(self.exp_path, 'val_samples')

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        # config 저장
        if self.rank == 0:
            with open(
                os.path.join(
                    self.exp_path, 'config.toml'), 'w') as f:
                toml.dump(config, f)
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = float('inf')

        # resume이면 마지막 체크포인트 복원
        if self.resume:
            self._resume_checkpoint()

        self.loss_func1 = self.loss_func1.to(self.device)
        self.loss_func2 = self.loss_func2

    def _set_train_mode(self):
        """모델을 train 모드로 전환"""
        self.model.train()

    def _save_checkpoint(self, epoch, score):
        """
        체크포인트 저장 (모델/옵티마이저/스케줄러/epoch 등 상태 포함)
        best_score 경신시 best model도 별도 저장
        """
        model_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        state_dict = {'epoch': epoch,
                      'optimizer': self.optimizer.state_dict(),
                      'scheduler': self.scheduler.state_dict(),
                      'model': model_dict}

        torch.save(state_dict, os.path.join(self.checkpoint_path, f'model_{str(epoch).zfill(4)}.tar'))
        torch.save(model_dict, os.path.join(self.checkpoint_path, 'model.pth'))

        if score < self.best_score:
            self.state_dict_best = state_dict.copy()
            self.best_score = score

    def _resume_checkpoint(self):
        """
        마지막 체크포인트에서 epoch, 옵티마이저, 스케줄러, 모델 파라미터 등 복원
        """
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, 'model_*.tar')))[-1]
        map_location = self.device
        checkpoint = torch.load(latest_checkpoints, map_location=map_location)

        self.start_epoch = checkpoint['epoch'] + 1
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

    def _train_epoch(self, epoch):
        """
        한 에폭 학습 및 손실/지표 기록, 진행상황 tqdm 출력
        """
        total_loss = 0
        total_dBAloss = 0
        total_nmseloss = 0
        train_bar = tqdm(self.train_dataloader, ncols=150)

        for step, ref in enumerate(train_bar, 1):
            ref = ref.to(self.device)

            with autocast():
                target = fir_filter(self.pri, ref)
                output = self.model(ref)
                nonlinear_out = SEF(output, self.square_eta)
                dn = fir_filter(self.sec, nonlinear_out)
                en = dn + target

                dBAloss = self.loss_func1(en.squeeze()) - self.loss_func1(target.squeeze())
                NMSEloss = self.loss_func2(en.squeeze(), target.squeeze())
                loss = 0.5 * dBAloss + 0.5 * NMSEloss
                if self.world_size > 1:
                    loss = reduce_value(loss)

            total_loss += loss.item()
            total_dBAloss += dBAloss.item()
            total_nmseloss += NMSEloss.item()

            train_bar.desc = '   train[{}/{}][{}]'.format(
                epoch, self.epochs + self.start_epoch - 1, datetime.now().strftime("%Y-%m-%d-%H:%M"))
            train_bar.set_postfix({
                'train_loss': '{:.4f}'.format(total_loss / step),
                'dBAloss': '{:.4f}'.format(total_dBAloss / step),
                'NMSEloss': '{:.4f}'.format(total_nmseloss / step),
                'lr': '{:.6f}'.format(self.optimizer.param_groups[0]['lr'])
            })

            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        if self.world_size > 1 and (self.device != torch.device("cpu")):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars('lr', {'lr': self.optimizer.param_groups[0]['lr']}, epoch)
            self.writer.add_scalars('train_loss', {'train_loss': total_loss / step}, epoch)

        return total_loss / step

    def train(self):
        """
        전체 에폭 반복 학습 루프, 체크포인트/로그/베스트 저장
        """
        if self.resume:
            self._resume_checkpoint()

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self._set_train_mode()
            score = self._train_epoch(epoch)
            self.scheduler.step()

            if (self.rank == 0) and (epoch % self.save_checkpoint_interval == 0):
                self._save_checkpoint(epoch, score)

        if self.rank == 0:
            torch.save(self.state_dict_best,
                       os.path.join(self.checkpoint_path,
                                    'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(4))))
            print('------------Training for {} epochs has done------------'.format(self.epochs))

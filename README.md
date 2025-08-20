# TargetedANC
소음은 스트레스 유발과 같은 정신적 피해와 지속적 환경 노출 시 심장질환, 청력장애 등 신체적 피해가 발생한다. 

이러한 소음을 저감하기 위해 능동형 소음 제어(ANC)는 원 소음에 대한 반대 위상을 생성하여 소음을 상쇄하지만, 비선형 특성 반영 부족과 처리 지연, 복합 소리에서의 선택적 저감에 한계가 있다. 이에 본 연구에서는 음원 분리와 소음 저감을 통합한 Targeted ANC 모델을 제안하였다. 

이 기법에서는 C-SuDoRM-RF++ 기반의 인과적 분리 모델을 통해 안내 방송음과·소음을 분리하고, 오디오 세거먼트 분류기가 소음으로 식별된 신호만을 WaveNet-Volterra Neural Network 기반의 소음  저감 모델로에 전달하여, 비선형 특성을 고려한 역소음을 생성한다. 모듈간의 overhead 발생을 최소화한 멀티태스크 구조로 설계된 Targeted ANC 모델은 실험 결과 연산량을 기존 대비 20.4% 감소시켰으며,  RTF 또한 21.43% 감소시킨 결과를 보였다. 또한 성능의 하락을 최소화한 결과 평균 dBA −35.12 dB·NMSE −43.72 dB를 기록하여 기존 FxLMS 대비 지연은 동등하게 유지하면서도 성능 저하 없이 저감 효율을 달성하였다. 

본 연구는 교통·모빌리티·웨어러블 디바이스 등 저지연 소음 제어가 요구되는 다양한 현장에 적용될 수 있을 것으로 기대한다.



## Model Architecture
![image](https://github.com/user-attachments/assets/4b1b942b-d494-4a87-baa0-9376238e0d46)

---

## Training Data
Training Data Download Link : https://drive.google.com/file/d/1odQm9jrT03vR3z78yDJt0k169AYluz3M/view?usp=sharing
<div align="center">
<img src="https://github.com/user-attachments/assets/a02c34e8-2300-44c7-8ad5-4a4dd7f9cf3f" alt="Dataset Analysis Tables" width="600">
</div>

총 36시간의 학습 데이터셋

서울 교통 공사의 1~8호선 안내방송음과 AI HUB의 도시 소음 데이터를 사용

분리(C-SuDoRM-RF++), 분류(ASC)에 사용

---

### Airplane Data
Airplane Data Download Link : https://drive.google.com/file/d/1sAq702S0YB-UkHnM5RuQGnYXmg5fniwf/view?usp=sharing

<div align="center">
<img src="https://github.com/user-attachments/assets/19263f40-3618-4bf6-b5b0-6e17ee62fd17" alt="airplane_data_statistic" width="600">
</div>

총 18시간의 항공 데이터셋

Simplaza의 74개의 항공사 기내 안전 안내방송 음원과 AI HUB의 도시 소음 데이터 중 비행기 소음 데이터를 사용

분리(C-SuDoRM-RF++), 분류(ASC)에 사용


## Inference_Pipeline
**분리, 분류, 저감 모델의 Training과 Inference Command line usage**

### C-SuDoRM-RF++ Training Command
```bash
!python /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/c_sudormrf_train.py \
  --model_type causal \
  --train "ANNOUNCENOISE" \
  --val "ANNOUNCENOISE" \
  --test "ANNOUNCENOISE" \
  --n_channels 1 \
  -fs 16000 \
  --batch_size 8 \
  --n_epochs 200 \
  --audio_timelength 4. \
  --enc_kernel_size 21 \
  --enc_num_basis 256 \
  --in_channels 512 \
  --out_channels 256 \
  --num_blocks 18 \
  -lr 0.001 \
  --divide_lr_by 3. \
  --patience 10 \
  --early_stop_patience 30 \
  --upsampling_depth 5 \
  --max_num_sources 2 \
  --min_num_sources 2 \
  --zero_pad_audio \
  --normalize_audio \
  -cad 0 \
  --n_jobs 4 \
  -clp <your_checkpoint_dir>
```




### C-SuDoRM-RF++ Inference Command
```bash
!python /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/c_sudormrf_inference.py \
  -ckpt /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/causal_best.pt \
  --input_dir <your_data_dir> \
  --output_dir <your_output_dir>
```

---

### Audio Segment Classifier(ASC) Training Command
```bash
!python /content/drive/MyDrive/inference_pipeline/ASC/ASC_train.py \
    --train_s1_dir /content/drive/MyDrive/final_data/train/spk1 \
    --train_s2_dir /content/drive/MyDrive/final_data/train/spk2 \
    --val_s1_dir     /content/drive/MyDrive/final_data/val/spk1 \
    --val_s2_dir     /content/drive/MyDrive/final_data/val/spk2 \
    --test_s1_dir    /content/drive/MyDrive/final_data/test/spk1 \
    --test_s2_dir    /content/drive/MyDrive/final_data/test/spk2 \
    --save_path <your_checkpoint_dir> \
    --sr 16000 \
    --window_len 16000 \
    --batch_size 16 \
    --lr 1e-4 \
    --epochs 15
```

### Audio Segment Classifier(ASC) Inference Command
```bash
!python /content/drive/MyDrive/inference_pipeline/ASC/ASC_inference.py \
  --test_s1_dir <your_broadcast_dir> \
  --test_s2_dir <your_noise_dir> \
  --model_path /content/drive/MyDrive/inference_pipeline/ASC/asc.pth
```

---


### WaveNet-VNNs Training Command
```bash
!python /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/train_opt_210.py \
  --config /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/cfg_train_opt_210.toml \
  --device 0
```

**Benchmark Dataset**

Demand: https://www.kaggle.com/datasets/chrisfilo/demand

ms_snsd: https://www.kaggle.com/datasets/jiangwq666/ms-snsd


### WaveNet-VNNs Infernce Command
```bash
!python /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/inference_opt.py \
  --model-path /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/model.pth \
  --config /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/config_opt_210.json \
  --test-data-dir <your_data_dir> \
  --output-enh-dir <your_denoise_dir> \
  --output-anti-dir <your_antinoise_dir>
```

---

### EndtoEnd Inference Command
```bash
!python /content/drive/MyDrive/inference_pipeline/end2end_inference.py \
  --sep_ckpt      /content/drive/MyDrive/inference_pipeline/C_SudoRM_RF/causal_best.pt \
  --noise_cfg     /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/config_opt_210.json \
  --noise_ckpt   /content/drive/MyDrive/inference_pipeline/WaveNet_VNNs/model.pth \
  --bcd_ckpt      /content/drive/MyDrive/inference_pipeline/ASC/asc.pth \
  --input_dir     <your_data_dir> \
  --sep_out      <your_seperation_dir> \
  --noise_out       <your_noise_dir> \
  --denoise_out      <your_denoise_dir> \
  --anti_out     <your_antinoise_dir> \
  --final_out   <your_final_dir>
```
**실행 환경은 Google Colab을 기준으로 작성되었습니다.**  
**모든 out 및 output 인자들은 빈 디렉토리로 미리 준비해두어야 합니다.**

## References

C-SudoRMRF++:
https://github.com/etzinis/sudo_rm_rf

WaveNet-VNNs:
https://github.com/Lu-Baihh/WaveNet-VNNs-for-ANC

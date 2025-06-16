# TargetedANC



## Model Architecture
![image](https://github.com/user-attachments/assets/4b1b942b-d494-4a87-baa0-9376238e0d46)

---




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

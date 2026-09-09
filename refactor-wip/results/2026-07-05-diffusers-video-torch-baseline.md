# PyTorch 原生 diffusers TextToVideoSDPipeline 基线（2026-07-05）

## 状态

✅ 已完成：建立原生 PyTorch 视觉质量基线，用于判断 Jittor 路径的视频 prompt 对齐预期。

## 环境约束

- Python：`/home/zy/miniconda3/envs/jt311/bin/python -S`
- 原生 PyTorch：通过只读加入 `/home/zy/rt_venv/lib/python3.11/site-packages`，避免 jt311 中已部署的 Jittor torch shim。
- GPU：`CUDA_VISIBLE_DEVICES=2`
- HF 缓存：`/home/zy/projects/jittor-lab/diffusers_video_jittor/hf_home`
- HF endpoint：`https://hf-mirror.com`
- 不修改 Jittor 源码，不 revert 现有改动。

## 运行记录

✅ 成功运行。

### 命令

```bash
cd /home/zy/projects/jittor

env -u PYTHONPATH \
  CUDA_VISIBLE_DEVICES=2 \
  HF_HOME=/home/zy/projects/jittor-lab/diffusers_video_jittor/hf_home \
  HF_ENDPOINT=https://hf-mirror.com \
  PYTHONNOUSERSITE=1 \
  /home/zy/miniconda3/envs/jt311/bin/python -S \
    ${JITTOR_LAB_ROOT}/diffusers_video_jittor/scripts/run_torch_text_to_video_sd.py \
    --model ali-vilab/text-to-video-ms-1.7b \
    --prompt 'a red sports car driving on a city street, cinematic, detailed, smooth motion' \
    --frames 16 \
    --height 256 \
    --width 256 \
    --steps 25 \
    --guidance-scale 9.0 \
    --seed 1234 \
    --fps 4 \
    --dtype fp16 \
    --output ${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/baseline_torch_car_g9_s25_seed1234_fp16.mp4 \
    --frames-dir ${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/baseline_torch_car_g9_s25_seed1234_fp16_frames \
    --contact-sheet ${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/baseline_torch_car_g9_s25_seed1234_fp16_contact_sheet.jpg \
  > ${JITTOR_LAB_ROOT}/diffusers_video_jittor/logs/baseline_torch_car_g9_s25_seed1234_fp16.log 2>&1
```

### 结果

- 状态：✅ 成功。
- 原生 torch：`2.12.1+cu130`，`/home/zy/rt_venv/lib/python3.11/site-packages/torch/__init__.py`。
- diffusers：`0.38.0`。
- GPU：`CUDA_VISIBLE_DEVICES=2`，脚本内可见 `NVIDIA GeForce RTX 4090`。
- load time：`3.51s`。
- generation time：`130.06s`。
- 视频：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/baseline_torch_car_g9_s25_seed1234_fp16.mp4`。
- metadata：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/baseline_torch_car_g9_s25_seed1234_fp16.json`。
- contact sheet：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/baseline_torch_car_g9_s25_seed1234_fp16_contact_sheet.jpg`。
- 日志：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/logs/baseline_torch_car_g9_s25_seed1234_fp16.log`。

### 验证

```bash
ffprobe -v error -select_streams v:0 \
  -show_entries stream=nb_frames,width,height,r_frame_rate,duration \
  -of json \
  ${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/baseline_torch_car_g9_s25_seed1234_fp16.mp4
```

结果：`width=256`，`height=256`，`r_frame_rate=4/1`，`duration=4.000000`，`nb_frames=16`。

PIL 抽查帧目录：

- 帧数：16。
- 首尾帧：RGB，256×256。

### 肉眼观察

✅ prompt 对齐：画面主体是红色跑车，场景是城市/道路，连续帧表现出车辆运动。

🟡 质量备注：画面有明显 `shutterstock` 水印，部分帧有运动模糊和遮挡，但这属于原生 PyTorch + 该模型/seed 的视觉基线输出，不是 Jittor 运行链路问题。

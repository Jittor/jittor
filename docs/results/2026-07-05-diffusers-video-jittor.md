# diffusers 视频生成支持记录（2026-07-05）

## 状态

✅ 已完成首要目标：在不修改 diffusers 原仓库代码的前提下，通过 `import jittor as torch` 跑通预训练视频生成模型，并已按官方默认 example 得到 demo 视频。

## 输出

- 预训练模型：官方 example 使用 `damo-vilab/text-to-video-ms-1.7b`（Hugging Face 页面重定向到 `ali-vilab/text-to-video-ms-1.7b`）
- pipeline：diffusers `DiffusionPipeline`，实际组件为 `TextToVideoSDPipeline`
- 官方 demo 视频：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/official_spiderman_surfs_jittor.mp4`
- 官方 demo 验证：16 帧，256x256，10 fps，时长 1.6 秒，uint8，mp4 文件大小 67856 bytes
- 官方 demo contact sheet：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/official_spiderman_surfs_jittor_contact_sheet.jpg`
- 官方 demo 元数据：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/official_spiderman_surfs_jittor.json`
- 官方 demo 日志：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/logs/official_spiderman_surfs_jittor.log`
- 输出视频：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/text_to_video_sd_smoke.mp4`
- 元数据：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/text_to_video_sd_smoke.json`
- 日志：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/logs/text_to_video_sd_after_ln_fix.log`
- 验证：4 帧，256x256，uint8，mp4 文件大小 88074 bytes

## 关键命令

```bash
CUDA_VISIBLE_DEVICES=1 \
CUDA_HOME=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux \
nvcc_path=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin/nvcc \
PATH=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/bin:$PATH \
LD_LIBRARY_PATH=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux/lib64:$LD_LIBRARY_PATH \
JITTOR_TORCH_PROJECT_ROOT=$PWD/diffusers_video_jittor \
HF_HOME=$PWD/diffusers_video_jittor/hf_home \
HF_ENDPOINT=https://hf-mirror.com \
PYTHONPATH=$PWD/python \
cache_name=diffusers_video_card1 \
/home/zy/miniconda3/envs/jt311/bin/python \
  ${JITTOR_LAB_ROOT}/diffusers_video_jittor/scripts/run_text_to_video_sd.py \
  --steps 2 --frames 4 --height 256 --width 256 \
  --guidance-scale 1.0 \
  --output ${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/text_to_video_sd_smoke.mp4
```

## 修复摘要

✅ `torch.linalg.inv_ex`：补到 `python/jittor/linalg.py`，返回 PyTorch 风格 namedtuple `(inverse, info)`。成功求逆时 `info` 为 batch shape 的 int32 零张量，用于 kornia `safe_inverse_with_mask`。

✅ import-time torch API：在 `python/jittor/torch_compat.py` 和 `python/jittor/torch_shim/torch__init__.py` 补齐：

- `torch.torch` 自引用，兼容 kornia 的 `torch.torch.Tensor` 注解与运行时 isinstance。
- `torch.amp.custom_fwd/custom_bwd` 与 `torch.cuda.amp.custom_fwd/custom_bwd` no-op decorator。
- 顶层 `torch.conv1d/2d/3d` 与 `torch.conv_transpose1d/2d/3d` 别名。

✅ CUDA LayerNorm fast path：`python/jittor/nn.py` 中 no_grad CUDA LayerNorm fast path 在 `float32 input + float16 affine` 时生成 `float * __half`，nvcc 报重载歧义。已把 scale/bias 显式 `static_cast<float>` 后参与计算。

✅ 脚本与回归：

- 新增 `${JITTOR_LAB_ROOT}/diffusers_video_jittor/scripts/run_tiny_text_to_video_sd.py`，用于随机权重 tiny pipeline smoke。
- 新增 `${JITTOR_LAB_ROOT}/diffusers_video_jittor/scripts/run_text_to_video_sd.py`，用于预训练模型生成。
- 新增 `${JITTOR_LAB_ROOT}/diffusers_video_jittor/scripts/run_official_text_to_video_sd_demo.py`，用于官方默认 example demo。
- 新增 `python/jittor/test/test_torch_compat_diffusers_video.py`，覆盖 `inv_ex`、`torch.torch`、AMP decorators、顶层 conv 与 CUDA mixed-dtype LayerNorm 编译。
- `python/jittor/misc.py` 补齐 `searchsorted(sorted, Number)`，支持 `DPMSolverMultistepScheduler` 的 scalar clipping index 路径；默认返回 int64，`out_int32=True` 返回 int32。

## 验证结果

✅ tiny 随机权重视频：

- `${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/tiny_text_to_video_sd.mp4`
- 4 帧，32x32，uint8

✅ 预训练视频：

- `${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/text_to_video_sd_smoke.mp4`
- 4 帧，256x256，uint8
- load time 14.29s，generation time 141.47s（含 JIT 编译/首次运行成本）

✅ 官方默认 example demo：

- `${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/official_spiderman_surfs_jittor.mp4`
- prompt：`Spiderman is surfing`
- pipeline：`DiffusionPipeline`
- scheduler：`DPMSolverMultistepScheduler`
- steps：25
- export：`diffusers.utils.export_to_video` 默认 10 fps
- 16 帧，256x256，10 fps，时长 1.6 秒
- load time 473.44s（含首次下载 `damo-vilab` 权重），generation time 414.10s
- 元数据：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/official_spiderman_surfs_jittor.json`
- 日志：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/logs/official_spiderman_surfs_jittor.log`
- contact sheet：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/official_spiderman_surfs_jittor_contact_sheet.jpg`
- 肉眼检查：主体为蜘蛛侠在冲浪，和官方 prompt 对齐；画面存在 `shutterstock` 水印伪影，原生 PyTorch 基线也出现同类水印。

✅ 追加 prompt demo：

- `${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/official_panda_surfs_jittor.mp4`
- prompt：`A panda is surfing on a wave`
- pipeline：`DiffusionPipeline`
- scheduler：`DPMSolverMultistepScheduler`
- steps：25
- export：`diffusers.utils.export_to_video` 默认 10 fps
- 16 帧，256x256，10 fps，时长 1.6 秒，mp4 文件大小 57509 bytes
- load time 14.72s（缓存热路径），generation time 336.85s
- 元数据：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/official_panda_surfs_jittor.json`
- 日志：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/logs/official_panda_surfs_jittor.log`
- contact sheet：`${JITTOR_LAB_ROOT}/diffusers_video_jittor/outputs/official_panda_surfs_jittor_contact_sheet.jpg`
- 肉眼检查：主体为熊猫在冲浪，和 prompt 对齐；仍有模型输出中的 `shutterstock` 水印伪影。

✅ targeted 回归：

```bash
/home/zy/miniconda3/envs/jt311/bin/python python/jittor/test/test_torch_compat_diffusers_video.py
```

结果：2 tests OK。

✅ diffusers guiders/kornia 导入：

```python
import jittor as torch
from torch.linalg import inv_ex
from diffusers.guiders import FrequencyDecoupledGuidance
```

结果：导入成功。

🟡 `python/jittor/test/test_torch_compat.py` 整体长测尝试未作为本次通过标准：它在既有 TransformerEncoder/softmax-grad 段报错，且此前有 `optimizer.step updates params` 检查失败；这些不在本次 diffusers 视频链路修复范围内。日志保留在 `${JITTOR_LAB_ROOT}/diffusers_video_jittor/logs/test_torch_compat_video_regressions.log`。

## 后续建议

🟡 扩展真实视频质量/速度验证：当前官方 demo 已按 16 帧、25 steps 跑通；后续可继续做同 prompt 的逐层数值对拍。

🟡 NPU 复验：本次首要目标按 GPU 先行完成，尚未在 Ascend 910B 上复验 TextToVideoSDPipeline。

🟡 kornia 低优先级候选缺口：explorer 扫描到 `torch.nn.utils.fusion.fuse_conv_bn_weights`、部署 shim 的 `_C._nn._parse_to` 等可能在其他 kornia 子路径触发，当前 TextToVideoSDPipeline 不阻塞。

## PPT 汇报提纲

### 技术路线

1. 不改 diffusers/kornia/transformers，入口层使用 `import jittor as torch`，把兼容工作收敛到 Jittor 主分支。
2. 先用 tiny 随机权重 pipeline 建立端到端闭环，再切预训练 `ali-vilab/text-to-video-ms-1.7b`，最后按官方默认 example 跑 `damo-vilab/text-to-video-ms-1.7b`。
3. 沿真实 import/runtime 失败链补 torch API 面：`inv_ex`、`torch.torch`、AMP decorators、顶层 conv、dlpack stub、`searchsorted(Number)`。
4. 计算保持在 Jittor CUDA 图内，遇到 mixed dtype CUDA codegen 问题修 Jittor kernel，不回退 CPU、不改模型。
5. 运行资产全部放 `${JITTOR_LAB_ROOT}/diffusers_video_jittor/`，固定 jt311、Jittor bundled CUDA、HF_HOME、cache_name，便于复现。

### 主要难点

1. diffusers video 的导入链会带出 kornia/guiders，API 面远大于核心 UNet/VAE/Scheduler。
2. kornia 的 import-time 注解和辅助函数依赖 PyTorch 边缘行为，例如 `torch.torch.Tensor` 与 `torch.amp.custom_fwd`。
3. `torch.linalg.inv_ex` 需要返回 PyTorch 风格 `(inverse, info)`，并且不能把不可逆矩阵静默吞掉。
4. fp16 预训练权重触发 LayerNorm fast path 的 `float * __half` nvcc 歧义，必须修 Jittor CUDA 源码生成。
5. 首跑耗时混有 JIT 编译成本，PPT 中要区分冷启动和缓存热路径性能。

### 结果数据

- tiny pipeline：4 帧，32x32，验证结构闭环。
- 预训练 smoke：4 帧，256x256，2 step。
- 官方 demo：`Spiderman is surfing`，16 帧，256x256，10 fps，1.6 秒，25 steps。
- 追加 prompt demo：`A panda is surfing on a wave`，16 帧，256x256，10 fps，1.6 秒，25 steps。
- targeted 回归：`test_torch_compat_diffusers_video.py` 2/2 OK。
- 中文汇报文档：`/home/zy/projects/doc/2026-07-05-diffusers-video-jittor.md` 已补充一页结论、PPT 页结构建议、最终产物索引和已知限制。

## 2026-07-05 继续整理 PPT 逐页讲稿

✅ 新增中文 PPT 逐页讲稿：`/home/zy/projects/doc/2026-07-05-diffusers-video-jittor-ppt.md`。

内容覆盖：

- 汇报定位：不改 diffusers，只改 Jittor 主分支和项目内脚本。
- 逐页结构：目标约束、最终 demo、官方 example 对齐、适配路线、API 缺口、CUDA kernel 难点、验证数据、PyTorch 基线、风险后续。
- Q&A 备答：官方默认 demo 只有 1.6 秒的原因、`enable_model_cpu_offload()` 与 `pipe.to("cuda")` 差异、水印伪影归因、精度/速度口径。
- 产物索引：官方 prompt 视频、追加 prompt 视频、contact sheet、脚本、总结文档和工作记录。

✅ 同步更新 `/home/zy/projects/doc/2026-07-05-diffusers-video-jittor.md` 的最终产物索引，加入 PPT 讲稿路径。

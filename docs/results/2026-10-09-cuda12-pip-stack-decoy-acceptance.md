# jittor[cuda12]：系统 CUDA 可见时的干净环境验收

- Status: uv 与 conda 两条安装路径、四种系统编译器情形在真 GPU 上通过
- Date: 2026-10-09
- Baseline commit: `6dbca3ec9`（含 `c546a81e6` 的 CUDA 13 nvcc + CUDA 12 运行时方案）
- Implementation commits: `3ed2c9efa`（运行库只取自 pip），以及提交本版报告的提交（本机 nvcc 优先）
- Owner: Jittor maintainers
- Review when: `cuda12` extra 的组件或版本、编译器发现、CUDA 库查找或预加载顺序变化
- Supersedes: [CUDA 13.4 编译器 + CUDA 12 运行时](2026-09-22-cuda13-nvcc-cuda12-runtime.md)

## 结论

`pip install "jittor[cuda12]"` 之后，CUDA 运行库（`libcudart`、cuBLAS、cuDNN 等）一律从
环境的 `site-packages/nvidia/` 加载。编译器先在本机找：JTCUDA、`PATH`、
`/usr/local/cuda/bin`、`/usr/bin`、`/opt/cuda/bin` 中第一个版本在 12.2–13.x、且能用
Jittor 的 C++ 编译器编出测试 kernel 的 `nvcc`；都不合格时用 pip 的 CUDA 13.4 `nvcc`。
跳过的本机编译器会记录原因。宿主机除驱动和 C++ 编译器外不需要别的。

pip 侧只能提供 CUDA 13 编译器：CUDA 12 的 `nvidia-cuda-nvcc-cu12`（核对过 12.2.140、
12.8.93、12.9.86）只带 `ptxas`，没有 `nvcc` 驱动程序。CUDA 13 只能为 SM 7.5 及更新的
GPU 生成代码，本机 CUDA 12 编译器因此仍有价值。低于 580 的驱动未验证。

## 只看版本号不够

本机（Debian 13，g++ 14.2，glibc 2.41）上实测：CUDA 12.4 的 `host_config.h` 拒绝
gcc 14；CUDA 12.9 的 `crt/math_functions.h` 与 glibc 2.41 的 `cospi`/`sinpi` 声明冲突；
CUDA 13.0 正常。所以本机 nvcc 必须先过一次试编译（结果按 nvcc 与 C++ 编译器文件缓存），
否则会选中一个第一次 JIT 就失败的编译器。

## 系统 CUDA 可见才暴露的缺陷

2026-09-22 的验收所在机器没有系统 CUDA，下面两处因此没有暴露：

| 现象 | 根因 | 修复 |
| --- | --- | --- |
| 进程里同时映射了系统的 `libcudart.so.12.4.127` 和 pip 的 `libcudart.so.12`，生成的 kernel 绑定到系统那份 | `jittor_core` 按名字依赖 `libcudart.so.12`，加载器先查 `LD_LIBRARY_PATH` 再查 core 的 `RUNPATH`；pip 的 cudart 在 core 之后才预加载 | `python/jittor/build/compiler.py`：有 pip 组件栈时，import core 前按绝对路径预加载 cudart，同名 SONAME 被复用 |
| pip 组件缺失或版本不符时只警告并回退到系统 CUDA，cuDNN 等可能从 `/usr/lib*`、`/usr/include` 解析 | 组件栈解析失败默认非致命；`setup_cuda_lib` 在组件目录后还会搜系统目录 | 选用 pip nvcc 时解析失败即报错并点名缺的包；有组件栈时 `setup_cuda_lib` 只搜组件目录；`find_pip_nvcc` 同时检查 `nvidia-cuda-cccl` |

另外，`nvidia-cudnn-frontend>=1.30` 的标记写成了 `python_version >= '3.9'`，而 1.29 起只有
cp310+ wheel，`uv.lock` 也没有随之重新生成，导致 `uv sync --locked --extra cuda12` 无法
解析。标记改为 `>= '3.10'` 并重新锁定。

## 验收

RTX 4090（SM 8.9），驱动 610.57.04，`CUDA_VISIBLE_DEVICES=1`，每次运行全新环境、独立
`JITTOR_HOME`、冷缓存、串行执行，未设置 `nvcc_path`。系统 CUDA 用 NVIDIA redist 包拼成，
在私有挂载命名空间里作为 `/usr/local/cuda` 出现，其 cudart（12.4）、cuBLAS 12.4.5.8、
cuDNN 9.1.0.70 链进 `/usr/lib/x86_64-linux-gnu`，并排在 `LD_LIBRARY_PATH`、`CUDA_HOME`、
`CUDA_PATH` 最前；有 nvcc 的情形下它也排在 `PATH` 最前并链到 `/usr/bin/nvcc`。

| 安装 | 系统编译器 | 选中的 nvcc | 系统 nvcc 编译调用 | 进程数 |
| --- | --- | --- | --- | --- |
| uv | CUDA 13.0.88（可用） | `/usr/local/cuda/bin/nvcc` | 9 | 1491 |
| uv | CUDA 12.4.131（拒绝 gcc 14） | pip 13.4.92 | 0（只有试编译） | 1478 |
| uv | 谎报 11.8 | pip 13.4.92 | 0（只有 `--version`） | 1472 |
| uv | 无 nvcc，只有库 | pip 13.4.92 | 0 | 1470 |
| conda | CUDA 13.0.88（可用） | `/usr/local/cuda/bin/nvcc` | 9 | 1493 |
| conda | CUDA 12.4.131（拒绝 gcc 14） | pip 13.4.92 | 0（只有试编译） | 1480 |

uv 用 `uv sync --locked --no-default-groups --extra cuda12`（Python 3.11.15，cuDNN
9.26.0.51）；conda 用 conda-forge Python 3.11.17 加 `pip install "<checkout>[cuda12]"`
（构建 wheel，cuDNN 9.27.0.42）。每次运行都满足：

- 加法、matmul 前向与两个梯度、conv2d 的结果经 `cuPointerGetAttribute` 确认在 GPU 内存，
  数值与 numpy 一致；conv2d 捕获到 `cudnn_conv precision select`。
- `libcudart.so.12`、`libcudnn.so.9` 的真实路径在该环境的 `site-packages/nvidia/` 下；
  未映射 `libcudart.so.13`。
- 全部进程的 `LD_DEBUG=files` 里没有从系统路径初始化的 CUDA 运行库。

在加入本机优先之前，同一组检查已在「只用 pip nvcc」的实现上通过（uv、conda 各一次）；
cudart 预加载修复前，同一 uv 环境的冒烟在映射检查处失败，修复后通过。

复现：[`cuda12-pip-stack-acceptance`](../../agent/skills/cuda12-pip-stack-acceptance/SKILL.md)。
原始日志在 `$JITTOR_LAB_ROOT/_state/cuda12-pip-stack/`（未版本化）。

## 仓库检查

- `uv lock --check`：通过。
- `tests/structure/build/test_cuda_wheel.py`：23 passed（含本机 nvcc 选择与版本范围）。
- `bash tools/check_repo_layout.sh`、`tools/build/generate_manifest.py --check`：通过。
- 完整 `tests/structure` 在 `3ed2c9efa` 前跑过一次：1389 passed，3 failed，均为未重新生成的
  `MANIFEST.in` 与运行中途改动的文档，修正后对应测试文件通过；本版只重跑了相关测试文件。
- 未跑 core/smoke/full 与 nox。

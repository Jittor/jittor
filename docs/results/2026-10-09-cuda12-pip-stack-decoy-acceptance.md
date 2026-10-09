# jittor[cuda12]：系统 CUDA 可见时的干净环境验收

- Status: uv 与 conda 两条安装路径在真 GPU 上通过
- Date: 2026-10-09
- Baseline commit: `6dbca3ec9`（含 `c546a81e6` 的 CUDA 13 nvcc + CUDA 12 运行时方案）
- Implementation commit: 与本报告同一提交
- Owner: Jittor maintainers
- Review when: `cuda12` extra 的组件或版本、编译器发现、CUDA 库查找或预加载顺序变化
- Supersedes: [CUDA 13.4 编译器 + CUDA 12 运行时](2026-09-22-cuda13-nvcc-cuda12-runtime.md)

## 结论

`pip install "jittor[cuda12]"` 之后，Jittor 只用这次装进环境的 NVIDIA 组件：JIT 编译用
`site-packages/nvidia/cu13/bin/nvcc`（13.4.92），运行时加载 CUDA 12.2 的 `libcudart`
和 cuDNN 9 的 `libcudnn`。宿主机除驱动和 Jittor 本来就要的 C++ 编译器外不需要别的；
机器上可见的系统 CUDA 不会被扫描、调用或加载。

CUDA 12 的 `nvidia-cuda-nvcc-cu12`（核对过 12.2.140、12.8.93、12.9.86）只带 `ptxas`，
没有 `nvcc` 驱动程序，所以编译器只能来自 CUDA 13 的 `nvidia-cuda-nvcc`。代价是只能为
SM 7.5 及更新的 GPU 生成代码；低于 580 的驱动未验证。

## 系统 CUDA 可见才暴露的缺陷

2026-09-22 的验收所在机器没有系统 CUDA，下面两处因此没有暴露：

| 现象 | 根因 | 修复 |
| --- | --- | --- |
| 进程里同时映射了系统的 `libcudart.so.12.4.127` 和 pip 的 `libcudart.so.12`，生成的 kernel 绑定到系统那份 | `jittor_core` 按名字依赖 `libcudart.so.12`，加载器先查 `LD_LIBRARY_PATH` 再查 core 的 `RUNPATH`；pip 的 cudart 在 core 之后才预加载 | `python/jittor/build/compiler.py`：有 pip 组件栈时，import core 前按绝对路径预加载 cudart，同名 SONAME 被复用 |
| pip nvcc 已选中、但某个组件包缺失或版本不符时，只警告并回退到系统 CUDA，cuDNN 等可能从 `/usr/lib*`、`/usr/include` 解析 | 组件栈解析失败默认非致命；`setup_cuda_lib` 在组件目录后还会搜系统目录 | `install_cuda.get_cuda_wheel_stack`：nvcc 位于 `site-packages/nvidia` 时解析失败即报错并点名缺的包；`compile_extern.setup_cuda_lib` 有组件栈时只搜组件目录；`find_pip_nvcc` 同时检查 `nvidia-cuda-cccl`（nvcc 13 必需、却不是它的依赖） |

另外，`nvidia-cudnn-frontend>=1.30` 的标记写成了 `python_version >= '3.9'`，而 1.29 起只有
cp310+ wheel，`uv.lock` 也没有随之重新生成，导致 `uv sync --locked --extra cuda12` 无法
解析。标记改为 `>= '3.10'` 并重新锁定。

## 验收

RTX 4090（SM 8.9），驱动 610.57.04，`CUDA_VISIBLE_DEVICES=1`，每条路径独立
`JITTOR_HOME`、冷缓存、串行执行。干扰项是用 NVIDIA redist 包拼成的完整 CUDA 12.4.131
工具链（cuBLAS 12.4.5.8、cuDNN 9.1.0.70），在私有挂载命名空间里作为 `/usr/local/cuda`、
`/usr/bin/nvcc`、`/usr/lib/x86_64-linux-gnu` 中的 CUDA 库出现，并排在 `PATH`、
`LD_LIBRARY_PATH`、`CUDA_HOME`、`CUDA_PATH` 最前面。未设置 `nvcc_path`。

| 路径 | 安装 | Python | cuDNN | 进程数（LD_DEBUG） |
| --- | --- | --- | --- | --- |
| uv | `uv sync --locked --no-default-groups --extra cuda12` | 3.11.15 | 9.26.0.51 | 1470 |
| conda | conda-forge Python + `pip install "<checkout>[cuda12]"`（构建 wheel） | 3.11.17 | 9.27.0.42 | 1472 |

两条路径都满足：

- 加法、matmul 前向与两个梯度、conv2d 的结果经 `cuPointerGetAttribute` 确认在 GPU 内存，
  数值与 numpy 一致；conv2d 捕获到 `cudnn_conv precision select`。
- `nvcc`、`libcudart.so.12`、`libcudnn.so.9` 的真实路径在该环境的 `site-packages/nvidia/` 下；
  未映射 `libcudart.so.13`。
- 干扰工具链的 `nvcc`/`ptxas` 调用记录为空；全部进程的 `LD_DEBUG=files` 里没有来自干扰
  路径的 CUDA 库。

修复前同一 uv 环境的冒烟在映射检查处失败（系统 `libcudart.so.12.4.127` 被加载），修复后通过。

复现：[`cuda12-pip-stack-acceptance`](../../agent/skills/cuda12-pip-stack-acceptance/SKILL.md)。
原始日志在 `$JITTOR_LAB_ROOT/_state/cuda12-pip-stack/`（未版本化）。

## 仓库检查

- `uv lock --check`：通过。
- `tests/structure/build/test_cuda_wheel.py`、`tests/build/test_cuda_library_search_dirs.py`：
  28 passed，2 skipped（与本机 cuDNN 位置有关的环境跳过）。
- `bash tools/check_repo_layout.sh`：通过。
- `JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure`：
  1389 passed，4 skipped，3 failed（683 s）。失败是未重新生成的 `MANIFEST.in` 和运行中途
  改动的文档；重新生成 manifest 后，这三项所在的文档、打包、私有路径与 CUDA wheel
  结构测试文件 62 passed。
- 未跑 core/smoke/full 与 nox。

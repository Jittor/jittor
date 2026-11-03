# 安装

[本栏目目录](index.md) · [English](installation.en.md)

先用 CPU 跑通一个例子，再配置加速设备。Jittor 会在本机即时编译算子，因此除了 Python，还需要 C++ 编译器。当前源码的依赖声明见 [`pyproject.toml`](https://github.com/Jittor/jittor/blob/2.0-refactor/pyproject.toml)。

## Linux：准备环境

以下命令适用于 Debian/Ubuntu，使用本机的 `python3` 创建独立环境。解释器需满足源码中的版本要求；仓库维护环境使用 Python 3.11：

```bash
sudo apt install python3-dev python3-venv libomp-dev g++ build-essential
STATE="${JITTOR_LAB_ROOT:-../jittor-lab}/_state/getting-started"
python3 -m venv "$STATE/venv"
. "$STATE/venv/bin/activate"
```

macOS 需要 clang 和 OpenMP（可通过 `brew install libomp` 安装）；其它平台要求见[根 README](https://github.com/Jittor/jittor/blob/2.0-refactor/README.md#install)。上面的 Linux 命令不能直接用于 Windows。

## 选择安装来源

只使用已发布版本：

```bash
python -m pip install jittor
```

阅读本分支文档并需要对应实现时，从已下载的仓库根目录安装源码：

```bash
python -m pip install --config-settings editable_mode=compat -e .
```

这两条是不同的安装方式，选择一条即可。PyPI 发布版不一定包含当前分支的全部功能。

## 验证 CPU 计算

```bash
python - <<'PYCPU'
import jittor as jt
jt.flags.use_cuda = 0
x = jt.array([1.0, 2.0, 3.0])
print((x * 2).numpy())
PYCPU
```

结果应为 `[2. 4. 6.]`。首次运行可能需要编译；若失败，先检查错误中提到的编译器、头文件或依赖。更多安装检查可运行 `python -m jittor.selftest`。

## 配置 ROCm / HIP

先安装设备驱动和完整的 HIP 开发工具链，包括 `hipcc`、HIP 运行时、hipBLAS 和 rocPRIM
的头文件与库，以及 hipRAND 设备头文件。Python 包不会安装这些系统依赖。海光 DTK 环境也应把 SDK 根目录作为
`ROCM_HOME`，而不是只指向其中的 `hip/` 子目录。

```bash
export JT_BACKEND=rocm
export ROCM_HOME="${ROCM_PATH:-/opt/rocm}"
export JITTOR_HOME="${STATE:-../jittor-lab/_state/getting-started}/rocm"
DISABLE_MULTIPROCESSING=1 python - <<'PYHIP'
import jittor as jt
assert jt.compiler.has_rocm
with jt.runtime.scope(use_cuda=1, backend_fallback="error"):
    result = (jt.array([1.0, 2.0, 3.0]) * 2).sum()
    result.sync()
    assert result.location() == "device"
    assert result.item() == 12.0
    print("HIP device computation passed")
PYHIP
```

`use_cuda` 是运行时通用的加速器开关；实际后端由 `JT_BACKEND=rocm` 选择。上面的
检查禁止 CPU 回退。缺少 HIP 运行时开发库时，配置阶段会明确报出 `libamdhip64.so`；
hipBLAS 和 rocPRIM 缺失也会直接报错。MIOpen 与 RCCL provider 尚未实现；此检查不代表
卷积库、多卡通信或任意模型已经通过验收，后端验证范围见[平台支持](../guides/hardware/platform-support.md)。

## 需要 Torch 兼容接口时

`jittor-compat` 是独立发行物。使用本仓源码时，在新建的专用环境中从仓库根目录同时安装两部分：

```bash
python -m pip install --config-settings editable_mode=compat -e .
python -m pip install -e ./compat
```

兼容包提供 `torch` 入口。为了避免与真实 PyTorch 争用模块名，请把两种运行时放在不同环境中；配套 wheel 的核心版本要求由包元数据约束。接着阅读[Torch 兼容使用指南](../compatibility/torch.md)。

## 下一步

完成[快速开始](quickstart.md)后，再按[设备与驻留教程](https://github.com/Jittor/jittor/blob/2.0-refactor/examples/notebooks/device_placement.md)配置设备。CUDA 需要对应工具链及驱动，NPU 等后端需要各自的运行环境；CPU 示例成功不代表其它芯片或某个下游库已经通过验收。

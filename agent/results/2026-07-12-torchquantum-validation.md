# TorchQuantum README 兼容性验证

## 状态

- ✅ 已完成：Jittor 已能通过 `import jittor as torch` 运行 TorchQuantum README 的 `Basic Usage` 和 `Usage`。
- ✅ CPU 与 CUDA 两个后端均已运行两段示例；CUDA Basic 的解析期望值和参数梯度与原生 PyTorch 对照一致。
- TorchQuantum 上游源码保持未修改；修复均落在 Jittor 兼容层、复数内核和回归测试中。

## 验证对象

- Jittor：分支 `2.0`；初始诊断基线 `1ae97707db931822f68d2d56842c42f6a34cb03f`，修复基线 `3b3e9856`
- Jittor 版本：`1.3.11.0`
- TorchQuantum：官方 `main`，commit `8dc3255c51477dd4c28892049571df032c77e2ff`，包版本 `0.3.0`
- README：该 commit 的 `README.md`，`Basic Usage` 位于 90-146 行，`Usage` 位于 182-240 行
- 验证脚本：`agent/skills/torchquantum-readme-validation/run_readme_examples.py`

TorchQuantum `v0.2.0` 与上述 `main` 的这两个代码块相同。本次选择当前默认分支 HEAD，避免用旧 PyPI `0.1.8` 代替“现在”的上游状态。

## 网络下载来源

TorchQuantum 不是使用本机预存副本或 PyPI 包，而是本次从用户指定的官方仓库网络克隆：

```bash
git clone --depth 1 https://github.com/mit-han-lab/torchquantum.git \
  /home/zy/projects/jittor-lab/torchquantum-validation/upstream
```

完成审计时的证据为：

```text
remote.origin.url=https://github.com/mit-han-lab/torchquantum.git
local HEAD=8dc3255c51477dd4c28892049571df032c77e2ff
remote HEAD=8dc3255c51477dd4c28892049571df032c77e2ff
remote main=8dc3255c51477dd4c28892049571df032c77e2ff
worktree=clean, branch main...origin/main
```

本机 `/home/zy/.gitconfig` 配有 `https://github.com/` 到 `https://gh-proxy.com/https://github.com/` 的 `insteadOf` 网络代理规则，因此 `git remote get-url` 会显示代理展开后的地址；仓库实际保存的 `remote.origin.url` 仍是上面的用户指定 URL。再次对该 URL 执行 `git ls-remote` 得到相同 SHA。

## 环境

- 主机：`cscg104`，8 张 NVIDIA GeForce RTX 4090；本机没有 NPU/CANN
- Jittor Python：`/home/zy/miniconda3/envs/jt311/bin/python`，Python 3.11.15
- CUDA：12.2.140，cuDNN 8.9.7，工具链 `/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux`
- 正式 Jittor 验证：物理 GPU 1；原生 PyTorch 对照：物理 GPU 2
- 独立缓存：`${JITTOR_LAB_ROOT}/_state/torchquantum-readme-validation/torchquantum_20260712`
- TorchQuantum 源码：`${JITTOR_LAB_ROOT}/torchquantum-validation/upstream`
- 隔离依赖：`${JITTOR_LAB_ROOT}/torchquantum-validation/deps`
- 关键依赖：NumPy 2.4.6、Qiskit 2.5.0、Qiskit Aer 0.17.2、Qiskit IBM Runtime 0.47.0、TorchPack 0.3.1、torchdiffeq 0.2.5、opt_einsum 3.4.0、matplotlib 3.11.0

TorchQuantum 当前 `requirements.txt` 要求 `numpy>=2.0`，而 Jittor `setup.py` 仍声明 `numpy<2.0`，所以二者的官方安装元数据存在直接冲突。为避免 pip 把 Jittor 的 torch shim 替换成真实 PyTorch，本次没有执行带依赖的 `pip install torchquantum`，而是把非 Torch 依赖安装到项目内隔离目录，并通过 `PYTHONPATH` 加载源码。初始诊断时，Jittor 在隔离 NumPy 2.4.6 下可以完成自身导入，随后失败在 torch compatibility API；这些阻断现已修复。

## 验证方法

入口首先执行：

```python
import jittor as torch
```

脚本随后确认：

```text
TORCH_IS_JITTOR=True
JITTOR_FILE=/home/zy/projects/jittor/python/jittor/__init__.py
```

这证明 TorchQuantum 内部的 `import torch` 确实解析到当前 Jittor，而不是环境里的原生 PyTorch。

首次 JIT 编译严格串行，后续所有 Jittor 进程也逐个执行，并设置：

```bash
export PYTHONPATH=/home/zy/projects/jittor/python:\
/home/zy/projects/jittor-lab/torchquantum-validation/deps:\
/home/zy/projects/jittor-lab/torchquantum-validation/upstream
export JITTOR_HOME=/home/zy/projects/jittor-lab/_state/torchquantum-readme-validation/torchquantum_20260712
export JTCUDA=/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux
export CUDA_HOME="$JTCUDA"
export nvcc_path="$JTCUDA/bin/nvcc"
export CUDA_VISIBLE_DEVICES=1
export use_parallel_op_compiler=0
```

正式命令为：

```bash
/home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/torchquantum-readme-validation/run_readme_examples.py \
  --case basic --device cuda

/home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/torchquantum-readme-validation/run_readme_examples.py \
  --case usage --device cuda

/home/zy/miniconda3/envs/jt311/bin/python \
  agent/skills/torchquantum-readme-validation/run_readme_examples.py \
  --case basic --device cpu
```

`Usage` 原代码块只定义 `QFCModel`，不会触发模型计算。验证脚本在原类定义后增加两条必要驱动：实例化模型，并用形状 `(2, 1, 28, 28)` 的输入做一次前向。批大小使用 2，避免 README 中无维度参数的 `.squeeze()` 把 batch 维一起移除。

## 初始严格运行结果（修复前）

三条正式命令退出码都为 1：

| 用例 | 设备配置 | 结果 | 到达阶段 |
|---|---|---|---|
| Basic Usage | CUDA / GPU 1 | 🔴 失败 | `import torchquantum` |
| Usage 前向 | CUDA / GPU 1 | 🔴 失败 | `import torchquantum` |
| Basic Usage | CPU | 🔴 失败 | `import torchquantum` |

共同错误为：

```text
RuntimeError: Numpy type not support, type_num: 15 type_char: D 24 void
```

因此 README 中的量子设备创建、门操作、QASM、采样、解析期望值、反向，以及 Usage 模型前向均未在未修改的 Jittor 路径中执行到。

## 修复前已复现阻断点

### 1. 显式 complex64 创建顺序错误，阻断顶层导入

TorchQuantum 在 `torchquantum/functional/paulix.py:116` 等位置使用：

```python
torch.tensor([... 1j ...], dtype=torch.complex64)
```

`python/jittor/torch_compat.py:1280-1304` 修复前先执行 `np.asarray(data)`，得到 NumPy `complex128`，再调用 `jt.array()` 创建 Var，最后才准备按请求的 `complex64` 做 cast。Jittor 没有 `complex128`，所以在 cast 前已经失败。

独立最小复现同时在 NumPy 1.26.4 和隔离 NumPy 2.4.6 下失败：

```python
import jittor as torch
torch.tensor([[1j]], dtype=torch.complex64)
```

### 2. complex64 张量不接受 Python complex 标量 setitem

仅为继续诊断，在单个进程中临时让上一项显式 complex64 字面量先转成 NumPy complex64。随后 TorchQuantum 导入继续到 `torchquantum/functional/sx.py:27`，执行：

```python
mat[14][14] = (1 + 1j) / 2
```

修复前 Jittor 报错：

```text
RuntimeError: Wrong inputs arguments
args = (int, complex)
```

该临时绕过没有修改源码，也没有计入验收通过；它证明修复首个创建顺序后仍有第二个导入阻断。

### 3. Usage 测量需要 `Tensor.mv`，修复前接口不存在

TorchQuantum `torchquantum/measurement/measurements.py:323` 的 `MeasureAll` 前向调用：

```python
res = probs.mv(observable.eigvals.real.to(probs.device))
```

GPU 最小复现：

```python
matrix = torch.ones((2, 3), device="cuda")
vector = torch.ones((3,), device="cuda")
matrix.mv(vector)
```

实际错误：

```text
AttributeError: 'jittor_core.Var' object has no attribute 'mv'
```

底层 `matmul` 已能表达矩阵向量乘，但 torch-compatible 的 `torch.mv`/`Tensor.mv` 接口没有暴露。

### 4. Basic Usage 参数反向缺 complex64 -> float32 cast

TorchQuantum RX 的 float32 可训练参数在门矩阵构造中 cast 为 complex64。Basic Usage 的 `expval[0].backward()` 必须把 complex64 梯度传回 float32 参数。GPU 最小复现：

```python
p = torch.tensor([0.5], dtype=torch.float32, device="cuda", requires_grad=True)
z = p.type(torch.complex64)
z.real.sum().backward()
```

实际在 nvcc 编译阶段失败：

```text
error: no suitable conversion function from "jittor::complex64"
       to "jittor::float32" exists
```

这与仓库 `test_complex64_gradfunctional.py:20-27` 已记录的限制一致，但本次已经用 TorchQuantum 所需的梯度形态独立复现，不再只是静态推断。

## 原生 PyTorch 对照

为排除 TorchQuantum 当前 README 或依赖本身失效，在同一上游 commit、同一隔离依赖下使用 `/home/zy/rt_venv` 的原生 `torch 2.12.1+cu130` 在 GPU 2 运行：

- Basic Usage：退出码 0；状态在 CUDA；解析 `ZX` 期望值为 `[0.8776] * 5`；RX 参数梯度为 `-0.4794`；`from_op_history` 后状态有限。
- Usage：退出码 0；输入 `(2, 1, 28, 28)`；输出 `(2, 2)`、CUDA、float32、数值有限；`exp(output)` 每行和为 1。

因此上游示例和所选运行依赖有效，Jittor 失败不是由 TorchQuantum README 自身造成。

## 修复内容

### 1. 复数创建、标量和梯度链

- `torch.tensor(..., dtype=torch.complex64)` 在首次创建 Var 前把 Python/NumPy 数据规范成 NumPy `complex64`，不再先构造不受支持的 complex128。
- Python complex 默认按 torch 的默认复数规则使用 complex64；显式 NumPy complex128 且未请求 complex64 仍响亮失败，不做静默窄化。
- complex64 张量 setitem 遇 Python/NumPy complex 标量时先物化为单元素 complex64 Var，支持 direct 和 `x[i][j]` 级联写入。
- `1j * tensor`、`tensor + np.complex64(...)`、complex scalar 除法等二元运算先把标量物化为 complex64 Var，实际计算仍在当前 device 上执行。
- complex64 -> real cast 在 codegen 中明确取实部；complex -> bool 同时检查实部和虚部。
- `UnaryOp` 补齐 complex -> real cast 的反向，real -> complex 的反向可把 complex dout 正确投影回真实参数；TorchQuantum RX 梯度链恢复。

相关源码：

- `python/jittor/torch_compat.py`
- `python/jittor/contrib.py`
- `python/jittor/src/type/complex_compute.h`
- `python/jittor/src/type/complex_op_type.cc`
- `python/jittor/src/ops/unary_op.cc`

### 2. TorchQuantum 导入与张量 API

- 注册可导入的 `torch.autograd.functional` 子模块，复用 `jittor.gradfunctional.vjp/jvp`，解除 TorchQuantum 顶层 pulse -> torchdiffeq 导入阻断；部署 shim 同步挂载该子模块。
- 新增模块级 `torch.mv` 和 `Tensor.mv`，含维度、内积长度检查以及 `out=` 支持；底层直接复用 device 端 `jt.matmul`。
- transpose/permute 入口把 NumPy integer 轴规范成 Python int，支持 TorchQuantum 的 `list(np.argsort(...))`。

相关源码：

- `python/jittor/torch_compat.py`
- `python/jittor/torch_shim/torch__init__.py`
- `python/jittor/__init__.py`

### 3. 0-D 导出和视图初始化语义

- Jittor 内部仍用单元素 Var 表示 torch 0-D；对 torch 语义上已完全 squeeze 的张量增加 `_torch_0d` 标记。
- 标记会穿过 `detach/to/cpu/cuda`，且仅在 `numpy()/tolist()` 边界导出标量。这样 RX 历史参数为 `0.5`，不会错误变成 `[0.5]` 并被 Qiskit 拒绝。
- `nn.init.constant_` 等初始化器写入 basic-index view 时沿记录的 parent/slices 回写父参数。实际复现的 `parameter[:, 1]` 只改临时 view、父参数不变问题已修复，TorchQuantum U3 参数初始化不再静默丢失。

### 4. 验证期间发现的既有 linalg 问题

- `linalg.matrix_rank` 在 torch compatibility 安装后会收到 `max(dim=...)` 的 `(values, indices)` 返回对象，却把整个对象参与阈值乘法。本次解包 `.values`，现有 linalg 回归恢复 11/11。

## 新增回归

- `test_complex64_native.py`：complex 标量 direct/chained setitem、complex/real/bool cast、双向 cast 梯度、Python/NumPy complex 标量二元运算。
- `test_torch_compat_dtype.py`：Python complex 容器和 NumPy complex128 输入显式创建 complex64，含 `as_tensor`。
- `test_torch_compat_linalg.py`：`torch.mv`、`Tensor.mv`、`out=`、错误维度和长度。
- `test_torch_compat_autograd.py`：从 `torch.autograd.functional` 导入并执行 `vjp`。
- `test_torch_compat_ops.py`：NumPy integer 轴 permute；squeeze 标量经 detach/cpu 的 NumPy/tolist 导出。
- `test_torch_compat_nn.py`：initializer 对 parameter view 的父张量写回和 trainable 状态。

## 最终验证结果

| 用例 | 后端 | 结果 | 关键证据 |
|---|---|---|---|
| Basic Usage | CUDA / GPU 1 | ✅ 通过 | `BASIC_USAGE_OK`，expval `0.8775823`，grad `-0.47942543` |
| Usage 实际前向 | CUDA / GPU 1 | ✅ 通过 | `USAGE_OK`，输出 `(2,2)`、float32、finite |
| Basic Usage | 严格 CPU (`use_cuda=0`) | ✅ 通过 | `BASIC_USAGE_OK`，expval `0.8775825`，grad `-0.47942543` |
| Usage 实际前向 | 严格 CPU (`use_cuda=0`) | ✅ 通过 | `USAGE_OK`，输出 `(2,2)`、float32、finite |

四次都打印：

```text
TORCH_IS_JITTOR=True
JITTOR_FILE=/home/zy/projects/jittor/python/jittor/__init__.py
```

最终 CUDA Basic 还确认 QASM 包含：

```text
rx(0.5) q[0];
```

回归结果：

- 受影响的 complex/gradfunctional/dtype/autograd/ops/nn 六模块：126 passed，2 个既有 expected skip。
- `test_torch_compat_linalg`：11/11 passed。
- `python/jittor/test/test_torch_compat.py`：172 passed，0 failed。
- 长组合进程中 CuPy det 曾出现一次 NVRTC denormal 常量编译错误；同一测试独立进程立即通过，未形成稳定代码问题。

## 依赖与环境边界

- 没有向 Jittor 增加新依赖。
- 继续使用项目内隔离依赖目录，避免 pip 安装 TorchQuantum 时拉取真实 PyTorch 覆盖 shim。
- TorchQuantum `numpy>=2.0` 与 Jittor 安装元数据 `numpy<2.0` 的冲突仍存在；本次运行通过 `PYTHONPATH` 隔离解决，没有修改安装元数据。
- 当前 cscg104 没有 NPU/CANN，无法做真 910B 复验。`mv` 和新增 Python 组合路径使用 device 原语，但现有 ACL dtype 映射尚不支持 native complex64，因此本记录不宣称 TorchQuantum NPU 全链可用。

## 结论

✅ 验收目标已完成：固定 TorchQuantum 官方 `main@8dc3255c` 后，Jittor 能以 `import jittor as torch` 方式运行 README `Basic Usage` 和 `Usage`，CPU/CUDA 均通过，Basic 的解析值与反向梯度对齐原生 PyTorch。

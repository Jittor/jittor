# CUDA parallel-range 与常见网络独立 oracle 验证

- Status: Accepted for CPU/CUDA correctness; training/performance incomplete
- Last reviewed: 2026-08-22
- Baseline: `0a3458b3`
- Owner: JIT compiler, CUDA backend, and model-parity maintainers
- Review when: ParallelPass templates, fused-op compilation, device launch
  partitioning, network parity builders, or real-Torch loading changes

## 结论

CPU parallel-range 修复曾被无条件应用到 CUDA 生成源，导致 CUDA launch 的线程位段
边界被累积两次。小张量可能通过，但较大 fused broadcast 只写出部分元素，进而使
ResNet18、ViT 和 diffusion UNet 的 CUDA 前向相对误差达到 `0.85` 至 `1.04`。

`0a3458b3` 将源码重写严格限制到 `JIT_cpu`。CPU 继续使用修正后的累计线程边界；
CUDA 保留其模板已有的后置累积。大 NCHW channel-bias broadcast、完整 parallel-pass
模块、八项紧凑网络 CPU/CUDA 前反向对拍，以及完整冷缓存 CUDA 门禁均通过。

## Verify-then-fix

独立二进制 PyTorch 2.12.1+cu130 的 `torch.__file__` 与 `torch._C.__file__` 均由
`REAL_TORCH_SITE` 校验，未加载 Jittor 部署的 Torch shim。初始网络结果为
`5 passed, 3 failed`：GPT-2 的 CPU/CUDA 与其他三个网络的 CPU 路径通过；三个
包含更大 fused broadcast 的 CUDA 网络失败。

最小探针排除了权重加载和 cuDNN：

- 大 NumPy 数组 CUDA round trip 误差为零；
- raw cuDNN 与无 bias Conv2d 对 PyTorch 的相对误差约 `2.7e-7` 至 `4.3e-7`；
- `(2,32,16,16)` channel-bias broadcast 仅写出 16,384 个元素中的 8 个；
- `(2,64,32,32)` 用例仅写出 131,072 个元素中的 2,048 个。

生成的 CUDA kernel 同时包含提前改写的累计赋值和模板原有的
`tn1=tn1+tn2`/`tn0=tn0+tn1`。这使 `tnum` 位段发生二次偏移，launch 中大量逻辑
索引没有对应线程。修复没有放宽网络容差，也没有切换到 CPU fallback。

## Focused verification

| Gate | Result |
| --- | --- |
| 大 NCHW channel-bias CUDA regression | 1 passed in 171.78s |
| 完整 `test_parallel_pass.py`（CPU/CUDA） | 8 passed in 168.39s |
| 完整 `test_cuda.py` | 5 passed, 1 skipped in 19.61s |
| 三个原失败 CUDA 网络 | 3 passed in 564.16s |
| ResNet18/ViT/GPT-2/diffusion UNet CPU+CUDA | 8 passed in 58.07s |

网络测试从真实 PyTorch 复制权重，比较前向输出、所有浮点输入梯度和全部可训练参数
梯度。CUDA 子类在真实 NVIDIA GeForce RTX 4090 上执行 Jittor 计算；PyTorch CPU
结果作为独立语义 oracle。

## Maintained CUDA gate

```bash
CUDA_VISIBLE_DEVICES=<allocated-device> \
nvcc_path="$(command -v nvcc)" \
JITTOR_CI_PYTHON="$(command -v python)" \
use_parallel_op_compiler=0 \
python -m nox -s cuda
```

Cold-cache results at `0a3458b3`:

| Stage | Result | Time |
| --- | --- | ---: |
| CUDA backend directory | 94 passed, 1 skipped, 1 xfailed | 25:41 |
| dtype coverage | 6 passed | 7:54 |
| CPU/CUDA device parity | 221 passed, 6 skipped, 12 warnings | 2:50:41 |
| Torch CUDA TF32 | 2 passed | 4.03s |
| strict CUDA OpInfo | 221 passed, 2 skipped, 4 xfailed | 10:54 |

Nox reported `Session cuda was successful in 4 hours`. The skip/xfail set is
unchanged from the accepted CUDA report: two logical bool reductions and four narrow integer
reductions remain explicit known limitations. FFT metric casts retain twelve existing
`ComplexWarning` messages.

## CPU and independent Torch gate

With `JITTOR_REQUIRE_REAL_TORCH=1`, the isolated CPU nox session succeeded in
27 minutes. Its independent oracle subset produced `18 passed, 1 skipped`; the skip is the
CUDA-only cumprod test in the CPU session. The same gate fails closed if `REAL_TORCH_SITE`
is absent or does not contain a binary `torch._C`.

Supporting repository gates:

```text
bash agent/scripts/check_repo_layout.sh
  repository layout OK

python -m pytest -q tests/structure
  218 passed in 230.22s
```

## Boundaries

- The four network builders are compact correctness models, not proof of full training
  convergence or real-scale throughput.
- No speed claim is made; the todo still requires Jittor to be no slower than PyTorch.
- NPU and ROCm were unavailable and remain unverified rather than passed.
- Optional downstream libraries and external assets remain separate gates.
- Raw logs, generated kernels, compiler caches, and the local diagnostic probe remain under
  `$JITTOR_LAB_ROOT` and are not versioned.

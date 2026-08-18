# 对拍框架、下游库覆盖与测试套件模式切分

日期：2026-08-18

代码基线：`3a3b04fa`（分支 `2.0`）

环境：`jt311`（Python 3.11.15，torch-shim 部署）、`jt312b`（Python 3.12.13，真实
PyTorch 2.12.1+cu126 与 torchvision 0.27.1+cu126），4x RTX 4090，CUDA 12.2/cuDNN 8。

## 结果

### 常见网络对拍（前向 + 反向）

新增 `tests/models/test_network_parity.py`。权重从真实 PyTorch 侧拷入 Jittor 侧，
比较输出、输入梯度和逐参数梯度，并断言实际比较的梯度数量，避免名称不匹配时退化成
只比前向。

| 网络 | CPU | CUDA | 比较的梯度数 |
| --- | --- | --- | ---: |
| ResNet-18（torchvision） | 通过 | 通过 | 63 |
| ViT | 通过 | 通过 | 33 |
| GPT-2 | 通过 | 通过 | 29 |
| 扩散 UNet | 通过 | 通过 | 45 |

容差：CPU 前向 2e-4 / 反向 2e-3，CUDA 前向 1e-3 / 反向 6e-3。

### 下游库数值对拍

新增 `tests/compat/torch/test_ecosystem_parity.py`。真实 PyTorch 与 Jittor 都要占用
`torch` 命名空间，无法同进程比较，因此同一个用例分别在两个解释器里执行：真实
PyTorch 侧生成权重与参考值，Jittor 侧加载同一份权重复算。权重按
`named_parameters + named_buffers` 传递并断言两侧完全覆盖——`state_dict` 不一定完整
（ms-swift 的 tuner 只报告 adapter），按 `state_dict` 传会让两侧骨干各自随机初始化。

CPU 与 CUDA 各 12 个用例全部通过：transformers 的 gpt2/llama/bert/vit/t5/whisper、
diffusers 的 UNet2DModel 与 DiTTransformer2DModel、peft LoRA、ms-swift LoRA、
mmcv `ConvModule`、mmengine `BaseModule`。

误差度量按整次比较的梯度量级设下限。注意力 key bias 的梯度在数学上恒为零，两侧都只
有 1e-8 量级的浮点噪声，按该张量自身最大值归一化会把它放大成 1.96e-2 的假失败。

ms-swift 4.5.2 与 peft 0.19 的 LoRA `Linear` 签名不兼容，**真实 PyTorch 2.12 上报
完全相同的 `TypeError`**，属于上游版本组合问题；peft 0.17.1 下两侧都能跑通。

### 测试套件模式切分

Torch 兼容模式是进程级的，会改变惰性执行、归约默认值与梯度语义。`conftest` 之前在
选择 `tests` 目录时给整个套件打开该模式。同一批文件在两种模式下的对比：shim 模式
13 失败，原生模式 4 失败。

| 会话 | 修正前 | 修正后 |
| --- | --- | --- |
| Torch 模式 | 36 failed，4 errors | 6 failed，0 errors，1447 passed |
| 原生模式 | 收集阶段即报错，运行到 70% 无汇总退出 | 收集干净 1624 条 |

`tests/ops` 下只有 `test_ops.py` 使用 OpInfo，其余断言的是原生惰性执行行为，已移出
Torch 模式。`tools/run_test_suite.py` 分两个会话跑完整套件并汇总。

## 修正的缺陷

| 缺陷 | 影响 | 证据 |
| --- | --- | --- |
| shim 预处理把 `use_mkl` 设为 0 | 去掉 `mkl_conv`/`mkl_matmul` relay，CPU 卷积 1.4ms→156ms、512x512 矩阵乘 0.6ms→11ms | `test_torch_bootstrap.py::test_preflight_leaves_onednn_enabled` |
| `reshape`/`view` 丢弃 `shape=`/`size=` 关键字 | diffusers DiT unpatchify 直接崩 | `test_torch_compat_reduce_shape.py::TestReshapeKeywordShape` |
| `state_dict` 不接受 `destination`/`prefix`/`keep_vars` | ms-swift 适配后的模型无法保存 | `test_torch_compat_serialize.py::TestStateDictTorchKeywords` |
| `home()` 把 `JITTOR_HOME` 写回共享配置，且原地重写 | 一次隔离运行变成整机默认；并发启动读到截断 JSON 直接崩 | `tests/compiler/test_jittor_home.py` |
| `print_trace` 无限等待 gdb | gdb 挂住或被崩溃收集器接管时整个进程一起卡死 | `test_tracer.py::test_gdb_backtrace_wait_is_bounded` |
| Python 变量追踪器在 3.11+ 直接 `LOGf` | `trace_py_var` 全部功能不可用 | `test_trace_var.py`，3.11 与 3.12 各 5 passed |
| 13 个带参数的模块级 `test_*` 辅助函数 | 被 pytest 收集后因缺 fixture 报错 | `test_pytest_contract.py::test_module_level_helpers_are_not_named_like_tests` |
| 结构门禁与布局脚本按文件系统扫描 | 跑过 lint 或 import 过包的工作区被判失败 | 布局脚本 4 种情况验证：干净 0、根目录多余文件 1、notebook 产物 1、`__pycache__` 0 |
| 下载外部数据集的用例无标记 | 主机不可达时阻塞到 900 秒超时而非快速跳过 | `network` 标记 + `--network` 开关 |
| `manual` 标记无人消费 | notebook 冒烟测试在整轮第 1100+ 条时把进程打死 | 整树运行跳过、显式路径仍运行 |

## Python 3.12

`jittor.selftest` 在 3.12 上通过（forward `(1.0, 4.0, 9.0)`，gradient `(2.0, 4.0, 6.0)`）。
完整套件在 3.11 与 3.12 上的失败集合逐条 diff：118 条共有，3.12 独有 4 条，3.11 独有
7 条。**没有 3.12 特有的框架性破坏。**

## 教程

新增三个纯计图、CPU 一分钟内跑完的教程，已纳入离线冒烟集：`transformer.md`（复制
任务 100% 准确率）、`diffusion.md`（完整 DDPM，双峰分布覆盖 48.7%/51.3%）、
`lora.md`（低秩适配，103x 改善且合并后最大差 2.4e-6）。

## 速度

CPU（`use_mkl` 修正后，空载测量）：4x64x32x32 卷积 Jittor 1.54ms 对 PyTorch 1.06ms
（1.45x），512x512 矩阵乘 0.70ms 对 0.54ms（1.29x）。

CUDA 下游库单步（小模型，启动开销主导）：peft LoRA 0.62x、llama 0.63x、ViT 0.78x
快于 PyTorch；gpt2 1.53x、diffusers UNet2D 1.46x、bert 1.94x 慢于 PyTorch。

`JITTOR_ECOSYSTEM_SPEED_RATIO` 可以把比值变成断言；默认只打印，避免共享机器上的
测量抖动造成假失败。

## 未解决

- 编译器 tuner 类测试（`test_parallel_pass`、`test_fused_op`、`test_reorder_tuner`、
  `test_conv_tuner`、`test_group_conv_tuner`、`test_broadcast_tuner`、
  `test_matmul_tuner`、`test_reduce_tuner`、`test_log`）依赖 JIT 缓存状态：同一组用例
  冷缓存 3 失败、热缓存 1 失败。这批断言的是 `log_capture_scope` 捕获到的 tuner 日志
  条数，需要先确定冷启动路径上 tuner 日志的去向再改。
- Torch 模式剩余 5 条整轮失败在单独运行时全部通过，属于顺序依赖；其中两条是
  installer 幂等性断言，需要查清是哪个前置用例改动了 `torch.*` 模块图。
- 下游库速度：CUDA 上 bert/gpt2/diffusers 仍慢于 PyTorch。当前用例都是极小模型，
  需要补一组真实规模的测量再判断是内核问题还是调度开销。

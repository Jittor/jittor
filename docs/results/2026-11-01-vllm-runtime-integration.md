# vLLM 所需通用运行时能力的主仓整合验证

[English](2026-11-01-vllm-runtime-integration.en.md)

- 状态：整合后的验证进行中；下表历史结果不能代替当前基线的验收。
- 日期：2026-11-01。
- 远端基线：`origin/2.0-refactor`，`88b48a1f18b2964f05090f65561e07ab64343715`。
- 整合代码：四项修复已重放至 `5bd26b7859c9589a6d641bbe78ee46ef987593fc`；没有覆盖远端后续实现。
- 范围：cuFFT/CUB 编译兼容、accelerator 显存查询、dtype-view 存储与写回语义、Triton constexpr 参数和缓存身份。
- 维护者：Jittor 核心与 Torch 兼容层维护者。
- 复查条件：相关核心/前端接口、CUDA 工具链、Triton、上游 vLLM 或已声明设备变化。

## 修复归属与基线

这些能力属于框架或通用 Torch 接口，应随主仓源码构建安装。vLLM 专属协议仍由独立
adapter 负责；此报告不把核心修复放进 adapter，也不将旧版本号相同的 wheel 视为已具备修复。

| 能力 | 主仓整合提交 | 行为 |
|---|---|---|
| cuFFT/CUB | `0467f3d991a9ae6b16fb3faf81041be4ab97212e` | 兼容 cuFFT 12 删除的枚举与 CUB 3 的 reduction 接口，保留旧版本分支 |
| 显存查询 | `379db21f22f6b69b9e5c730d0fcc9134180cd49d` | `torch.accelerator.get_memory_info` 复用已有 CUDA 查询 |
| dtype-view | `267888bded3f2f68ea38ba6c42e3d108b8503b70` | 合法非连续布局保持字节 stride、offset、共享存储与双向写回；dtype-view 不参与求导 |
| Triton 参数 | `5bd26b7859c9589a6d641bbe78ee46ef987593fc` | `None` 按 constexpr 特化；完整签名区分缓存身份，运行时参数打包不包含 constexpr |

复核另外将现代 Triton 的 native-specialization helper 检查与通用 ASTSource 缓存检查拆分：
旧 Triton 没有该 helper 时只跳过前者，不跳过缓存身份断言。

## 历史证据：2026-10-30，代码 `40d4414f4`

以下结果来自独立开发基线 `40d4414f45b89b7cb7f5efdc06ca622b674d772a`，
不是本次与远端整合后的测试结果。环境为 Python 3.11、RTX 4090、CUDA 13.0.88 工具链、
cuFFT 12/CUB 3、Triton 3.7.1。

| 范围 | 历史结果 |
|---|---|
| CPU core tier | 327 通过、45 环境/opt-in 跳过、1 预期失败 |
| dtype-view CPU 与邻接契约 | 27 通过；另有 9 个共享语义 case 通过独立 PyTorch 检查 |
| dtype-view CUDA | 11 通过 |
| accelerator 显存查询 CUDA | 3 通过 |
| cuFFT CUDA | 4 通过 |
| softmax CUDA | 5 通过，包含调度、梯度与独立数值参考 |
| Triton CUDA 后端文件 | 23 通过、3 个多设备用例因仅分配单卡跳过 |

历史原始日志和构建产物位于 `$JITTOR_LAB_ROOT/_state/vllm-product/20261030/`，未版本化。
历史 vLLM 固定版本为 0.29.0，提交 `98dff2a81d747d1dba01a47f939f48c3526d4206`；
这些框架回归不等于 vLLM 全库支持声明。

## 当前整合基线的验收

当前结果待补充。复验入口包括布局检查、`tests/structure`、CPU core/smoke，以及
`compat/tests/torch/test_dtype_view_strides.py`、`test_accelerator_memory_info.py`、
`tests/backends/cuda/test_cufft_errors.py`、softmax CUDA 测试和
`compat/tests/triton/test_triton_backend.py`。安装验收必须使用本次正常构建的 core/frontend，
再从独立环境调用官方 vLLM API/服务，不能复用手工补丁 wheel 冒充当前源码。

## 未覆盖与剩余限制

不新增 ROCm、Corex、NPU 或多卡验收声明。Storage façade 指针、一般带 gap 的
`as_strided` 和历史部分 split Python 写回路径仍有独立限制，见
[KI-COMPAT-024](../../agent/manuals/known-issues.md#ki-compat-024-storage-façades-and-some-constructed-views-retain-alias-gaps)。
验证本次 dtype-view 的物理共享应使用 Tensor.data_ptr，并检查 stride、offset 与写穿；
正确读值或外部 kernel 写入不能替代一般 Python alias 语义验收。

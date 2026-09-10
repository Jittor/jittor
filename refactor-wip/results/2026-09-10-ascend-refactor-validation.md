# 2.0-refactor 原生与独立 Torch compat 的 Ascend 复验

- 日期：2026-09-10；状态：两项已复现OpInfo缺口已修复；原生选择集215通过/2跳过；独立compat与定向OpInfo 65通过/0跳过；完整CPU structure 1357通过/4跳过，后续受影响结构回归通过
- 初始基线：`1a1f175e7d7f2403353bb7e4d5da13f42f02dff8`
- 历史续测基线：`2.0-refactor` / `860863960f9fd38a1c28398d54268dd0166a7414`
- 当前续测基线：远端 `2.0-refactor` / `7df15e31`；本地修复重放后 `952b5042` 加本轮工作树补丁
- 修改状态：测试时工作树补丁未提交；最终源码随本报告所在修复提交保存，不能把测试时间追溯为提交后
- Owner：Jittor 核心、ACL 后端、独立 Torch compat 与测试基础设施维护者
- 当前状态入口：[整改看板](../architecture/refactor-board.md)

## 结论与验收范围

已在 Ascend 910B3 上执行原生 Jittor 与独立 Torch frontend。本轮已修复此前两项
OpInfo失败：Abs的NPU dtype不再继承CUDA声明，支持的float16/float32与明确拒绝的
float64均经过真机测试；非全局 `adaptive_avg_pool2d` 已有CANN前向/反向实现，
覆盖非整除重叠窗口、输出大于输入、3D/4D输入以及`None`保留轴。
池化float16/bfloat16/float32保持输入dtype，按独立NumPy窗口与梯度累加参考验算。

当前基线原生Jittor完整选择集已 **215 passed / 2 skipped，886.79s**，217项全部处理。
两个skip来自既有FlashAttention类入口缺失，函数式SDPA另有真机通过证据。
独立compat与定向OpInfo合并轮已 **65 passed / 0 skipped，25.44s**，65项全部处理。选择集含58项独立compat及7项原生OpInfo，其中12项为CPU共享语义用例，
不能把65项全部称为独立compat的NPU计算。完整CPU structure已 **1357 passed / 4 skipped，
914.54s**，1361项全部处理；最后归约及nox改动另通过370项受影响结构回归。

65项首轮发现ACL归约descriptor错误压掉保留维度，55通过/1失败后中断。
修复后先通过28项CPU/NPU归约前后向，再完整重跑65项得到上述结果；
最终日志为`compat/logs/final-65-compat-opinfo-fixed.log`，失败轮保留用于缺陷追踪。

**本报告不声称全库或全部OpInfo算子验收通过。** 扩展扫描曾在后续`amax`失败处
停止，Python层keepdim、标量参考及新增回归暴露的NPU descriptor问题均已修复，
上述定向回归已通过。
之后的全库扫描尚未完成。
未单独审计的NPU dtype条目在OpInfo报告中标为 **UNVERIFIED**，仍是待验证候选，
不能从CUDA声明或少量已通过的算子推导支持。历史`86086396`的原生185通过/2跳过、
独立compat38通过/0跳过保留在下表，不冒充当前补丁的验收结果；其中compat包含CPU共享用例。

## 环境、复现与证据位置

| 项目 | 记录 |
| --- | --- |
| 设备 / 驱动 / CANN | Ascend 910B3 / 25.5.1 / 9.0.0 |
| Python / NumPy / pytest | 3.9.9 / 1.26.4 / 7.4.4 |
| 宿主编译器 | g++ 10.3.1，aarch64 Linux |
| 源码与发行版 | 当前 checkout；task venv 安装独立 `jittor-torch` 发行版 |
| 计算证据 | 运算同步后 device 驻留、独立数值/梯度参考、运行时零回退 guard |
| 原始产物根 | `$JITTOR_LAB_ROOT/_state/ascend-refactor/`，不进入 Git |

恢复任务时先同步目标远端：从初始基线快进6个上游提交至续测基线，改动集中在
工具与known-issues；已保护并重新应用本地补丁，known-issues三方整合保留双方。
下表中明确标注“续测”的结果属于86086396；先前门禁、性能JSON仍属于1a1f175e，
不追溯更改基线。core未被这6个上游提交修改，不据此免除本地修复的定向复验。
本轮恢复后再次以远端为准同步至`7df15e31`，本地既有修复重放至`952b5042`后再修改。
以下新增复验记录基于该HEAD加本轮补丁；旧日志仍保留原基线，性能未在新基线重测。

下文日志路径相对此产物根。设备映射、环境完整快照、个人路径、机器标识只保留在
外部原始记录，不写入本文。读取 tensor 到 NumPy 会改变驻留，故先检查设备再取值。
CPU reference 阶段不是 NPU fallback；负向拒绝测试的回退尝试也不计作 NPU 成功计算。

复现需先加载 CANN 环境，并把 `ASCEND_RT_VISIBLE_DEVICES` 指向已分配设备；
`VALIDATION_PYTHON` 选择上述已安装 task venv，`VALIDATION_CHECKOUT` 指向本分支。
原生与 compat、性能各使用独立进程和缓存；以下给出不含个人路径的等价启动方式：

```bash
source "$CANN_SET_ENV"
export PYTHONPATH="$VALIDATION_CHECKOUT/python"
export backend_fallback=error DISABLE_MULTIPROCESSING=1
export JITTOR_TEST_DEVICES=npu JITTOR_TEST_REQUIRE_ACL=1
export JITTOR_TORCH_KEEP_TMPDIR=1
validation_run="$JITTOR_LAB_ROOT/_state/ascend-refactor"
cd "$VALIDATION_CHECKOUT"
"$VALIDATION_PYTHON" -m pip install -e ./compat --no-deps --no-build-isolation
mkdir -p "$validation_run/native/jittor-home" "$validation_run/native/tmp" \
  "$validation_run/compat/jittor-home" "$validation_run/compat/tmp"
JITTOR_TORCH_SHIM=0 JITTOR_HOME="$validation_run/native/jittor-home" \
  TMPDIR="$validation_run/native/tmp" \
  "$VALIDATION_PYTHON" -m pytest -vv --timeout=600 -x \
  tests/backends/acl/test_acl.py tests/backends/acl/test_aclop.py \
  tests/backends/acl/test_acl_indexing.py tests/backends/acl/test_acl_random.py \
  tests/backends/acl/test_acl_scope.py tests/backends/acl/test_acl_pooling.py \
  tests/backends/acl/test_acl_adaptive_pool.py tests/backends/acl/test_acl_reduce_keepdims.py \
  tests/ops/test_opinfo_npu_dtypes.py \
  tests/ops/test_comparison_tolerances.py tests/ops/test_adaptive_pool_output_size.py \
  tests/ops/test_floor_divide.py::TestFloorDivideNPU \
  tests/debug/test_kernel_traps.py::TestKernelTraps::test_nan_handling_isfinite_isnan_isinf \
  tests/ops/test_fusion_correctness.py::TestFusionCorrectness::test_float_comparisons_with_nan
JITTOR_TORCH_SHIM=1 JITTOR_HOME="$validation_run/compat/jittor-home" \
  TMPDIR="$validation_run/compat/tmp" \
  "$VALIDATION_PYTHON" -m pytest -q --timeout=600 -x \
  tests/backends/acl/test_acl_torch_compat.py compat/tests/torch/test_contiguous_storage.py \
  compat/tests/torch/test_slice_assignment.py \
  compat/tests/torch/test_adaptive_avg_pool2d.py \
  compat/tests/torch/test_extreme_reduction_scalar.py \
  tests/ops/test_ops.py::TestCommonNPU::test_reference_abs_float16 \
  tests/ops/test_ops.py::TestCommonNPU::test_reference_abs_float32 \
  tests/ops/test_ops.py::TestCommonNPU::test_reference_adaptive_avg_pool2d_float16 \
  tests/ops/test_ops.py::TestCommonNPU::test_reference_adaptive_avg_pool2d_bfloat16 \
  tests/ops/test_ops.py::TestCommonNPU::test_reference_adaptive_avg_pool2d_float32 \
  tests/ops/test_ops.py::TestCommonNPU::test_reference_amax_float32 \
  tests/ops/test_ops.py::TestCommonNPU::test_reference_amin_float32
```

累计原生复验实际复用了外部名为 `cpu/jittor-home` 的已热缓存；目录名称不是执行
设备，真机用例由运行时 ACL 标志与驻留/回退证据验收。首次核心冷构建可单独并行
预热后退出，算子执行验证串行。另一次重复 shim 冷构建被 SIGINT 停止，未计结果。

## 缺陷、根因与修后证据

所有修复均基于真实失败或独立最小复现；下面保留首次根因日志，不重复罗列被前序
失败图污染的后续错误。表内通过数来自不同选择范围，不能简单相加当作总覆盖数。

| 缺陷与根因 | 修改 | 原始证据与验证 |
| --- | --- | --- |
| ACL 缺共享加速器宏，`fetch_op.cc` 无法识别 mutex / event queue / current device | 增加 `HAS_ACCELERATOR` | `native/logs/probe-before.log`；后续启动 matmul 成功 |
| normalization 错误分支使用未绑定的 runner 名 | helper 显式接收 owner | `native/logs/probe-norm-before.log`；后续启动及 norm 路径执行 |
| provider runtime 未进入 BuildConfig，导入缺 `mallocWorkSpace`；JIT compiler 配置遗漏 | 独立 `BuildSource` 注册 backend/workspace，补 accelerator/JIT 契约 | `native/logs/probe-link-before.log`；host 配置14 passed |
| ACL 错误启动 CUDA fake library 的 cuDNN 构建 | 关闭该历史构建路径 | `native/logs/probe-cudnn-before.log`；`probe.log` 得到正确2×2 matmul |
| 专用 CANN runner 被通用 registry guard 拒绝 | 显式 Direct 分类，generic 缺注册仍报错 | `native/logs/tests-first.log`、`tests-second.log`；All/Any、KV 等后续真机回归 |
| random 历史名映射、执行 flag 和 provider compile 回调遗漏 | 注册当前 native owner 并保留 JIT 回调 | `native/logs/random.log`；`random-second.log` 4 passed，含负向 float64 拒绝契约 |
| `flag_scope` alias 逐项备份污染；异常恢复 setter 重新提交失败图 | 写前一次性 snapshot，失败设备恢复不重复同步 | `native/logs/tests-first.log`；`scope-random.log` 3 passed，host alias/异常回归 |
| forward/backward 共用 typed code-data namespace；backward 属性安装晚于 run | 独立字段命名空间，先安装属性再执行 | `host/logs/acl-mixed-attributes-before.log`；mixed/order后续各40 passed（重叠选择，不计80） |
| KV 部分写入面向 scalar broadcast-zero 存储，且缺旧 cache 真实依赖 | 先物化并发布 holder；旧 cache 为输入，不同输出 buffer 先 D2D 保留 | `native/logs/kv-before.log`；`kv-dense-native.log` finite、未写区域0；三条paged回归2+1 passed |
| BMM owner 名与 schema 不一致，rank4直接进入仅接受3D的 CANN API；广播梯度归约不完整 | canonical名、batch广播/展平/还原、梯度sum-to-shape | `native/logs/kv-paged-tests.log`、`compat/logs/activation-sdpa-fixed.log`；host有限差分8 passed，两条SDPA均在最终compat门禁通过 |
| AdamW step原生标量为rank0，CANN tiling要求长度1向量，报561103 | 仅在SDK边界构造`[1]` descriptor，不改变单个step值 | `native/logs/adamw-cann-before.log`；最终compat含BF16两步参数/状态精确对拍 |
| AdamW 状态是广播零 view，compat `.contiguous()` 却直接返回自身 | 真正物化 dense storage；保留兼容 Tensor 身份与梯度；optimizer 发布更新状态 | `compat/logs/final-focused-2.log`、`final-compat.log`；AdamW及CPU/NPU contiguous数值回归通过 |
| ReLU/LeakyReLU构造未显式发布属性，历史scale参数与negative_slope别名不完整 | 补构造参数、inplace/negative_slope属性和native scale兼容 | 最终compat构造属性/前向/梯度通过；`host/logs/activation-native-cpu.log` native历史scale参数1 passed |
| trainable mask guard 仍依赖旧 stop-grad 字段 | 使用当前 requires-grad 语义 | 最终compat两条SDPA含普通 additive mask接收、可训练mask明确拒绝 |
| Pool geometry helper 迁移后仍从旧包导入 | 使用 `pooling.average` owner；host fixture不再伪造旧导出 | `native/logs/tests-fifth.log`；`resnet-pooling-after.log` 1 passed |
| AvgPool backward SDK参数顺序错位，含padding梯度数值错误 | 对齐ceilMode/countIncludePad/divisorOverride/cubeMathType | `native/logs/pooling-numerical-after.log` 修前1 failed/3 passed；`pooling-attributes-after.log` 8 passed |
| SwiGlu 专用 runner 缺 typed schema | 补齐生产属性通道 | `native/logs/tests-sixth.log`；`swiglu-after.log` 1 passed |
| 公开批量matmul_transpose入口错误展平batch | 恢复batch分发；7形状×CPU/NPU前向及两侧有限差分梯度，连同原grad_h_a/d共3 passed | `native/logs/tests-seventh.log::test_grad_h_a`；`matmul-transpose-after.log` 44.59s |
| ReLU/LeakyReLU/Sigmoid/SiLU/Softmax旧前向fixture用整数，违反既有float-only契约 | fixture明确FP32；五种激活补整数TypeError拒绝，不修改provider、不新增skip | `native/logs/activation-fixtures-before.log`最小2失败；`relu-fixtures-after.log`3通过、`softmax-fixture-after.log`2通过 |
| slice RHS包装成0D后未广播，部分写入缺base依赖；广播梯度未还原RHS形状 | 按公开slice形状广播再还原SDK轴，物化base并保留输入边；native/ACL梯度sum-to-shape | `native/logs/setitem-before.log`；`setitem-after-second.log`3 passed/63.78s；续测独立compat slice/mask14通过 |
| bool mask标量RHS被错误要求为1D，标量梯度未归约 | 0D标量按masked-scatter需求展开，空/部分/全mask梯度正确归约；1D RHS仍严格校验长度 | `native/logs/tests-tenth.log::test_setitem_3`；`setitem-mask-after-first.log`2 passed/16.25s；续测compat14通过 |
| 旧Softmax max-tie参考给每个并列最大值完整梯度，与当前均分契约不符 | 等logits产生精确0.5概率，独立解析梯度±0.125；CPU/NPU均保留默认allclose | `native/logs/tests-thirteenth.log::test_softmax_grad`；`softmax-tie-contract-final.log`1 passed/28.57s |
| 旧测试调用已移除的triu_入口 | 使用当前jt.triu并保留数值断言，不恢复legacy alias | `native/logs/remaining-first.log`；`remaining-second.log`尾部36通过 |
| 旧编译日志文本被当作设备执行证明 | 改同步后驻留、数值/梯度与回退证据 | 缓存命中不再造成假失败；未降低数值门槛 |

KV 原始探针的 K/V 最大误差为 `0.00097581744` / `0.00097092986`，符合 BF16
舍入量级；output/cache 均 finite，未写区域绝对值0。通用 CodeOp writable-view
构造安全边界仍需单独证据，本次不声称所有共享可写 view 问题都已解决。

## 本轮追加修复与验收

| 缺陷与根因 | 修改及范围 | 证据 |
| --- | --- | --- |
| KI-BACKEND-009：NPU dtype错误继承CUDA，Abs选择不支持的float64 | 新增逐OpInfo独立`dtypesIfNPU`；Abs声明float16/float32，保留CPU/CUDA float64覆盖；未审计项明确UNVERIFIED，不通过空集合隐藏 | `compat/logs/opinfo-npu-dtypes-fixed-env.log` 12 passed；含Abs两种支持dtype执行、float64拒绝与声明结构合同 |
| KI-BACKEND-010：非全局adaptive pooling进入ACL未注册的reindex | 注册CANN AdaptiveAvgPool2d及梯度runner，typed属性传递output_size；公开入口正规化3D/4D、整数和None轴，拒绝无效维度/输出大小 | `native/logs/adaptive-pool-final.log` 2 passed；`compat/logs/adaptive-pool-compat-final.log` 12 passed，覆盖三dtype前后向 |
| amax/amin全维归约忽略keepdim；OpInfo参考把真实标量强制转成1D | 归约传递keepdims；NumPy参考保留0D；追加标量rank、保留维度和并列极值梯度测试 | `compat/logs/opinfo-extrema-abs-pool-final.log` 定向11 passed / 522 deselected；新独立compat keepdim=True/NPU样本进一步暴露descriptor压维问题，修后28项通过，见下一行 |
| ACL全轴归约无条件把输出descriptor改为标量，破坏keepdims=True | Sum/Mean/Max/Min保留singleton维度；Prod全归约SDK无keepdims参数，保留其scalar descriptor特殊处理 | `compat/logs/reduce-keepdim-focused-final.log` 28 passed；20项CPU/NPU×五归约×keepdims，另8项独立compat；每种归约检查rank1/2、数值和加权梯度 |
| 比较助手在NumPy staging后丢失BF16原dtype，错误套用FP32容差 | 在传输前读取tensor精度，沿用已有BF16容差，显式容差优先；FP32/FP64严度不变 | `tests/ops/test_comparison_tolerances.py` 3项正负断言；纳入最终原生选择集 |
| 非缓存allocator不支持共享view；释放broadcast/strided view错误使用逻辑tensor大小 | 为不支持共享的策略加共享所有权包装，仍保留原分配策略；按实际storage span释放 | `structure-cpu/logs/shared-allocator-regression-final.log` 17 passed / 5 CUDA skipped；`native/logs/shared-allocator-npu-final.log` 12 passed / 6 deselected，其中10项CPU/NPU五种策略回归、2项池化入参回归 |
| structure夹具/路径/接口合同过时以及环境隔离缺失 | 更新当前typed code_program夹具、错误分类、optimizer与runtime接口合同、文档路径；修正子进程依赖/构建环境和Python3.9清理兼容 | 中间完整轮 `structure-cpu/logs/followup-full.log` 1338 passed / 18 failed / 5 skipped；剩余失败已修复；`structure-cpu/logs/final-full-fixed.log` 最终1357 passed / 4 skipped |
| 收集测试时查询后端并吞异常；flag清理不被结构门禁识别 | CUDA capability查询移至setup，失败不转skip；PEFT仅缺包可skip；可写flag显式finally恢复，不可写flag保留拒绝和原值断言 | `structure-cpu/logs/hygiene-targeted.log` 13 passed；未扩充scanner豁免 |

两项KI-BACKEND记录已关闭并移出活动问题总账，原始失败与修后证据在本报告保留。
Abs不支持float64的事实仍由负向测试明确保证；关闭的是错误能力声明，不是新增float64实现。
池化覆盖一般窗口，未依赖CPU fallback，也不是只对原先4×4→2×2样本进行特判。

| 当前基线最终范围 | 结果 | 原始日志 |
| --- | --- | --- |
| 原生完整选择集，descriptor修后重跑 | 215 passed / 2 skipped，587 warnings，886.79s；217 collected / 217 executed，exit0 | `native/logs/final-followup-native-fixed.log` |
| 独立compat与定向OpInfo，遇错停止 | 55 passed / 1 failed，604.91s；65 collected / 56 executed，未完成；后续修复保留维度descriptor，首轮结果不改写 | `compat/logs/final-65-compat-opinfo.log` |
| 独立compat与定向OpInfo，descriptor修后重跑 | 65 passed / 0 skipped，196 warnings，25.44s；65 collected / 65 executed，exit0 | `compat/logs/final-65-compat-opinfo-fixed.log` |
| 完整CPU structure | 1357 passed / 4 skipped，914.54s；1361 collected / 1361 executed | `structure-cpu/logs/final-full-fixed.log` |
| descriptor修后受影响structure | 370 passed / 0 skipped，141.82s，exit0；覆盖ACL结构、nox、grad、打包、cleanup、pytest与flag合同 | `structure-cpu/logs/post-reduce-affected-final.log` |
| native-only dtype-preservation及warning-as-error | 6 passed / 0 skipped，1.33s；独立native模式CPU进程 | `structure-cpu/logs/native-mode-extra-final.log` |

最终compat轮仍报告holder/lived-var各`0→1`的状态诊断（未计失败）；通过计数不等于证明零残留。

本轮structure的4个skip为2项CUDA专项nn测试，以及2项环境配置owner文件的既有豁免。
`tests/structure/backends/acl/test_acl_dtype_preservation.py`按native-only模式规则在该
CPU/shim结构轮收集0项，其5项已与warning-as-error另在独立native模式CPU进程通过，
不把模式排除冒充执行，也不再列为未验证缺口。完整结构轮早于随后追加的ACL归约
descriptor修复及nox目标更新；后者另有28项CPU/NPU数值/梯度回归及370项受影响structure复验，
完整结构结果不追溯到后续补丁。最后的构建边界子进程环境清理补丁另通过19项定向回归
（`host/logs/build-boundary-final.log`，0.53s），避免canonical `JT_BUILD_*`污染隔离构建。

`native/logs/final-followup-native.log`原197项轮次为冻结源码修改窗口而中断，
不计完整门禁；后续使用`final-followup-native-fixed.log`补入20项归约回归，完整重跑217项。

## 历史验收记录与范围边界

下表是初始`1a1f175e`及历史续测`86086396`上的原始记录。失败和中断结果保留用于
追踪修复，不作为本轮未修复项重复列账；表内“续测”“最终”均指该历史轮次。

| 范围 | 结果 | 原始日志 |
| --- | --- | --- |
| 原生完整选择集（续测） | 185 passed / 2 skipped，521 warnings，82.10s；187 collected/executed，exit0 | `native/logs/final-full.log` |
| 独立compat合并：ACL22 + contiguous2 + slice/mask14（续测） | 38 passed / 0 skipped，136 warnings，20.22s；其中额外CPU用例8项 | `compat/logs/final-combined.log` |
| slice/mask独立compat CPU/NPU（续测，包含于38项） | 14 passed / 0 skipped，17.70s | `compat/logs/slice-assignment-mask-final.log` |
| 原生slice（初始） / mask（续测）定向 | 3 passed / 2 passed，63.78s / 16.25s | `native/logs/setitem-after-second.log`、`setitem-mask-after-first.log` |
| 既有共享CPU setitem | 12 passed，81.41s；同步前core相同 | `host/logs/setitem-shared-final.log` |
| setitem相关structure（续测） | 122 passed / 20 failed，13.85s；20项均既有transpose fixture，无新增失败 | `host/logs/setitem-structure-final.log` |
| 原生 random/arg-reduce | 4 passed | `native/logs/random-second.log` |
| KV block128/BF16 fresh decode、prefill/decode | 2 passed + 1 passed | `native/logs/kv-decode-tests.log`、`kv-prefill-after-bmm.log` |
| 原生 pooling / ResNet / SwiGlu | 8 passed / 1 passed / 1 passed | `native/logs/pooling-attributes-after.log`、`resnet-pooling-after.log`、`swiglu-after.log` |
| 共享 CPU 数值与语义 | 18 passed / 5 skipped | `host/logs/cpu-shared-final.log` |
| 既有共享CPU矩阵回归 | 7 passed / 0 skipped，23.25s | `host/logs/matrix-shared-final.log` |
| native CPU LeakyReLU历史scale参数 | 1 passed | `host/logs/activation-native-cpu.log` |
| 完整 CPU structure | 1257 passed / 85 failed / 5 skipped，1347 collected/executed，1042.87s | `structure-cpu/logs/full.log` |
| 最终host定向追加 | 149 passed / 23 failed / 0 skipped，172 collected/executed，27.68s | `structure-cpu/logs/host-final.log` |
| manifest后续定向 | 1 passed；生成器check通过，未重跑完整structure | `structure-cpu/logs/manifest-after.log` |
| 扩大compat+OpInfo，遇错停止 | 24 passed / 1 failed | `compat/logs/final-suite.log` |
| OpInfo显式float16/float32选择，遇错停止 | 3 passed / 1 failed / 326 deselected | `compat/logs/opinfo-float16-float32.log` |

完整structure的85项分类见外部 `structure-cpu/logs/failure-classification.md`：
66项静态基线合同/fixture（28 ACL fixture、7 ACL文档、17错误边界静态guard、14其他），
12项环境/隔离、6项运行时或运行时合同（未用旧binary确认基线归因）、1项manifest。
Python3.9缺 `TestCase.enterContext`、缺pytest-xdist等环境项不等于框架修复。
当时后续manifest已通过、pool fixture子集14通过，但未重跑整体，保留该历史完整结果。
最终host追加的23项失败均属于已核实初始HEAD合同：20项transpose fixture缺code_program、
2项runner旧文档字面断言、1项flag_scope静态扫描报告的5处写入（后续复核包含已有scope恢复及预期拒绝写入）。没有把该轮
149 passed隐藏23项失败后当作全绿；MANIFEST生成检查及layout检查已通过。

原生最终轮的两个skip来自既有 `test_aclop` 中历史FlashAttention类不存在的
`test_flashattention` 与 `test_flashattention_grad`，本轮没有为失败新增skip。

历史扩展OpInfo分别在`test_reference_abs_float64`及
`test_reference_adaptive_avg_pool2d_float32`处停止，原始失败日志保留如上。
两项修复及关闭证据见“本轮追加修复与验收”，不能再把历史中断描述为当前仍待修复。
后续amax的Python语义修复已落地，独立compat新增NPU descriptor失败也已修复并通过28项定向回归；剩余全库算子扫描未完成。

模块构造参数及其前向/梯度已验证；构造后修改activation公开属性的动态配置不在
本轮保证内。CPU复核LeakyReLU把slope属性从0.2改成0.5后仍输出旧配置的-0.2，
属于原分支 `_FunctionModule` 缓存kwargs语义（`host/logs/activation-mutable-review.log`），
不能把构造时参数修复外推为动态修改属性受支持。

初始基线compat冻结轮曾为24 passed，续测合并轮已扩到38 passed；pytest整轮耗时
含编译/CPU参考等，不是算子性能。该历史compat轮报告holder/lived-var各1残留（非失败）；
该历史原生轮依次报告各`0→6→8→26`，indexing新增18中FFT有界缓存解释6、另12未解释；
前两个文件新增的8也保留原始观察。passing计数不等于没有状态诊断，
已解释的有界缓存也不直接等同于泄漏。

## 同步矩阵乘性能

使用 [可复用脚本](../../agent/skills/jittor-transformers-perf/scripts/benchmark_npu_frontends.py)，
每个模式一个进程，单独 `JITTOR_HOME`。原始JSON位于
`benchmark-{native,compat}/matmul-{256,1024}-r{1,2,3}.json`。

```bash
perf_script=agent/skills/jittor-transformers-perf/scripts/benchmark_npu_frontends.py
mkdir -p "$validation_run/benchmark-native/jittor-home" "$validation_run/benchmark-native/tmp" \
  "$validation_run/benchmark-compat/jittor-home" "$validation_run/benchmark-compat/tmp"
# 每轮顺序执行native与compat；各模式/尺寸重复3轮，实际结果不含首次编译。
JITTOR_TORCH_SHIM=0 JITTOR_HOME="$validation_run/benchmark-native/jittor-home" \
  TMPDIR="$validation_run/benchmark-native/tmp" \
  "$VALIDATION_PYTHON" "$perf_script" --mode native --m 256 --k 256 --n 256 \
  --output "$validation_run/benchmark-native/matmul-256-r1.json"
JITTOR_TORCH_SHIM=0 JITTOR_HOME="$validation_run/benchmark-compat/jittor-home" \
  TMPDIR="$validation_run/benchmark-compat/tmp" \
  "$VALIDATION_PYTHON" "$perf_script" --mode compat --m 256 --k 256 --n 256 \
  --output "$validation_run/benchmark-compat/matmul-256-r1.json"
```

统一FP32、seed20260910、3组输入、10次预热、每轮50次计时。窗口为Python构图到
设备同步完成，每次调用同步，不含D2H、冷编译、backward。每组输入计时前后都对
NumPy float64累积参考检查。12份JSON均passed、device驻留、fallback `0→0`；
最大绝对误差256为`1.81596e-6`，1024为`9.80110e-6`，阈值rtol/atol均`2e-4`。

单位ms；R1/R2/R3每格为“median, p95”。汇总取三个轮次统计量的中位数，
不是合并150个样本后的p95。

| M=K=N / 模式 | R1 / R2 / R3：median, p95 | 三轮median中位数 | 三轮p95中位数 |
| --- | --- | ---: | ---: |
| 256 / native | 0.562371, 0.577213 / 0.588707, 0.607435 / 0.550176, 0.588623 | 0.562371 | 0.588623 |
| 256 / compat | 1.009338, 1.027981 / 1.052307, 1.079814 / 1.010373, 1.036924 | 1.010373 | 1.036924 |
| 1024 / native | 0.564091, 0.583090 / 0.565587, 0.579667 / 0.571531, 0.587456 | 0.565587 | 0.583090 |
| 1024 / compat | 1.036667, 1.053607 / 1.024513, 1.048179 / 1.060918, 1.082069 | 1.036667 | 1.053607 |

这里的compat/native延迟比约1.80（256）和1.83（1024），仅描述本次同步前向窗口，
不外推为模型训练/推理性能。共享host仍有其他任务，未建立系统级独占或稳定负载基线。
全部JSON基于初始HEAD `1a1f175e`，不是续测HEAD `86086396`，且工作树未提交；native256-r1的tracked diff摘要与其余11份
不同（分别以`d93ef6ebbcb6`、`551d229b9516`开头），脚本摘要全部一致（`a99298054229`）。
因此它们不是同一冻结提交的严格性能对比；tracked diff摘要也不覆盖所有untracked文件。
原始JSON保存各轮环境、源码来源、状态与摘要，合并后仍应在冻结提交上重新测量。

后续记录沿用 [NPU缺陷与性能模板](../../docs/development/npu-validation-templates.md)，
未完成项由当前看板维护，不新建平行验收账本。

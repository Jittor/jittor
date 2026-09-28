# 2026-09-13：合并远端后的增量修复与 Ascend 验证

- 状态：原生、compat选择集、完整结构及严格注册补验均通过。
- 复查日期：2026-09-13。
- Owner：Jittor Ascend 验证维护者（本轮由 Codex 执行）。
- 复查触发：合并基线、运行库、后端实现或对应测试发生变化。

## 结论

远端 `2.0-refactor` 的 `cb823778bd1b59b0c8955d0e05a831190676e9ed` 已合并到本地原提交 `074b7b3a7be284585090cc9dbe54bb5358be562d` 的工作树，冲突按远端接口适配；本轮未提交、未推送。再次查询远端后 SHA 未变化。

原生 Jittor 选择集已完成：**231 通过、2 跳过，233 项全部处理**，115.12 秒、退出码 0。包含 CPU 共用及控制案例，不能把 231 项全部称为纯 NPU 用例。独立 compat 与定向 OpInfo 已65通过、0跳过，65项全部执行；完整结构修后已1358通过、4跳过，915.73秒；严格注册补验30项全部执行且通过。本报告不宣称全库 OpInfo 验收通过。

## 基线和证据口径

本轮最终原生测试源码补丁 SHA256 为 `c3fc11a1d4d2ab7c98c084c661c78621cd2f3497dd1a3d3e6e1bbc6c71accc48`，相对上述远端生成，测试期间源码未变。测试时合并未提交，不能事后描述成某新提交上的测试。

原始证据位于 `$JITTOR_LAB_ROOT/_state/ascend-refactor/postmerge/`。NPU轮次的 `.json` 保存 HEAD、MERGE_HEAD、参数、环境、patch SHA256、退出码和源码是否变化，同名 `.patch` 保存跟踪文件差异；结构轮次的 JSON 保存跟踪差异及未跟踪文件的复合摘要。`logs/*.log` 保存原始输出。不同选择集存在重叠，不直接累加通过数量。

|环境项|本轮使用|
|---|---|
|硬件与运行库|Ascend 910B3，驱动 25.5.1，CANN 9.0.0|
|主机工具链|aarch64 Linux，g++ 10.3.1|
|Python与测试依赖|Python 3.9.9，NumPy 1.26.4，pytest 7.4.4|
|原生代码|当前 worktree/python/jittor，JITTOR_TORCH_SHIM=0|
|独立compat|task venv中的jittor-torch发行版，JITTOR_TORCH_SHIM=1|
|后端约束|JITTOR_TEST_DEVICES=npu，JITTOR_TEST_REQUIRE_ACL=1，backend_fallback=error|
|缓存隔离|原生 postmerge3/jittor-home；compat compat/jittor-home；CPU结构单独缓存|

## 相对远端累计保留的改动

这些内容包括历史本地修复与本轮适配，不是本轮全部新写。`git diff MERGE_HEAD` 中 A 只表示远端基线没有该文件，并不能证明远端曾删除本地测试。

|功能组|关键文件|当前本地增量|
|---|---|---|
|NPU dtype能力声明|tests/opinfo/core.py、definitions/core_ops.py、definitions/nn_conv_pool.py、report.py|NPU声明独立于CUDA；未审计dtype候选标为UNVERIFIED，继续实际验证；Abs明确支持与拒绝测例|
|自适应平均池化|python/jittor/nn/functional/pooling/average.py；backends/acl/kernels/ops/pool_op.py、native/pool_op_acl.cc、ops/_attributes.py、include/aclops/pool_op_acl.h、acl_code_attributes.h；kernels/install.py、neural.py|非整除窗口、上采样、None轴、rank3/4前向及反向；保留f16/bf16/f32；输入几何校验；本轮迁移acl_emit/acl_program|
|归约shape与参考|python/jittor/ops/reductions.py；backends/acl/kernels/native/reduce_op_acl.cc；tests/opinfo/definitions/reductions_extra.py|全归约keepdim、标量参考、混合保维CANN descriptor|
|共享分配器|src/mem/allocator/shared_allocator.h、src/mem/allocator.cc、src/core/var.cc|底层不能共享时包装引用计数，视图最后释放真实存储跨度与原指针，避免别名提前回收|
|切片赋值及compat存储|backends/acl/kernels/ops/setitem_op.py、native/getitem_op_acl.cc；src/ops/composite/setitem_op.cc；compat/torch/installers/tensor/method_api.py、methods.py|partial writer保留base依赖，标量/广播赋值及梯度还原，contiguous存储语义|
|设备作用域|python/jittor/_core/flags.py；src/runtime/device.h、device_state.h、device_mode_scope.cc|alias写前统一快照，异常时恢复设备且不再次提交失败图|
|ACL随机与优化器|backends/acl/src/acl_op_exec.cc；native/adamw_op_acl.cc；python/jittor/optim/algorithms/adam.py|随机native入口和dtype检查；AdamW步数使用CANN所需一元素descriptor|
|注意力与矩阵运算|backends/acl/kernels/native/flashattention_op_acl.cc、include/aclops/flashattention_op_acl.h；kernels/ops/flashattention_op.py；python/jittor/nn/functional/matrix.py|KV cache专用copy runner适配当前dispatch；批量矩阵转置复用共享matmul|
|测试基础设施|tests/_helpers/common.py、tests/structure/各契约、noxfile.py、MANIFEST.in|BF16比较保留原dtype信息；真实dtype fixture、构造签名、错误类别计数、路由归属及资源清单对齐|
|文档与性能工具|docs/development/npu-validation-templates.md；agent/skills/jittor-transformers-perf/scripts/benchmark_npu_frontends.py；旧结果报告|保留缺陷/性能记录模板和可复用脚本；旧性能数值不作为本轮测量|

## 本轮失败、修复和验证

|问题|归因及修复|修复前与修复后证据|
|---|---|---|
|旧runner构造与pool_cmd|本地保留代码与远端接口迁移不兼容；移除Dispatch::Direct并使用acl_emit(acl_program)|早期编译日志与native-routing-6的NameError；native-pool-7为22通过，最终原生覆盖通过|
|KV cache memcpy的分发|专用copy runner不走需要aclnn workspace的通用路径；仍保留通用未注册算子拒绝|native-routing-6三个paged attention用例通过，最终原生再次通过|
|matmul_transpose批量B|将B最后两轴交换后复用共享matmul，不错误按二维B处理|新增7组shape，CPU/NPU前向与双输入有限差分梯度；native-reduce-13和最终原生通过|
|混合keepdims输出descriptor|从axes及keepdims生成CANN规范形状，保留flat storage，避免Jittor混合shape误导CANN|native-roots-11为17通过1失败；native-reduce-13为29通过，最终原生通过|
|整数sigmoid/silu测试契约过期|真机探针证明int32正确提升float32；增加dtype/数值/device正例，保留relu/leaky_relu/softmax拒绝断言|探针及最终原生选择集通过；不把实际受支持操作改成强行拒绝|
|FlashAttention op.run()恢复|远端原本已有，本地集成曾误删；这是本地集成回归，不是上游bug|对比MERGE_HEAD与native-full-9.patch；恢复后函数式SDPA通过|

`native-reduce-12` 的 21 通过、10 失败源于新测试调用错误（把位置dims当掩码、调用不存在的reduce_mean），改为关键字掩码和4个公开reducer后修正。它不计入产品bug。

## 新增测例和覆盖

相对远端累计新增56个测试函数/方法，分布于26个文件，数量在参数化展开前统计。其中52个来自原本地版本保留，4个在本次合并轮新引入（相对原HEAD与远端均不存在）：

- `tests/backends/acl/test_acl_reduce_keepdims.py::test_mixed_keepdims_reduce_value_and_gradient`
- `tests/backends/acl/test_aclop.py::TestACL::test_integer_sigmoid_silu_promote_on_acl`
- `tests/structure/backends/acl/test_acl_production_attributes.py::test_flash_attention_backward_attributes_precede_single_launch`
- `tests/structure/backends/acl/test_acl_runner_failure_contract.py::test_registry_owners_and_custom_copy_cleanup_compile_and_execute`

已有函数的断言更新不计为新函数；例如批量matmul_transpose测例在旧本地版本已存在，本轮更新其覆盖与验证，不能重复报作本轮新建。完整累计清单如下。

### compat/tests/torch/test_adaptive_avg_pool2d.py

- `test_adaptive_avg_pool2d_torch_forward_backward`

### compat/tests/torch/test_contiguous_storage.py

- `test_contiguous_storage_and_gradient_cpu`
- `test_contiguous_storage_and_gradient_npu`

### compat/tests/torch/test_extreme_reduction_scalar.py

- `test_full_extreme_scalar_shape_and_gradient`

### compat/tests/torch/test_slice_assignment.py

- `test_mask_scalar_assignment_forward_backward_cpu`
- `test_mask_scalar_assignment_forward_backward_npu`
- `test_slice_assignment_forward_backward_cpu`
- `test_slice_assignment_forward_backward_npu`

### tests/backends/acl/test_acl.py

- `TestACL::test_matmul_transpose_broadcast_forward_and_both_gradients_cpu_and_acl`
- `TestACL::test_setitem_mask_scalar_forward_and_gradients_cpu_and_acl`
- `TestACL::test_setitem_slice_scalar_broadcast_and_gradients_cpu_and_acl`

### tests/backends/acl/test_acl_adaptive_pool.py

- `TestACLAdaptiveAveragePool::test_invalid_shape_and_dtype_are_rejected`
- `TestACLAdaptiveAveragePool::test_windows_and_weighted_backward`

### tests/backends/acl/test_acl_pooling.py

- `test_pooling_forward_backward_with_padding`

### tests/backends/acl/test_acl_random.py

- `TestACLNativeRandom::test_arg_reduce_provider_outputs_and_gradient_without_fallback`
- `TestACLNativeRandom::test_random_distributions_and_dtypes_without_fallback`
- `TestACLNativeRandom::test_random_float64_declines_acl_with_explicit_fallback_error`
- `TestACLNativeRandom::test_seed_replays_sequence_and_advances_offset_without_fallback`

### tests/backends/acl/test_acl_reduce_keepdims.py

- `test_full_reduce_shape_value_and_gradient`
- `test_mixed_keepdims_reduce_value_and_gradient`

### tests/backends/acl/test_acl_scope.py

- `test_failed_graph_restores_scope_without_resubmission`

### tests/backends/acl/test_acl_torch_compat.py

- `TestACLTorchCompat::test_independent_frontend_tensor_executes_on_acl`

### tests/backends/acl/test_aclop.py

- `TestACL::test_activation_float_providers_reject_integer_input`
- `TestACL::test_integer_sigmoid_silu_promote_on_acl`

### tests/mem/test_shared_allocator.py

- `test_materialized_views_keep_aliases_and_lifetime`
- `test_reshape_alias_lifetime_on_selected_backend`
- `test_shared_allocator_releases_original_tuple_after_last_owner`

### tests/nn/test_relu.py

- `TestNativeRelu::test_leaky_relu_native_scale_alias`

### tests/ops/test_adaptive_pool_output_size.py

- `test_native_shape_and_numpy_integer_output_sizes`

### tests/ops/test_comparison_tolerances.py

- `test_bfloat16_uses_tensor_precision_before_host_conversion`
- `test_float_precision_is_not_relaxed_by_bfloat16_support`

### tests/ops/test_opinfo_npu_dtypes.py

- `test_abs_declared_npu_dtype_executes`
- `test_abs_float64_rejects_npu_without_silent_conversion`

### tests/structure/backends/acl/test_acl_bmm_shape_contract.py

- `test_broadcast_forward_and_gradients`

### tests/structure/backends/acl/test_acl_build_config.py

- `test_provider_sources_preserve_existing_units_and_override_kernel_defaults`

### tests/structure/backends/acl/test_acl_kv_cache_contract.py

- `test_memcpy_wrapper_preserves_cache_dependency_and_storage_target`
- `test_native_memcpy_preserves_unwritten_cache_for_shared_and_separate_buffers`

### tests/structure/backends/acl/test_acl_launcher_contract.py

- `test_batch_norm_descriptor_setup_compiles_and_preserves_runner_diagnostics`

### tests/structure/backends/acl/test_acl_production_attributes.py

- `test_flash_attention_backward_attributes_precede_single_launch`

### tests/structure/backends/acl/test_acl_runner_failure_contract.py

- `test_all_runner_constructors_have_a_valid_registry_or_direct_owner`
- `test_registry_owners_and_custom_copy_cleanup_compile_and_execute`

### tests/structure/runtime/test_device_mode_scope.py

- `test_native_device_mode_snapshot_restores_only_saved_state`

### tests/structure/runtime/test_flag_scope_contract.py

- `test_alias_only_scope_flushes_both_device_boundaries`
- `test_alias_scope_body_exception_restores_without_another_flush`
- `test_alias_scope_exit_flush_failure_still_restores_all_originals`
- `test_alias_scope_nesting_and_decorated_calls_restore_outer_state`
- `test_alias_scope_restores_device_with_both_keyword_orders`
- `test_alias_scope_setting_failure_rolls_back_and_preserves_outer_entry`
- `test_alias_scope_snapshot_failure_does_not_mutate_or_flush`
- `test_direct_device_setter_still_rejects_switch_on_submission_failure`
- `test_scope_body_error_does_not_resubmit_through_native_style_setter`
- `test_scope_flush_failure_restores_without_retrying_native_style_setter`

### tests/structure/test_opinfo_dtype_declaration.py

- `TestIndependentNPUDtypes::test_abs_has_independent_npu_declaration_and_retains_cpu_float64`
- `TestIndependentNPUDtypes::test_npu_declaration_does_not_change_cpu_or_cuda`
- `TestIndependentNPUDtypes::test_report_distinguishes_audited_from_candidate_dtypes`
- `TestIndependentNPUDtypes::test_unaudited_npu_candidates_never_inherit_cuda_override`


## 可复现命令

命令在当前checkout执行。`CANN_ENV`指向CANN的set_env.sh，`VALIDATION_PYTHON`指向所选原生或独立compat Python；`ASCEND_RT_VISIBLE_DEVICES`由执行者按已分配设备设置。`JITTOR_LAB_ROOT`指向实验工作区根目录。

```bash
: "${CANN_ENV:?设置CANN环境脚本}"
: "${VALIDATION_PYTHON:?设置验证Python}"
: "${JITTOR_LAB_ROOT:?设置实验工作区}"
: "${ASCEND_RT_VISIBLE_DEVICES:?设置已分配设备}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
source "$CANN_ENV"
export PYTHONPATH="$PWD/python"
export JITTOR_TEST_DEVICES=npu JITTOR_TEST_REQUIRE_ACL=1
export DISABLE_MULTIPROCESSING=1 backend_fallback=error
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/ascend-refactor/reproduce-native/jittor-home"
export TMPDIR="$JITTOR_LAB_ROOT/_state/ascend-refactor/reproduce-native/tmp"
mkdir -p "$JITTOR_HOME" "$TMPDIR"
```

原生完整选择集：

```bash
JITTOR_TORCH_SHIM=0 "$VALIDATION_PYTHON" -u -m pytest -vv \
  -ra \
  --timeout=600 \
  tests/backends/acl/test_acl.py \
  tests/backends/acl/test_aclop.py \
  tests/backends/acl/test_acl_indexing.py \
  tests/backends/acl/test_acl_random.py \
  tests/backends/acl/test_acl_scope.py \
  tests/backends/acl/test_acl_pooling.py \
  tests/backends/acl/test_acl_adaptive_pool.py \
  tests/backends/acl/test_acl_reduce_keepdims.py \
  tests/ops/test_comparison_tolerances.py \
  tests/ops/test_opinfo_npu_dtypes.py \
  tests/ops/test_adaptive_pool_output_size.py \
  tests/mem/test_shared_allocator.py::test_reshape_alias_lifetime_on_selected_backend
```

切换到安装了独立compat的Python后，使用独立缓存执行同一compat选择集：

```bash
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/ascend-refactor/reproduce-compat/jittor-home"
export TMPDIR="$JITTOR_LAB_ROOT/_state/ascend-refactor/reproduce-compat/tmp"
mkdir -p "$JITTOR_HOME" "$TMPDIR"
JITTOR_TORCH_SHIM=1 "$VALIDATION_PYTHON" -u -m pytest -vv \
  -ra \
  --timeout=600 \
  tests/backends/acl/test_acl_torch_compat.py \
  compat/tests/torch/test_contiguous_storage.py \
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

证明bug应使用隔离修前/修后快照、同一新增测例、同一依赖和独立缓存，检查jittor/torch导入位置以及实际设备、禁止静默CPU回退。修前在指定数值、shape或构造错误处失败，修后相同断言通过；无关依赖缺失不算bug复现。上述命令在当前源码执行，不会自动恢复历史代码；历史`.patch`必须先应用到匹配远端基线的隔离工作树。


### CPU结构及补验命令

以下沿用前文环境，但为CPU结构设置独立缓存并明确禁用加速器。原生补验用shim=0；完整结构及严格注册使用shim=1。各组存在重叠，不相加作为覆盖总数。

```bash
export JITTOR_HOME="$JITTOR_LAB_ROOT/_state/ascend-refactor/reproduce-structure/jittor-home"
export TMPDIR="$JITTOR_LAB_ROOT/_state/ascend-refactor/reproduce-structure/tmp"
mkdir -p "$JITTOR_HOME" "$TMPDIR"
export JT_BACKEND=cpu nvcc_path='' use_cuda=0 use_mkl=0 use_mpi=0
export use_nccl=0 use_cutt=0 use_cutlass=0 JT_BUILD_NVCC_FLAGS=''
export JT_CUDA_KERNEL_MATH=backend JITTOR_TEST_DEVICES=cpu JITTOR_TEST_REQUIRE_ACL=0
JITTOR_TORCH_SHIM=1 "$VALIDATION_PYTHON" -u -c 'import jittor as jt; assert not jt.compiler.has_acl and jt.runtime.use_cuda == 0; import pytest; raise SystemExit(pytest.main(["-q", "-ra", "--timeout=600", "tests/structure"]))'
JITTOR_TORCH_SHIM=1 JITTOR_TEST_REQUIRE_EXECUTION=1 "$VALIDATION_PYTHON" -u -m pytest -q -ra --timeout=600 tests/structure/backends/acl/test_acl_python_registration.py
JITTOR_TORCH_SHIM=0 "$VALIDATION_PYTHON" -u -m pytest -q -ra --timeout=600 tests/structure/backends/acl/test_acl_dtype_preservation.py tests/backends/acl/test_acl_tensor_routing.py
```

另外7个CPU共享/宿主案例是在原生NPU运行环境中执行，完整覆盖各分配策略；从前文原生环境重新启动后运行：

```bash
JITTOR_TORCH_SHIM=0 "$VALIDATION_PYTHON" -u -m pytest -vv -ra --timeout=600 \
  tests/mem/test_shared_allocator.py::test_materialized_views_keep_aliases_and_lifetime \
  tests/mem/test_shared_allocator.py::test_shared_allocator_releases_original_tuple_after_last_owner \
  tests/nn/test_relu.py::TestNativeRelu::test_leaky_relu_native_scale_alias
```

## 结果与边界

|验证组|结果|证据|
|---|---|---|
|原生完整选择集|231通过、2跳过，233收集/执行，593 warnings，115.12秒，exit0，源码未变|postmerge/native-full-14.json及logs/native-full-14.log|
|独立compat与定向OpInfo|65通过、0跳过，65收集/执行，196 warnings，503.27秒，exit0，源码未变，与原生同补丁|postmerge/compat-selected-16.json及logs/compat-selected-16.log|
|ACL结构定向|274通过，53.79秒|postmerge/logs/structure-acl-final3.log|
|完整结构修后轮|1358通过、4跳过，915.73秒，exit0，源码未变|postmerge/logs/structure-full-final.json及.log|
|native补验|15通过、0跳过，0.39秒|postmerge/logs/native-fixtures-final3.json|
|严格注册执行门禁|30通过、0跳过，2.63秒，REQUIRE_EXECUTION=1|postmerge/logs/registration-strict-final.json及.log|
|新增CPU测例补验|7通过、0跳过，7.72秒，exit0，源码未变|postmerge/native-added-17.json|
|结构失败组定向补验|74通过，2.43秒；与完整结构重叠，不累加|postmerge/logs/structure-failures-focused.log|
|打包补验|8通过，1.30秒；与完整结构重叠，不累加|postmerge/logs/structure-packaging-focused.log|

原生两项skip为`test_aclop.py`旧`jt.nn.FlashAttention`类入口不存在的前向/梯度用例；函数式SDPA已有通过证据。compat选择集为58项独立compat加7项定向原生OpInfo，其中12项CPU共享案例，不能把65项全部称为独立compat的NPU计算。

诊断失败轮保留：native-full-10为204通过18失败2跳过、776.68秒，期间结构fixture变更因此source_unchanged=false；完整structure首轮为1346通过12失败4跳过、1049.35秒，失败涉及manifest过期、9项错误类别计数、优化器签名和dispatch查询归属。compat-selected-15在jit_utils要求重建重启时exit3，随后使用新的compat-selected-16日志重跑，未覆盖历史失败记录。

全库OpInfo未全量完成；本轮没有新基线性能测量，旧9月10日性能数据仅保留历史用途。pytest耗时包含编译、收集和CPU参考，不是算子性能。通过计数也不等于证明无holder残留或无内存泄漏。

独立compat日志仍有holder/lived-var状态诊断时应按原始日志解释；本轮通过数只表示断言结果，不外推零残留。

完整结构4项跳过为`tests/structure/build/test_env_var_manifest.py`中2项resolver归属例外和`tests/structure/nn/test_nn_functional_split.py`中2项CUDA不可用案例。它们是既有边界，未通过新增skip隐藏本轮失败。

运行时代码一致性：`postmerge/runtime-delta-check.json`确认33个运行时文件的差异块与原生native-full-14、compat-selected-16测试快照相同；之后变化仅测试、文档和manifest，由补验及完整结构验证覆盖。

完整结构与严格注册测试使用源码复合摘要`e4c11e461c031ae94bc0c0e97def144096177947a6718fb29e94679282773ef8`，包含当时未跟踪的新报告文件，两轮均确认源码未变；本文最终结果回填晚于测试，不追溯为当时报告已包含最终数字。

严格注册门禁在`JITTOR_TEST_REQUIRE_EXECUTION=1`下30项通过、0跳过。`postmerge/registration-coverage-audit.json`对比远端确认测试函数名和参数化装饰器一致，没有本地删减用例。历史记录的40项不属于当前分支计数；KI-TEST004据当前30项实际执行证据关闭。

compat-selected-16首次重编译267个核心单元耗时1118.73秒，另于pytest的503.27秒执行时间记录；这些耗时不是稳态性能结论。

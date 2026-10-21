# Torch 默认转置使用真实存储 view

- Status: verified only for the stated CPU/CUDA contracts; R33 and KI-COMPAT-004 remain open.
- Date: 2026-10-16
- Owner: native tensor/autograd and Torch frontend maintainers.
- Measured clean source: `5ba7f3aff965ea03f91b8040461071d3f2d94fbe`.
- Starting source: `b7c4128b4377218c3123c057002fea47698ba381`.
- Native observer test-only follow-up: `c33967bae79585719bebba459bffa96fe66bbc57`.
- Shared layout prerequisite: `12a9a143d323b41115729d1e8aa1e79028627161`.
- Review when: storage-view provenance, wrapper GC, reshape/copy, recorded history or layout propagation changes.

## 行为与归属

`Tensor.transpose/permute/t/T/mT` 及对应 Torch 函数的默认入口，现在请求既有原生物理
permutation view。值、stride、真实共享指针和写回共同验证；identity、标量和 unit-axis
排列也保留实际 producer，能够作为独立中间 AD target。没有翻转 Jittor 全局
`transpose_storage_view`，原生默认仍为零，也没有关闭布局优化。

Torch 前端通过一个显式类型属性请求保留实际 Python 根对象。原生 view record 使用既有
opaque Python owner/GC traverse/clear 接缝保存这个引用，不导入 Torch、不建 Python ID 表。
转换后注册的是最终 wrapper，而非临时 holder。分配所有权与 provenance 分离：`.data`
换存储后 `_base` 仍指向真实原始 root；普通原生类型保持既有弱引用策略。测试覆盖嵌套根、
构造器和 factory、临时对象、三轮循环释放、重入析构、无效类型属性、线程及异常恢复。

无 grad 写入后，如果实际 allocation 和源边策略都未改变，物理 view 的 history 查询保留
原 producer 和已物化 buffer。源冻结/解冻则继续走既有 recorded-edge 规则，不能把已断的
边接回。快速 view 入口使用真实 `grad_input`；一次不合法的 `view` 也会在抛错前观察该
history，这个观察顺序由独立 Torch 四行对照证明。

可共享存储的 reshape 优先走原生 view。`Tensor.view` 对不兼容 stride 报错；
`Tensor.reshape` 仅在这个明确条件下允许真实 contiguous copy，再接原生 reshape producer，
不把私有复制临时量作为公开 `_base`。一阶、二阶与冻结源梯度沿真实 producer 求得。
`clone` 的 preserve/contiguous format 也由原生 copy owner 按实际非重叠 dense stride 执行，
展开/有洞布局不伪装成 dense 物理 view。

## 来源与执行协议

本批十三个生产文件和三个测试文件在 `5e3492b68` 冻结；合并远端 `12a9a143d` 后得到
测量源 `5ba7f3aff`。`formal-source-final-v1.json` 核对十六个本批文件与 `5e3492b68`
逐字节一致，同时四个共享 layout 文件与 `12a9a143d` 一致。旧 `0d4e4598e`、
`2f7b3a67b` 和 `5e3492b68` 各轮日志保留各自身份，没有重标。

独立 oracle 为真实 Torch `2.11.0+cu128`，完整官方 wheel SHA256 为
`c9a7ca4c74fae10a58e6175b4b2cea953f9322bb6562bbf339ad6a05f52190ad`，
实际 `_C` SHA256 为 `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`。
oracle 使用完整私有 provider，未修补旧共享安装。相同维护测试文件逐字节复制到仓外后执行，
以避免 monorepo shim 收集策略，不修改其断言。

候选 CPU 显式禁 CUDA；真实 CUDA 使用独立设备、cu122 provider 和 strict kernel math，
验证实际 location/指针及梯度。各次首次 JIT 串行，各阶段与 canonical core 使用独立缓存，
单测不共享 benchmark 缓存，未复制 JIT。新维护测试、原生 owner 测试及原始诊断使用
`forbid_backend_fallbacks` 并同步；八组诊断另记录 fallback counter 为零。
验证耗时不作为性能结论。

## 验证

| 范围 | CPU | CUDA 可见环境 |
| --- | --- | --- |
| 相同新维护测试，独立 Torch | 112 passed / 112 CUDA skips | 224 passed（112 CPU + 112 CUDA） |
| 候选：原十五文件完整邻接 + 新维护文件 | 766 passed / 615 CUDA skips，390.75 s | 1343 passed / 38 源码未配置跳过，1813.50 s |
| 原生 wrapper/GC 七项 + 共享 layout 四十八项 | 55 passed / 0 skipped，2.94 s | 55 passed / 0 skipped，5.03 s |
| public scalar-promotion 独立十六行 | 16 行完全匹配 / fallback 0 | 16 行完全匹配 / fallback 0 |

原生 owner/layout 表中两侧结果对应干净 `c33967bae`；其余数值结果仍对应 `5ba7f3aff`。

Canonical core：native **98 passed / 41 skipped / 1 existing xfail**，Torch
**224 passed / 1 skipped**，合计 **322 passed / 42 skipped / 1 existing xfail**，两模式均退出零。
完整结构分别执行：`tests/structure` **1516 passed / 4 skipped**（228.59 s），
`compat/tests/structure` **114 passed / 0 skipped**（16.27 s）；合计 **1630 passed / 4 skipped**，
不是单次 combined 命令。测试观察顺序修改后另跑提交前完整 `tests/structure`：
**1516 passed / 4 skipped**（187.67 s）；该工作区的逐文件哈希与随后 `c33967bae` 提交一致，
不把当时的 HEAD 重标为新 SHA。目录布局检查通过；相对共享前置没有新增 Ruff 诊断。
新增本报告后的文档定向检查为 **18 passed / 1 failed**，唯一失败是新页面尚未写入中央
results 索引；生产门禁与该文档登记分开，索引由中央整合后复验。

独立初始八十三行对照覆盖默认 view、写回、leaf/grad_fn、no-grad、源冻结/解冻与异常。
修前八十二行有字段差异；该计数包含布局、元数据和错误契约，不表示八十二个数值错误。
修后 CPU、CUDA 的八十三行分别全部匹配，均为 fallback 0。
另外 history15、fast-view6、strict-copy2、invalid-history4 保持逐字段零差异。
base9 中六个无循环场景一致；另外三个真实 root/type 循环候选均能释放，oracle 仍保留，
完整原始差异保留，要求候选不新增泄漏，不声称循环行为完全相等。旧 orphan-root 四行
另由真实 Torch `2.11.0+cpu` 的独立原始脚本/JSON确认；此项与主 oracle 的 cu128 身份分别保留。

## 保留的失败与边界

初始干净 `0d4e4598e` 的完整十五文件邻接得到 **6 failed / 648 passed / 503 skipped**。
其中四项涉及 expanded sibling、buffer/producer 与旧 orphan-root 断言；后者经真正 Torch
证实应保持 root，直到最后的 view 释放，原四项 leaf 写入拒绝和值不变断言仍保留。
另外两项 FP16/BF16 是共享 layout scalar promotion 错误，由独立 `12a9a143d` 修复。
最终 CPU、CUDA 完整邻接均没有失败；CUDA 的 38 项跳过全部为
`native flash-attn source not configured`，不授该扩展通过。

最后原生 owner/layout 第一轮 CUDA 为 **54 passed / 1 failed**（143.66 s）。失败的
reentrant-finalizer 测试先 `.numpy()`，再检查同一张量必须仍在 GPU。纯原生两分支诊断
`native-finalizer-observation-cuda-v4.json` 证明：不做首次 host read 时，清理 root 前后
均为 device，真实 device pointer 相同；先做 `.numpy()` 则立即变为 CPU 暂存，清理 root
与 sync 不应伪装成 GPU 执行。两分支值和析构顺序正确，fallback 0。`c33967bae` 只把
首次 host read 改为 sync 加真实 location 断言，保留最终数值与设备断言，十三个生产文件
和全部 compat 测试均不变。修正后干净提交的两侧 owner/layout 各 **55 passed / 0 skipped**。
早期探针的属性误调用、析构清理诊断与原始一项失败均保留，没有修改生产驻留行为。

最初 strict-copy 新测试的四项失败来自读取未发布的 `_version`；没有补虚构属性，而改用
真实 no-grad view 的后续写入/inspection 行为检查独立版本边界。初版 raw 审计也保留一次
拒绝日志，原因是候选增加了两项原生 producer/location 观察字段，最终审计显式验证其内容，
没有删除原始差异。

返回 `autograd.grad` 张量的 stride 仍有差异：reshape 三行中候选 `(1,3)`、oracle `(4,1)`；
copy 三十五行中十六行的返回梯度 stride 也不同。对应数值、shape 和前向实际布局相同，
`.backward` 叶梯度的实际布局单独验证一致。完整差异逐字段保留，未把这些返回布局差异
授予完整梯度 ABI 等价。candidate-only 的两行 native 观察另验证 producer ID 和真实
physical location 在查询前后完全不变。

一般 saved-input version 保护仍存在独立真实缺口：旧平方 consumer 之后执行 no-grad
公共 `copy_`，真实 Torch 拒绝版本不匹配，而既有候选仍可能接受。这是下一批独立原生
修复任务，不以本批 view 通过代替；必须保留 `.data` 的真实允许行为、常量乘法和保存
output 的反向语义。也未授予未执行的加速后端、全部签名错误、完整 C-extension ABI、
全部下游覆盖或 R33 维护者 PR 处置完成。可微 `copy_` 的实际物理存储保持也属于下一批
原生合同，不能由本批普通 view 写回测试推定所有 in-place API 已对齐。

## 复现与证据

```bash
# 每条设备轨使用独立仓外 JITTOR_HOME；首次编译串行。
CUDA_VISIBLE_DEVICES='' JT_USE_CUDA=0 JITTOR_TEST_DEVICE=cpu \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  compat/tests/torch/test_default_transpose_views.py
CUDA_VISIBLE_DEVICES=<gpu> JT_USE_CUDA=1 JITTOR_TEST_DEVICE=cuda \
  JITTOR_TEST_ACCELERATOR_MIN_EXECUTED=1 JT_CUDA_KERNEL_MATH=strict \
  JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  compat/tests/torch/test_default_transpose_views.py
JITTOR_TORCH_SHIM=0 <jittor-python> -m pytest -q \
  tests/core/test_view_owner_protocol.py tests/ops/test_storage_layout_dtype.py
<jittor-python> tools/run_test_suite.py --tier core --serial-compile
bash tools/check_repo_layout.sh
JITTOR_TORCH_SHIM=1 PYTHONPATH=python <jittor-python> -m pytest -q \
  tests/structure compat/tests/structure
```

未版本化证据：`$JITTOR_LAB_ROOT/_state/native-default-view/20261016/`。
精确完整十五文件选择在 `final-adjacency-targets-v1.json`，每轮环境白名单、命令、实际 SHA
和退出码在 `logs/*.command.json`；最终结果为 `logs/final-*.log`。
`formal-source-final-v1.{json,tar}` 保留冻结源，`final-live-process-platform-v1.json` 保留
实际进程环境及设备身份。原始 probe/comparison/audit 的版本文件与失败历史均保存。
`formal-native-observer-source-v1.{json,tar}` 绑定测试修正和原生产字节，
`final-owned16-delivery-v2.json` 给出十六个准确文件及按依赖顺序的可整合 refs。
最终 `final-default-view-evidence-seal-v2.json` 封存 **485 文件 / 287,644,875 bytes**，
SHA256 `947944ef0bd9c4103d49b666a4547485fdfff48ec69aa6b33f9cb5429c5a1267`，
独立重新逐项 hash 校验零差异。清单包括原始失败、各阶段源归档和实际加载的二进制身份，
没有复制 JIT 来启动验证。

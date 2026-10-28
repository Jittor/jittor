# ACL 后端契约

- 状态：已接受。仅主机契约已实现并有门禁；昇腾 910B3/CANN 上的设备验收是另一项仍需完成的
  门禁。
- Owner：ACL 后端维护者
- 复查触发：某个 runner 不再使用 `BaseOpRunner::launch`、属性 schema 变化、描述符缓存被挂到
  某个 runner 上，或 `BackendOps` 改变 ABI 时

本页是 ACL（Ascend/CANN）后端的开发者契约：runner 如何发射、算子属性如何从 Python 传到
C++、描述符缓存外壳保证什么，以及什么才算设备证据。构建布局、注册与
`jt.code(..., backend="acl")` 的源码规则见[源码架构](source-architecture.md)（"ACL kernel 注册"与
"后端回退策略"）；安装与日常使用见[昇腾 910B 指南](../guides/ascend-910b.md)。

## 仅主机证据能证明什么

没有硬件也可以推进代码组织。仅主机证据（stub SDK 语法检查、纯 CPU 解码器探测、静态源码契约）
可以确立一条代码边界，但**不得**被报告为 NPU 验证。真实的昇腾 910B3/CANN 验收仍是独立的设备
门禁，且每次此类运行都必须证明没有 CPU 回退。

## 标准归属者的共享发射器迁移已关闭

标准的 ACL workspace/查询/执行尾部集中在 `BaseOpRunner::launch`
（`backends/acl/kernels/native/base_op_acl.cc`）。最后几个自己直接调用 aclnn 执行的标准归属者
——SWhere、Sigmoid 反向、BatchNorm 前向与 BatchNorm 反向——都已改为经过它，各自保留其同步执行
策略；BatchNorm 反向仍然在发射完成同步之后才释放它的输出掩码。因此一次失败的 `aclnn` 执行会
带上算子名和解码后的 ACL 状态抛出，而不是打印一行日志、让输出保持未定义、图却继续跑下去。

收口时的清点是"全部 71 个 `executeOp` 归属者都不再有尾部"；此后新增的归属者直接按 `launch`
编写，`tests/_helpers/acl_launch_tails.py` 的普查仍然找不到手写的尾部（现在统计到 80 个归属
者）。Reduce prod 的三条路径——整张量、单轴、以及跨中间张量的分阶段多轴归约——都调用
`launch`，分阶段那条异步调用，使其无条件屏障仍然落在中间张量被释放之前。有两个归属者出于共享
尾部无法表达的原因保留了自己的 `syncRun()`：AdamW 循环在最后一步之后同步一次，而不是每个张量
同步一次；分阶段 product 路径在释放其中间张量之后同步。KVCacheMemcpy 从来没有尾部；它是一条
逐 token 的 `aclrtMemcpyAsync` 路径，没有 aclnn workspace 执行器，也不在下面的属性与描述符契约
之内。

同一次转换也关闭了静默失败：过去失败的 product 会把未改动的输出缓冲当作归约结果返回，失败的
`aclnnMaxDim`/`aclnnMinDim` workspace 查询则让值与索引都保持未初始化。现在两者都会抛出。这条不
变量（没有任何族自己发起执行调用、自己分配 workspace 或自己处理查询失败）由
`tests/structure/backends/acl/test_acl_launcher_contract.py` 与
`test_acl_runner_failure_contract.py` 断言，而不是靠站点计数。
`tools/build/acl_launch_program.py <old-tree> <new-tree>` 把每个归属者化简成它的（workspace 查询、
执行入口、同步策略）token 流，因此可以证明一次发射器重构没有改变任何设备可见的序列。

在 sourced CANN 并确认设备之后，在昇腾 910B3 上验证：

```bash
source "$CANN_SET_ENV"
npu-smi info
export ASCEND_RT_VISIBLE_DEVICES=<allocated-device>

set -o pipefail
backend_fallback=error sync_run=1 python -m pytest -q -s \
  tests/backends/acl/test_acl.py tests/backends/acl/test_aclop.py \
  2>&1 | tee "$TMPDIR/acl-launcher.log"
```

只有归属者确实在 NPU 上执行、并有独立的结果/梯度检查时，这次运行才作数。每个归属者都必须在
`forbid_backend_fallbacks()` 内同步，并记录 `backend_fallback_count()` 增量为零。回退尝试**不是**
NPU 验证，即使该请求被拒绝并被测试捕获也一样。保留 `execute launcher failed` 与
`aclrtSynchronizeStream failed` 诊断用于失败归因；日志匹配不是回退验收门禁。用 `sync_run=0`
重跑一次，确认异步路径在同样的检查下仍然在 ACL 上发射。没有 CANN 与昇腾设备的主机不得为发射器
改动报告硬件验证。

## 没有 CANN 时对 ACL 源码做语法检查

没有 CANN 的主机仍可以针对生成的 stub SDK 解析 ACL 翻译单元；见
`agent/skills/acl-host-syntax-check`。该检查能抓到解析错误、未知标识符，以及把 workspace 查询传
到了本该是发射器的位置。它无法检查 `aclnnXxxGetWorkspaceSize` 调用的实参，因为不装 SDK 就无从
得知那些签名。它不是硬件验证。

## 结构边界

- `AclOpFunctions` 的类型擦除已实现：一个统一的查询回调和一个带检查的执行指针取代了按签名区分
  的槽位（`backends/acl/include/acl_op_registry.h`）。分组的一元/cast/二元/add 查询族保留其参数
  适配；其它 runner 保留其带类型的直接查询。`backends/acl/src/acl_jittor.cc` 中有一份不可变的
  注册表，跨翻译单元共享，所有消费者与预检签名检查都使用它。
- 属性数据管道已为属性归属者接通：Softmax 及其反向、Triu、Flip、Cumsum、Gather 与 Scatter，以及
  卷积、归一化、池化、索引、注意力与 matmul 各族，都在 `acl_code_attributes.h` 中声明 schema，
  它们的 Python 构造器改为通过 `_code.py` 发送值，而不是把值插值进生成的源码。数据通道契约见
  下文。
- 描述符缓存仍然是仅主机的壳体。目前没有 runner 缓存 `aclTensor` 描述符；在挂上任何按形状索引的
  缓存之前，必须先接通归属与失效机制，而且在地址仍可变时不得按形状缓存描述符。

属性归属者与它的 Python 调用方、前向/反向属性构造、wire schema 和 key 契约一起迁移。只改一处
生成的赋值，留下的是一个不完整的边界。

## 原子属性迁移门禁

| 字段 | 必需的不变量 | 静态证据 |
| --- | --- | --- |
| `schema_version`/`op` | 解码之前校验版本与注册归属者 | 解码器拒绝未知版本或归属者 |
| 标量/向量值 | 保留类型标签、必需/默认规则与规范向量顺序 | schema 契约覆盖合法与畸形记录 |
| 生成的 `OpAttr` | C++ 接收解码后的值，不解析生成的源码文本 | 该归属者没有属性字符串插值 |
| JIT/缓存 key | 编译代码标识归属者/schema 与张量签名；属性标识使用排序后的类型化值与 schema 版本 | 不同属性共享源码但保持不同的数据；规范 key 排除指针/对象标识 |
| 失败路径 | 畸形数据抛 `UserError`；内部 schema 不匹配抛 `InternalInvariantError` | 负向用例在任何 ACL 调用之前断言 |

JIT 源码包含一个固定的解码器调用，而不是插值进去的属性值。CodeOp 持有每次调用的数据映射。解码
后的规范 key 不是编译后的 kernel key：改变运行时维度不会重新编译解码器。
`apply_acl_code_attributes` 在有界线程局部表中按不同的 payload 记忆化解码，因此每次执行都会变化
的 payload（例如带种子的 dropout）无法让它无界增长。描述符缓存与注册表改动仍与属性切片分成独立
的原子改动，以便各自评审和回滚。

## 数据通道 schema 与解码器

wire schema 是一张带版本、以算子为作用域映射：

- `schema_version`：整数，当前为 `1`，必需；
- `op`：与已注册 ACL 归属者匹配的不可变字符串，必需；
- 标量字段：带类型的 `int64`、`float64` 或 `bool`；缺失字段使用算子文档化的默认值，绝不隐式取
  零；
- 向量字段：类型一致的同质 `int64[]`/`float64[]`/`bool[]`；缺失向量使用显式的空/默认值；
- `cache_key`：排序后的 `(field_name, type_tag, value)` 元组加上 schema 版本；禁止指针地址与
  Python 对象 id。

非有限浮点值被拒绝。C++ key 序列化器始终使用 `std::locale::classic()`，因此使用逗号小数分隔符的
主机与使用句点的主机把同一个 float 序列化成相同结果，图在一台主机上准备、在另一台主机上执行时
key 也保持稳定。

共享的 C++ 解码器边界是 `backends/acl/include/aclops/acl_data_channel.h`。生产适配器
`acl_code_attributes.h` 是一个消费者；解码器在归属者构造 `OpAttr` 之前校验算子名、schema 版本、
类型标签与必需字段。该头文件不包含 ACL/CANN 头，可以在纯 CPU 主机上编译。它的 CodeOp wire 桥是
`acl_code_data.h`；属性类与 runner 赋值仍留在感知 CANN 的适配器里。

```cpp
AclDecodedData decode_acl_data(
    const AclDataRecord& record, const std::string& expected_op,
    const AclAttrSchema& schema, std::string& canonical_cache_key);
```

Python 那一半是 `backends/acl/kernels/ops/acl_data.py`，不依赖 CANN。`validate_acl_data()` 应用
schema 默认值、拒绝未知或类型错误的字段，并给出与地址无关的 `canonical_cache_key`。
`encode_code_data()` 通过既有的字符串到 double 的 DataMap 发送记录：有符号 int64 值使用两个精确
的 uint32 通道，因此超过 2^53 的值不会被舍入。保留前缀内的未知键被拒绝；其它 CodeOp 数据不受
影响。

`AclDataOwner` 持有不可变的算子名与 schema 副本，并暴露 `op()`、`schema()`、
`decode(record, canonical_cache_key)` 与 `consume(record, canonical_cache_key, consumer)`。构造
归属者时会校验其 schema（包括某个没有默认值的字段上的非法类型标签），并拒绝空算子名为
`InternalInvariantError`，因此错误的注册表声明永远不会被误判成调用方数据。畸形的用户数据——未知
字段、错误类型、缺少必需值、非规范的向量表示，或不受支持的 schema 版本——在任何 ACL 调用之前抛
`UserError`。

`consume()` 把 `AclDataView` 传给回调：一个借用、不可拷贝、不可移动的视图，带类型化访问器
（`int64`、`float64`、`boolean` 以及三种向量形式）、`has(name)` 和已校验的元数据。消费者请求错误
类型或不存在的字段会得到 `InternalInvariantError`。视图只在回调期间有效：发射器把值拷贝进自己的
属性对象，不得保留该视图。`AclAttrRunnerContract` 在其上再加一份固定的 `AclAttrBinding` 列表；
构造时拒绝重复、未声明或类型不兼容的绑定，其视图拒绝绑定之外的字段。

## 描述符标识与缓存外壳

仅有属性 key 不构成描述符标识。描述符 key 还包含完整形状（空形状是合法的零维情形）、dtype、
layout 与 device；字符串在固定的 key 版本下带长度前缀，负维度、空元数据、指针值与 Python 对象
id 在查找之前就被拒绝。为另一个设备、layout 或形状准备的描述符不能因为算子与属性匹配就被复用。

`AclDescriptorKey`、`canonical_descriptor_key()` 与 `make_descriptor_key()` 在
`acl_data_channel.h` 中实现这一点；`descriptor_cache_key()` 与 `DescriptorCache` 在
`acl_data.py` 中与之对应，在每个修改或查找入口之前校验完整 key。`AclDescriptorCache<T>` 是一个
生命周期外壳：调用方提供值和构造器，外壳从不创建、别名或保留裸的 `aclTensor`。两半都提供：

- 只构建一次的 `get_or_create`，以及在已有条目上租用的 `acquire(key)`；
- `is_current(handle)`/`get(handle)`，在租约过期或畸形时以内部错误失败关闭，以及
  `release(handle)`，对过期租约是 no-op（`false`），因此迟到的回调无法擦掉一个重建的等价条目；
- 归属者释放或替换设备分配之前的 `erase(key)`（逐 key 的墓碑 epoch 让更旧的租约即使在同一 key
  被重建后仍然过期）、用于分配器/上下文拆除的 `erase_device(device)`（即使该设备没有条目也推进
  其代号），以及用于全局拆除的 `clear()`；
- `device_size(device)`，一个只读的拆除诊断，统计某个设备的存活条目。

代号是生命周期记账，不出现在规范 key 中，因此在拆除之后重建的等价描述符具有相同标识，而旧句柄
仍可被检测为过期。将来的 CANN runner 可以用一个 RAII `aclTensor` 归属者替换该值；它必须在让 key
失效之前释放该值，并保留 schema 检查、key 字段、代号失效与用户/内部错误的划分。

## 迁移顺序

1. **已实现：**数据通道 schema、CodeOp wire 编码与 C++ 解码器。
2. **已实现：**属性归属者通过该通道消费运行时属性，前向与反向一起。
3. **仅主机前提已定义：**上面的描述符标识与缓存外壳。下一步：设备侧地址重绑与失效，然后把外壳
   挂到真实的 ACL runner 上。按形状的缓存绝不能复用地址已过期的描述符。
4. **已实现：**`AclOpFunctions` 的擦除查询加带检查的执行 ABI、一份不可变注册表、带类型的 runner
   实参，以及保留的同步/异步发射策略。

## 仅主机门禁

```bash
python -m pytest -q \
  tests/structure/backends/acl/test_acl_data_channel_contract.py \
  tests/structure/backends/acl/test_acl_data_schema_normalizer.py \
  tests/structure/backends/acl/test_acl_production_attributes.py \
  tests/structure/backends/acl/test_acl_launcher_contract.py
```

`test_acl_data_channel_contract.py` 在纯 CPU 主机上编译并运行一个消费者探针（默认值、向量顺序、
缓存 key 标识、错误类型失败、绑定白名单与注册失败）。通过这些并不构成昇腾设备执行过任何东西的
证据。

## 设备验收

在 sourced CANN 之后于昇腾 910B3 上运行：

```bash
source "$CANN_SET_ENV"
npu-smi info
JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 \
  python -m pytest -q -s tests/backends/acl/test_acl.py
```

只有预期的 ACL 算子确实执行、独立的值/梯度与驻留检查通过、原生
`backend_fallback_count()` 增量为零，且 `npu-smi info` 确认目标卡时，这次运行才被接受。
`tests/backends/acl` 的 autouse fixture 在 Jittor 已加载时施加 `forbid_backend_fallbacks()`；
每个测试都必须在该作用域内同步或取回数据。独立探针与该目录之外的测试需要显式导入该作用域：

```python
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks

before = jt.core.backend_fallback_count()
with forbid_backend_fallbacks():
    result = run_owner()
    jt.sync_all(True)
    actual = result.numpy()
assert jt.core.backend_fallback_count() == before
```

`run_owner()` 是被验证的操作；也要在该作用域内构造它的设备输入。该作用域不做隐式同步。
`jt.runtime.backend_fallback` 接受 `error`、`warn` 与 `allow`；只有 `error` 是验收策略。计数器
包含被拒绝的尝试，因此捕获一个不支持操作的异常并不能让测试通过。只有预检阶段的不支持决策可以
请求回退；执行异常在清理之后向上传播，且不得在 CPU 上重试。`warn` 与 `allow` 是显式的调试策略。
SDK/发射器日志仍有助于失败归因，但没有回退日志行并不是没有 CPU 回退的证据。

对属性归属者，还要跑它确切的节点，例如：

```bash
JITTOR_TEST_DEVICES=npu backend_fallback=error sync_run=1 \
  python -m pytest -q -s tests/backends/acl/test_aclop.py -k 'softmax or triu'
```

把卡型号、CANN 版本、所选节点、设备驻留与零回退尝试增量随结果一起记录（见
[NPU 验证模板](npu-validation-templates.md)）。仅主机或静态通过永远不能关闭设备验收。

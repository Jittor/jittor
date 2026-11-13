# 原生存储布局契约

[本栏目目录](../index.md) · [English](storage-layout.en.md)

修改 view、索引、分配器或后端张量描述符时查本页。需先了解 Var/Op 图；运行时生命周期见[原生运行时](native-runtime.md)，kernel 路由见[算子分派](operator-dispatch.md)。

统一的 Var/Op 图和七个元算子仍然是执行与自动微分模型。**存储布局是原生元数据；前端
不维护第二份 stride 表。**

## 表示与归属

`Var::shape` 与 `num` 描述**逻辑元素**，`size` 仍是逻辑字节数。`storage_stride(axis)`
返回**元素** stride；`storage_strides` 对规范的稠密存储为空，对其它布局（包括 stride
为 0 的情况）则显式表示。`storage_offset_bytes` 定位第一个逻辑元素相对分配起点的位置，
`mem_ptr` 已经包含该偏移。

分配的标识与生命周期仍是 `(allocator, allocation)` 加共享分配环。`share_with()` 记录
一个待定别名；分配时建立该环并持有另一份分配引用。

**物理分配、迁移、换出和释放都使用 `storage_span_bytes()`**，绝不能把一个 expand 张量
的逻辑字节数拷贝到它更小的物理存储上。迁移可以压紧活跃存储跨度并更新偏移，**前提是
保持每个活跃别名的相对地址不变**。

初版原生描述符支持**非负**元素 stride。正步长的基本切片与 expand 共享物理存储；负步长
选择和高级索引仍然是计算型的 gather 运算。兼容的 `view()` reshape 只重组 stride 块而
不拷贝，不兼容的布局报错；`reshape()` 可以显式把输入实体化为稠密。`contiguous()` **只在
输入确实需要时**才产生可见的 `ContiguousOp`。

## Kernel 边界

`Op::accepts_storage_strides` 声明读取支持。七个元算子消费真实的输入 stride；融合的
逐元素 kernel 把这些 stride 作为**运行时标量参数**接收，其带 stride 的 key 变体在不把
stride 值烘进 key 的前提下保持缓存分离。索引张量本身在构造边界处规范化——因为 gather
要求稠密的索引存储。

生成的算子工厂会**在构造具体 Var 成员或边之前**，为仅支持稠密的实现插入显式的
contiguous 图输入。这次转换是**可导的、在图中可见的**，不是执行期对既有 Var 的临时改写。

会修改输入存储的 kernel 声明 `mutates_storage_inputs`；**非连续的可写输入被拒绝**，
而不是把写入重定向进一份没人注意到的拷贝。预分配的 CodeOp 输出缓冲同样要求连续存储。

`Op::is_storage_view()` 标识**仅元数据**的操作。执行器直接服务它们的别名分配，不调用
provider 的计算 kernel。**后端重载不得把一个 Expand kernel 发射进 stride 为 0 的别名
那块更小的分配里。** clone/detach、reshape/view 和基本切片的元数据都遵循这条边界；
它们仍是图节点，有正常的梯度规则。

## 语言与设备边界

`_storage_strides()`、`_storage_offset()`、`_storage_is_contiguous()` 与既有的
`_storage_address` 向前端暴露原生事实。Torch 的 stride 与连续性查询委托给它们。公开的
基本视图记录仍然支配 holder 赋值；对源的赋值会刷新活跃的 slice、reshape、expand 和
transpose 记录。**通过相互重叠的 expand 视图写入会被拒绝。**

`numpy()` 按存储 stride 做 gather，返回稠密的 NumPy 拷贝。`data` 导出字节 stride 并带
一个持有分配的 base；相互重叠的 stride-0 导出是只读的。标量 `item()` 读取一个物理元素。
稠密数据赋值散射进非重叠的正 stride 存储，**拒绝重叠的目标**。按字节重新解释要求显式
连续的源，而不是静默改变它的存储标识。

CUDA 生成的元 kernel 消费原生 stride。稠密的外部库调用使用显式的 contiguous 图输入。
ACL 描述符助手接受源 Var，把它真实的 stride 和物理存储跨度传给 CANN；原生的 contiguous
操作使用同 dtype 的 Cast 写入稠密输出。索引拷贝的规划器仍然只支持稠密，并**显式报告
这一要求**。

> ACL 描述符与 kernel 的改动需要**真实 CANN/NPU 验证**；仅有主机侧语法检查不构成对某个
> 设备或 dtype 的支持证据。

## 协作边界

| 层 | 拥有什么 |
| --- | --- |
| 核心 | Var 布局、分配跨度、别名建立、视图形状规则 |
| 代码生成 | 输入布局适配、带 stride 的访问 |
| Provider | 描述符转换、实现特有的连续性要求 |
| 前端 | 查询原生状态、记录公开的视图关系 |

**新的消费者应当在注册时声明自己的布局要求**，而不是在 kernel 内部藏一次隐式实体化，
或者在 Python 里按形状重建一份 stride 表。

定向测试在 `tests/core/test_storage_strides.py` 与 `tests/core/test_view_storage.py`，
覆盖强制 expand 的别名标识、其后的算术/归约/索引、分配预算、赋值刷新、NumPy 导出、
正步长的存储偏移、稠密库边界以及梯度。

## 显式异步存储拷贝

卸载器等需要自行管理执行顺序的代码可以使用 `jt.core._copy_into_async(dst, src)`，
再用 `jt.core._copy_wait(dst)` 等待完成。它们不创建图节点，也不隐式加入计算流；
将参数绑定到目标的 view 不会触发这次拷贝。普通张量计算仍应使用图内的拷贝接口。

这是按字节工作的非可导接口：两侧必须已实体化、连续、字节数相等且不部分重叠，
不转换形状或 dtype。CPU 到 CPU 同步完成；主机到设备、设备到主机和同设备内的
拷贝使用设备的拷贝流。跨加速器设备以及图捕获、状态捕获中调用会显式报错。
主机存储是否锁页决定驱动能否可靠地重叠传输，不能仅凭接口名假定调用一定立即返回。

目标应当是通过 `dst.sync(False, False)` 提前分配的全新 `empty` 存储；复用旧目标前，
调用者必须完成所有此前读取或写入它的工作。已提交的设备源生产者会被排序到拷贝前。
从提交到等待完成之间，调用者不能修改源，也不能读取或复用目标。
`_copy_wait` 等待与目标分配对应的完成事件，而不是同步整张卡。

运行时独立持有两侧分配，直到等待完成；删除或重绑定 Python 对象不会提前回收这些字节。
每次提交都必须配对等待。重绑定目标前应保留原存储的独立别名（例如 `dst.detach()`），
通过它等待原分配；普通 view 可能跟随其 base 的重绑定，不能用作旧分配的固定句柄。
按目标的任一原分配别名等待会释放该分配对应的保留记录。写入会更新共享修改版本，
因此不能把该接口当成绕过反向保存值检查的途径。

回归入口为 `tests/core/test_async_storage_copy.py`；
`tests/core/test_async_copy_inflight.py` 使用真实设备流验证传输中的存储生命周期和独立等待。
设备验收与性能记录见
[PR #18149](https://github.com/Jittor/jittor/pull/18149)。修改分配所有权、流依赖或
别名规则时，核心与后端维护者需要重新执行对应设备的用例。

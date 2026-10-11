# CUDA 库的 plan 缓存

- 状态：已实现
- 上次复查：2026-10-05（基线 `e3c369acb`）
- Owner：CUDA 库维护者
- 复查触发：计算流归属、后端 teardown 顺序或 cuFFT/cuTT/cuDNN 的 plan API 变化时

cuFFT、cuTT 和 cuDNN 都要先构造一个 plan（或算法选择、执行计划）再执行，构造一次
动辄几毫秒，所以 Jittor 把它们缓存起来。这一页说明这些缓存**归谁所有**、**什么时候
释放**、以及怎么从 Python 观察它们。

## 每张卡一份，按创建顺序淘汰

cuFFT 与 cuTT 的缓存**按 CUDA 设备分库**：每个设备的库拥有自己的计算流、创建顺序
队列和 plan 句柄，容量上限也按设备计。一张卡上形状不断变化，**淘汰不到另一张卡的
工作集**。缓存键是 POD 结构体，超出上限时淘汰最早创建的 plan。共同的实现在
`backends/cuda/include/device_plan_cache.h`（`DevicePlanCache`）。

- **SDK 构造一成功就归缓存所有。** cuFFT 用 `cufftCreate` 再 `cufftMakePlanMany`，
  因此配置 workspace 或绑定流失败时释放的是同一个句柄；不会在 `cufftCreate` 之后再
  调一次会新建句柄的 `cufftPlanMany`（那正是早期泄漏第二个 plan 的写法）。
- **消费与显式缓存操作都经过 `ExecutorEntryScope` 串行化。** 释放一个 plan 之前先等待
  它所属的计算流；其它拷贝/通信流不被同步。
- **`CacheDeviceScope` 选中真正分配的那张卡，并在退出时还原。** 它不改 Runtime 的放置
  状态，也不触发库的切换钩子。
- **析构报告 SDK 失败，而不是在终结器里抛出。** cuTT 的分配回调保留原分配器与设备，
  清理时不会按执行器的当前设备去猜分配器；释放回调在异常越过 cuTT 自己的 `noexcept`
  析构之前就截住，plan 所有者把这次清理记为失败，而不是算作成功销毁。

cuDNN 的句柄、三张传统算法表和 Backend API 的 plan 表都放在
`backends/cuda/libraries/cudnn/src/cudnn_wrapper.cc` 的**每设备状态**里。2-D 与 3-D
卷积的六条路径经同一个所有者取得各自的 POD 键算法表；Backend API 的 plan 存储不再是
头文件里的内联单例。每个条目拥有它的 plan 及其描述符依赖，**先于该设备的 cuDNN 句柄
销毁**；构建失败的条目会被删除，调小算法缓存上限会立即清掉超额的表。

## 卷积走 cuDNN Backend API

传统 cuDNN API 每次调用都重做启发式并在内部构建执行计划：在 cuDNN 8.9 上
`cudnnGetConvolution*WorkspaceSize` 加 `cudnnConvolution*` 每次约 110 µs CPU 时间，
不论描述符是否复用。一个 diffusers UNet2D 步约 150 次这种调用，占它 42 ms CPU 时间
里的 20 ms。

`backends/cuda/libraries/cudnn/include/cudnn_conv_plan.h` 让三个 2-D 卷积算子（前向、
对输入、对权重的反向）先走 Backend（graph）API：按 kind、形状、步长、dtype、参数与
数值档位构成键，每个键只构建一次 plan（启发式模式 B，再 A，再回退列表；需要运行时
编译的引擎跳过；fp32 操作数在不允许 tensor op 时，用 tensor core 或下转换输入的引擎
也跳过），之后每次执行约 12 µs。设了 `cudnn_benchmark` 时，像传统的 `cudnnFind`
那样在真实缓冲上计时最多六个候选。Backend 拒绝的请求会被记住，交给传统路径。

收益：UNet2D 一步从 PyTorch 的 1.48 倍降到 1.10 倍。代价是每个新键几毫秒的构建，
**形状从不重复的负载每次都要付**——所以缓存有上限，传统路径保留。

传统卷积 API 还要求主机侧缩放标量匹配张量的缩放 ABI：double 张量用 double，其余用
float。六处调用共用 `CudnnScalingType`；此前给 double 卷积传 float 指针会得到错误的值。

## 观察与清理

| 库 | 查询 | 清理 |
| --- | --- | --- |
| cuFFT | `cufft_plan_cache_size(device)`、`cufft_plan_build_count(device)`、`cufft_plan_destroy_count(device)`、`cufft_plan_destroy_failures(device)` | `cufft_clear_plan_cache(device)`；`cufft_set_plan_cache_size(n)` 设每设备上限 |
| cuTT | 同名的 `cutt_*` 四个计数 | `cutt_clear_plan_cache(device)`、`cutt_set_plan_cache_size(n)` |
| cuDNN | `cudnn_algorithm_cache_size(device)`、`cudnn_plan_cache_size(device)`、`cudnn_plan_destroy_count(device)` | `cudnn_clear_algorithm_cache(device)`、`cudnn_clear_plan_cache(device)` |

- 省略 `device`（默认 `-1`）时查询汇总所有设备、清理作用于所有设备。
- **计数在显式清理之后保留**，所以调用方能区分"复用""重建"和"销毁失败"。
- `cudnn_plan_cache_size` 只数**有效的 SDK plan**，不数被记住的"Backend 不支持"请求；
  `cudnn_plan_destroy_count` 只数成功的 SDK 销毁。
- 进程退出时销毁所有库，且可重复调用。

## 验证

- `tests/backends/cuda/test_plan_cache_lifetime.py` 在**两张卡**上执行真实的 cuFFT、
  cuDNN 与 cuTT 运算：复用、按设备独立、定向且可重复的清理、原始 CUDA 设备的还原、
  重建，以及对 NumPy 的前向/梯度参考。cuDNN Backend API 的用例要求三个存活 plan 与
  恰好三次成功销毁，因此**传统路径的回退满足不了 plan 生命周期断言**。
- `tests/backends/cuda/test_plan_cache_bounds.py` 守住 cuFFT/cuTT 缓存有界；
  `tests/backends/cuda/test_cudnn_conv_plan.py` 守住 Backend API 卷积路径。
- cuTT 的用例显式调用 `cutt_transpose`：原生 `transpose` 现在可以是存储视图，**不能当作
  cuTT plan 执行过的证据**。plan 未命中的重叠用例检查一条无关的通信流仍处于挂起状态。
- 重跑这些用例时使用匹配的核心与库头文件；缓存下来的自定义库在核心 ABI 变化之后不能
  互换。

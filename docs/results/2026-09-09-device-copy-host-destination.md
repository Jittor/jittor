# device→host 拷贝在设备侧分配目标缓冲

- Status: 缺陷已复现，未修复
- Date: 2026-09-09
- Baseline: `33e0e34c6` 加同批 pyjt codegen 修复
- Owner: 内存与 CUDA 后端维护者
- Review when: `device_copy` 的 D2H 分支改变目标缓冲归属，或缓存分配器回收策略变化

## 结论

`Var.cpu()` / `Var.to("cpu")` 触发的 `device_copy` 把**目标缓冲分配在设备上**，
因此一次 device→host 搬运的设备峰值是张量本身的两倍。后果不是峰值难看，而是
**占用超过显存一半的张量无法搬回主机**——一个用途就是释放显存的操作，因为先把
显存翻倍而失败。同一张量走 `numpy()` 成功，说明这不是 D2H 通路的硬性约束。

本报告同时排除了"显存泄漏"这一常见误判：缓存分配器行为正常。

## 环境

单卡 NVIDIA GeForce RTX 4090（24564 MiB），CUDA 12.2.140，g++ 12.3.0，
Python 3.11.15。全部测量在同一进程内用 `nvidia-smi --query-gpu=memory.used`
对目标卡取值，配合 `jt.liveness_info()`。

## 证据一：缓存分配器正常，不是泄漏

2 GiB 张量分阶段测量（同一张卡的 `memory.used`）：

| 阶段 | 已用 |
| --- | --- |
| 基线（尚未初始化 CUDA） | 119 MiB |
| CUDA context 建立 | 533 MiB |
| 分配 2 GiB 到设备 | 2581 MiB |
| `del` 该张量，不调用 `clean()` | 2581 MiB |
| 再次分配同尺寸 2 GiB | 2581 MiB |
| `jt.clean()` | 537 MiB |
| `del` 后再 `jt.clean()` | 533 MiB |

释放后 `memory.used` 不下降是缓存分配器的设计，不是泄漏：同尺寸的后续分配复用了
缓存块（第 5 行没有增长），`jt.clean()` 能把缓存交还驱动（2581 → 537）。CUDA
context 固定占用约 414 MiB，进程存续期间不归还，与框架无关。

## 证据二：`.cpu()` 在设备侧再分配一份等大缓冲

同一序列中执行 `b = b.cpu(); b.sync()`，`memory.used` 由 **2581 MiB 升到
4633 MiB**——净增约 2 GiB，即张量自身大小。该增量也未复用缓存中已空闲的同尺寸块。

放大到 14 GiB（`float32[3584,1024,1024]`）后失败可稳定复现：

```text
[OP TYPE]: device_copy
[Input]:  float32[3584,1024,1024,]
[Output]: float32[3584,1024,1024,]
[Reason]: cudaMalloc failed
```

`Output` 与 `Input` 同形且分配失败，说明目标缓冲落在设备侧。对照组：

| 操作 | 14 GiB 张量，24 GiB 卡 |
| --- | --- |
| `jt.ones((3584,1024,1024)).cuda(7)` | 通过 |
| `.cpu()` | `RuntimeError: cudaMalloc failed` |
| `.numpy()` | 通过，返回 `(3584, 1024, 1024) float32` |

`numpy()` 在同一张量上成功，确认 D2H 通路本身不需要设备侧目标缓冲。

## 边界

未验证 ACL/NPU 与 ROCm 的同名路径是否有相同归属问题；未测量修复后的性能影响；
未覆盖非连续张量，其 `contiguous` 物化本身就需要设备侧缓冲，与本条无关。

## Workaround

```python
c = jt.array(b.numpy())   # 需要主机侧 Var
n = b.numpy()             # 只需要数据
jt.clean()                # 把缓存真正交还驱动
```

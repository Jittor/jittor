# 仓库结构现代化阶段 3：领域包收敛

## 结果

阶段 3 已删除四组公开文件与私有实现包的配对形态：`nn.py + _nn/`、
`misc.py + _misc/`、`pool.py + _pool/`、`torch_compat.py + _torch_compat/`。
canonical 实现现在分别位于 `jittor.nn`、`jittor.misc`、`jittor.pool` 和
`jittor.compat.torch`。四份 `_JittorRuntimeProxy`、runtime 注入和
`preserve_facade_origins` 元数据改写均已删除。

`nn/__init__.py` 继续承担公开组合，功能实现按 `functional/`、`modules/` 和
`backends/` 分类；`Linear` 的真实路径为 `jittor.nn.modules.linear`。兼容层复用真实
`jittor.nn.functional` 与 `jittor.nn.modules`，不再创建动态替代模块。旧
`jittor.torch_compat` 通过 `sys.modules` 指向 canonical 模块同一对象，旧 pickle
GLOBAL 路径和历史 `jt.float32(value)` 构造用法均有回归覆盖。

布局门禁新增全部八个旧物理路径，并继续禁止仓库根 `jittor_fsdp2` 与已删除的
`python/jittor/torch_fsdp2_compat.py`。`_torch_fsdp2` 是唯一剩余的 facade/private
迁移脚手架，将在 shim 与 installer 收尾阶段并入 `jittor.compat.fsdp2`。

## Wheel 审计

构建只读取显式暂存索引；用户原有的 `var_holder.cc` 与 `test_setitem.py` 换行修改未
进入制品。构建目录为：

`/home/zy/projects/jittor-lab/_state/verify/repository-modernization/stage3-build.FJiCUX`

| 产物 | SHA-256 |
| --- | --- |
| direct wheel | `f927fac12a0c689299f7f40fd92065af42bc60ed11b78f369721c139f529f61d` |
| sdist wheel | `d8a7a11ec2337907d8681b24dfe9bdc9761f84aa49949c24fcbd62ff0235b87c` |
| sdist | `1ab4962cc4c6f2a467250ac81972591182761db3da89929b5001bcad456911a0` |

两份 wheel 的 ZIP 时间戳不同，因此整包哈希不同；其 1,052 个成员路径与内容哈希完全
一致。相对阶段 2 的 1,054 项基线，转换精确锁定为 42 项 canonical 路径新增、44 项
旧路径删除和 6 项内容变化。新增与内容变化按目标 SHA-256 固化，删除逐路径固化，
未消费或多余白名单同样失败。

## 验证

| 验证 | 结果 |
| --- | --- |
| Ruff / format / Mypy | 固定版本全过；Mypy 5 个 ratchet 文件无问题 |
| Python 3.7 grammar | 526 个已跟踪 Python 文件通过 |
| Stage 3 结构契约 | 最终源码 60/60；隔离 wheel 56 pass、4 个源码元数据项正常 skip |
| 工程结构测试 | 暂存快照 38/38 |
| CPU 回归 | autograd 9/9、silent-wrong 11/11；LSTM/RNN 26 pass、12 个无 CUDA skip |
| wheel gate | 1,054 -> 1,052 精确转换；direct/sdist 成员 diff 0 |
| 隔离 wheel CPU | 安装来源、canonical/legacy 身份、深层资源、loss/grad 通过 |
| 隔离 deploy | 7 项部署、`--check`、torch/flash-attn 导入通过 |
| 隔离 wheel CUDA | RTX 4090 / JTCUDA 12.2 / cuDNN 8；loss `14.0`，grad `[2,4,6]` |

系统 `/usr/local/cuda` 没有 `cudnn.h`，因此首次系统 CUDA 探测按设计失败；切换到仓库
既有的 JTCUDA 12.2 / cuDNN 8 完整工具链后，同一隔离 wheel 的 CUDA 回归通过。

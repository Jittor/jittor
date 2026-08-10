# 2026-08-11 源码架构重构第一批

## 结论

完成源码现代化的第一批落地：`torch_compat.py` 从 11,008 行降到 8,683 行，
类型/设备、梯度、nested tensor、序列化、纯函数、优化器和 scheduler 被拆到
`jittor._torch_compat` 的 9 个私有模块。公开 `jittor.torch_compat` 继续作为稳定
facade，原有对象身份、模块元数据、pickle 路径和安装顺序由结构契约锁定。

测试总入口同时拆出纯调度模块，修复过滤前导入、skip 逻辑失效以及子进程失败不
影响总退出码的问题。主仓库没有引入新依赖，构建和 JIT 状态均写在
`/home/zy/projects/jittor-lab/_state/verify/source-refactor`。

这只是渐进重构的第一批，不代表整个源码结构已经完成。后续优先处理 `nn.py`、
`misc.py`、`torch_shim/torch__init__.py` 和根启动流程。

## 规模变化

| 指标 | 重构前 | 本批结果 |
| --- | ---: | ---: |
| `torch_compat.py` | 11,008 行 | 8,683 行 |
| facade 净减少 | - | 2,325 行（21.1%） |
| 私有实现包 | 无 | 9 个模块、2,508 行 |
| 单个私有模块最大值 | - | 765 行 |
| 测试入口 | 79 行内联脚本 | 64 行入口 + 143 行可测 runner |

非测试 Python 基线为 166 个文件、70,620 行；体积最大的 12 个文件约占 51%。
首批选择 Torch 兼容层，是因为它原为最大文件且 optimizer/scheduler 等边界清楚，
同时已有较完整的 CPU/CUDA/FSDP2 回归。

## 实现结构

```text
python/jittor/
  torch_compat.py                 # 稳定 facade 与安装编排
  _torch_compat/
    runtime.py                    # composition root 注入的最小运行时代理
    types.py                      # dtype、device、resident hint
    grad.py                       # grad context、autocast、GradScaler、clip
    nested.py                     # Size、NestedTensor、Parameter/leaf registry
    serialization.py             # safetensors 适配
    functional.py                # norm/where/diff/trapz/repeat/isin
    optimizers.py                 # optimizer 状态与 step 适配
    lr_scheduler.py               # scheduler 与 SWA
```

私有模块在模块作用域不反向导入 `jittor`。需要根运行时的模块通过
`runtime.bind_runtime()` 在 facade 组合时显式绑定，避免给现有 33 模块强连通导入
分量继续加边。

根 `jittor.__init__` 原来的局部名 `_torch_compat` 会遮蔽同名私有包，本批将 facade
局部别名改为 `_torch_compat_facade`。`setup.py` 显式加入
`jittor._torch_compat`，没有贸然切换自动 package discovery。

## 兼容契约

`test_torch_compat_structure.py` 锁定：

- facade 与私有实现对象同一身份。
- 所有搬迁 callable 的 `__module__` 仍为 `jittor.torch_compat`。
- 搬迁的顶层类/函数可通过原路径 pickle 并恢复同一对象。
- scheduler、SWA 和 LBFGS 的模块元数据不暴露私有实现路径。
- `_TorchSize` 实例 pickle 往返不变。
- 私有模块不得在模块加载域绝对导入 `jittor` 或使用 `from ..`。
- 导入检查递归覆盖模块级 `try/if/with/class`，但跳过函数体。
- facade 不超过 8,700 行，单个私有实现模块不超过 800 行。
- 源码 checkout 的 `setup.py` 必须声明私有包。

优化器搬迁后发现两处原相对导入会错误指向 `jittor._torch_compat`，已改为惰性的
`from jittor import torch_fsdp2_compat`。对应 FSDP2 用例通过。

## 测试入口

`python/jittor/test/__main__.py` 现在只做环境解析、选择、调度和退出；可独立测试的
逻辑位于 `test/_runner.py`。修复内容：

- 先按文件名、区间、`test_only` 和 skip marker 选择，再导入测试模块。
- skip marker 使用整体判断，不再只 `continue` 内层循环。
- 独立子进程返回非零、超时或启动异常会让总入口返回 1。
- 子进程使用参数数组和 `shell=False`。
- 日志目录按需创建，输出解码错误使用替换策略。
- 保留历史环境变量拼写 `seperate_test`。

7 个 runner 自测覆盖无根包副作用的加载、选择顺序、空 skip、直接执行、子进程
失败/超时、unexpected success 和总退出码。

## 验证

| 验证 | 结果 |
| --- | --- |
| 搬迁定义 AST 对照 | 32 个定义机械等价；仅 FSDP 相对导入按新包层级重定向 |
| Python 语法 | facade、9 个私有模块、runner 和结构测试通过 |
| CPU 受影响回归 | 共 145 项：143 通过，2 项按环境跳过 |
| CUDA 主兼容回归 | RTX 4090 / JTCUDA 12.2 / cuDNN 8：172 passed，0 failed |
| 结构与 runner 源码态 | 20 项通过 |
| wheel 内容 | 1,004 文件；9 个私有模块齐全；0 个禁止项 |
| wheel 隔离安装 | 从安装目录冷编译导入成功；共 20 项，19 通过、1 项源码态检查跳过 |
| 仓库检查 | `git diff --check` 与 `check_repo_layout.sh` 通过 |

CPU 模块明细：`test_type_system` 32、`test_torch_compat_grad_management` 6、
`test_torch_compat_optim` 21、`test_torch_compat_fsdp2` 13、
`test_torch_compat_serialize` 23、`test_torch_compat_ops` 23、
`test_torch_bootstrap` 7、结构 13、runner 7。

wheel 检查确认存在 `_torch_compat/*.py`、`test/_runner.py` 和两项新增测试；不存在
`.pyc`、`__pycache__`、`jittor/projects` 或 `jittor_fsdp2`。

## 基线对照与已知问题

同一 Python 进程先执行一个 FSDP2 optimizer 用例，再执行
`test_frombuffer_requires_grad_registers_leaf` 时，后者会因全局 leaf registry 状态
泄漏而失败。临时 detached worktree 在未重构的 `751050b7` 上以相同顺序得到完全
相同失败，因此不是本批回归。相关模块改用独立进程验收；后续应单独修测试隔离。

CPU 验证必须显式设置空 `nvcc_path`。仅设置空 `CUDA_VISIBLE_DEVICES` 仍会发现系统
`/usr/local/cuda`，而该路径缺少 `cudnn.h`。CUDA 验证使用现有
`/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux`。

## 后续顺序

1. 将 `nn.py` 的 loss、normalization、RNN、convolution 分批迁到 `_nn/`。
2. 将 `misc.py` 按 shape/indexing、reduction/scatter 和 sequence 拆分。
3. 将 `torch_shim/torch__init__.py` 按 nn/optim/cuda/distributed/data 注册拆分。
4. 抽出 `_version.py`、stdlib-only `_bootstrap/` 和运行时 loader。
5. 收紧 wheel 数据清单，减少递归 `package_data` 带来的发布面。
6. 单独修复 FSDP2/leaf registry 的测试顺序状态泄漏。

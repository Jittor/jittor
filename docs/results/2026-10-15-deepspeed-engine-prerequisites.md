# DeepSpeed Engine 前置条件：Module 存储与 final callback

- Status: 有界的前置条件验证；**DeepSpeed R14 尚未完成**。
- Date: 2026-10-15。
- Baseline commit: `a1a3f44afdcec9d0ce3670ff77e78c09f16b80a2`，加上本报告的提交。
- Owner: native Module/autograd 与 Torch 兼容维护者。
- Review when: native 子节点遍历、Module 序列化/收编、Tensor 反向累加、
  optimizer 执行策略或 runtime callback 归属变化。

DeepSpeed 0.19.7 读取 `engine.__dict__['_modules']` 并把它的 client model 写入其中。
返回临时 dict 的 property 无法满足该契约。native Module 现在暴露一个持久的、弱拥有的
注册视图，其值就是实际的子属性。赋值、替换、删除与注册的 `None` 值都会影响活跃
module、其参数与 checkpoint 键。遍历在需要时对别名去重，并使用回溯式祖先集合，
避免在每个节点复制祖先集合。

pickle 一个 Module 只会重绑定已经存在的注册缓存。从未缓存其子节点的 native pooling
实例保留原始字段布局。已分离视图的 copy/deepcopy/pickle 产生可用的 dict 快照；
持有原始活跃视图不会使其父对象存活，父对象销毁后再访问该视图会抛出 `ReferenceError`。
native 子节点收编会丢弃源实例的缓存，并把新视图绑定到被收编的子节点。

公开的 `torch.autograd.Variable` Tensor 别名现在拥有真实的 final-callback 队列。
成功反向运行的回调在梯度累加之后按 FIFO 顺序运行；回调可以再追加回调。Hook/callback
失败会传播，不会把回调留在下一个图中。在反向之外调用会显式失败。该实现共享既有的
native 微分与累加；它不实现第二个反向引擎。fidelity 仍为近似：
不提供 PyTorch worker 图 ID 与 compiled autograd。

Adam/AdamW 初始化现在为既有更新选择器保留 `foreach`，包括参数组覆盖。稠密 Var
存储暴露 `is_sparse=False`。`GradBucket` 与 `Adagrad` 可作为 DeepSpeed module 主体的
导入依赖，但仍**未实现**：构造会报错，而不是伪造 native bucket 或 optimizer 更新。
可导入并不计为 optimizer 或分布式支持。

验证使用 Python 3.11.16、NumPy 1.26.4、pytest 7.4.4、g++ 11.5 与隔离的
RTX 4090。native CUDA 使用真实 nvcc 12.2.140/pip-cu122；独立二进制 oracle 为
PyTorch 2.11.0+cu128。CPU 进程隐藏所有 GPU 并显式禁用 CUDA。CUDA 张量执行使用
单独的编译缓存与真实设备驻留。原始日志与数组保留在 lab 的
`_state/refactor-goal/deepspeed/` 下，位于仓库之外。

```bash
python tools/run_test_suite.py --tier core
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q \
  compat/tests/torch/test_module_child_storage.py \
  compat/tests/torch/test_module_custom_state.py \
  compat/tests/torch/test_torch_module_registration_order.py \
  compat/tests/torch/test_backward_final_callbacks.py \
  compat/tests/torch/test_deepspeed_public_dependencies.py
bash tools/check_repo_layout.sh
JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure
```

官方 CPU core 门禁通过 **322** 项，跳过 **42** 项，保留 **1** 项既有 xfail。
最终定向的 CPU/CUDA owner 门禁通过 **40** 项，跳过 **0** 项，包含既有的
custom-state/`_apply` 行为。首次完整结构运行发现三项 pooling pickle 字段回归；
其 owner 已修正，随后全部 **12** 项 pooling 结构用例通过。重复的完整结构扫描在本报告
加入之前通过 **1441** 项、跳过 **6** 项、失败 **0** 项；把新报告登记到共享文档索引中
是整合 owner 的步骤。在最终的收编修改之后，独立 frontend 与 NN 兼容性的 CPU/CUDA
回归也通过 **36** 项，跳过 **0** 项。

另一项尚未发布的 DeepSpeed adapter/Engine 门禁实验使用干净的上游 `v0.19.7` checkout
`f5af15c2b4da10411e340850c00ba0cb5b14fd87`，来自
<https://github.com/deepspeedai/DeepSpeed>。它完成了真实的 Engine
前向/`backward`/`step`、六个 microstep、梯度累加到三次 AdamW 更新，以及保存/加载
到新 Engine 后的续训。两个 runtime 使用同一份源码构建的 DeepSpeed wheel。全部八个
参数梯度与两个输入梯度都被请求；256 个输出/梯度/参数/optimizer 数组与独立 oracle
匹配。按各参考量级与公共梯度下限归一化的最大偏差在 CPU 上为 `3.37e-5`，在 CUDA 上
为 `1.26e-5`。Jittor 运行在 error 策略下同步，后端 fallback 尝试为零。这是有界的
**FP32、world-size-one、禁用 ZeRO** 证据，不确立多 rank NCCL、ZeRO、bf16、融合扩展、
DeepCompile 或弹性启动支持。可选的导入 adapter 及其自身的准入 gate 不包含在该
前置条件提交中。

在该前置条件基线上，`cpu-owner-target.log` 中的 leaf-hook 负向复现显示，native 叶子
上的 `register_hook` 不会触发，而独立 Torch 会；final callback 通过中间/loss hook
测试。随后的 native owner 修正与实际 ZeRO 1/2 Engine 验收记录在
[有界 FP32 Engine 报告](2026-10-15-deepspeed-zero-engine.md)中。
原始复现仍可获取，单 rank FP32 验收也仍不完成多 rank bf16 R14。

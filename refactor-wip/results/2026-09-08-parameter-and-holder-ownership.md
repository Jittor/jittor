# 参数类型与 Tensor holder 生命周期

- 状态：本批完成；7.12整体未关闭。
- 日期：2026-09-08；owner：协调分区；基线：36c88b7ac。
- 复查：Parameter构造、Module注册、backward或安装事务变化时。

native Parameter改为真实Var子类，删除假metaclass及复制类成员的实现；共享存储、
独立叶子、Python子类动态属性、pickle/deepcopy得到保留。native Module与ParameterList
按名字/容器注册，不再给Var写Torch角色标记。原生层的普通Var权重仍是注册参数，但
不再假装其Python类型是Parameter；独立Torch层将构造的权重提升为真实Parameter。
legacy plain-tensor规则留在compat installer，显式register_parameter按名字保留原对象。

独立Tensor的leaf/retained索引改弱引用，内核is_backward_leaf决定叶子身份。
移除对独立holder的optimizer启发式修剪、断连误删、每轮retained.clear；
retain标志、设备提示和RMSNorm缓存进入每对象TensorObjectState。legacy无weakref的
Var保留强引用清理路径，不宣称这一路已拥有独立Tensor的生命周期语义。

四个缺陷实际4 failed→4 passed：非叶误登记、无关图使leaf丢登记、重复backward
retain失效、丢弃holder无法GC。扩展为7个生命周期案例后，连同独立前端2项，
真实CUDA9 passed（49.09秒，输入驻留断言防CPU fallback）。CPU对应案例、事务重放、
注册与rootcause首次23 passed/1 failed；失败是旧假类型断言，改为真实类型和按名
注册契约后该节点通过。native参数/容器/角色20项通过，最终含新增序列化的参数类型
文件3项通过。未跑完整套件，未重建本批wheel，未做NPU实机验证。

日志（未版本化）：`/home/zy/jittor-lab/_state/compat-dist-OIR3DI/`中的
`registry-before.log`、`registry-after.log`、`native-parameter*.log`、
`state-cpu.log`、`state-cuda.log`、`parameter-identity-final.log`。

剩余7.12边界：安装级状态向native发布的旧别名、legacy激活/类型修改路径，以及
视图/存储完整语义。后续应按这些边界成批处理，不再以逐API补洞替代架构收口。

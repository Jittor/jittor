# ms-swift 兼容环境合同

本合同按任务声明选择设备 profile。Ascend/NPU profile 需要真实 ACL 与 torch_npu oracle；
CPU profile 需要独立原生 PyTorch CPU oracle；CUDA 或其他设备只有在任务矩阵声明后加入。
不能因为当前分配只有 NPU 就把 CPU 公共语义从 skill 的适配面删除，也不能用 CPU fallback
冒充目标加速器证据。

## 调度与资源

所有需要调度的计算、Python 导入检查、测试、JIT、benchmark 和模型运行都读取任务状态目录的
`current-job-id`，通过 `srun --jobid=<current-job-id> --overlap` 进入当前分配；活动 launcher
必须校验文件内容为纯数字，并在 worker 内断言 `SLURM_JOB_ID` 与该文件一致且存在
`SLURM_STEP_ID`。登录节点只做 Git、文本编辑和调度编排，不用 Mac 或其他未授权主机代跑。具体节点、设备映射和进程数写入运行 manifest，不能依赖
登录节点的可见设备推断分配结果。

当前分配由 `current-job-id` 指向；资源探测只有一个实际主机时，真实多机轨道处于资源阻塞。
Ascend 多机恢复条件是获得至少两个真实主机且每主机至少两张 NPU，并按授权调度入口运行；
CPU 单机工作不得伪装成多机加速器证据。
同一主机启动多个进程、使用主机别名或伪造 `hostname` 都不是多机证据。当前必需的单机训练规模为 2 和 4 NPU；未取得的规模保持未验证。新增规模只有在任务明确要求时才进入必验矩阵。

每 rank 绑定一张声明的目标设备。单卡轨道只暴露所需设备；多卡轨道按实际拓扑暴露对应
设备，不能沿用单卡 `device_count == 1` 限制去屏蔽分布式问题。Ascend 两侧都使用真实
Ascend，分布式必须验证实际 HCCL process group；Gloo、CPU collective 或未初始化的通信
桩不能代替。CPU 参考只证明 CPU 语义，不证明任何加速器执行。

## 两个独立运行时

oracle 使用原生 PyTorch、torch_npu 与真实 ms-swift；candidate 使用当前 worktree 的
Jittor、torch shim、ACL/HCCL 与相同版本的 ms-swift。下游依赖来自同一锁定清单；记录
两侧解释器、Python ABI、包版本与模块实际路径。ABI 不同的扩展各自安装，不共享会改变
`torch` 解析结果的 site-packages。安装前检查依赖 pin，不能让安装过程替换 oracle 的
PyTorch 或把 shim 部署进 oracle。

- candidate 的 core/compat 采用维护的 source-path、兼容 editable mode 或构建 wheel，
  避免默认双 editable 安装制造空 namespace。
- 用维护的 `jittor-torch-shim --target "$CANDIDATE_SITE"` 部署入口及发行包元数据，
  再执行 `--check`；这些 Python 操作也必须放在上述 `srun` 内。
- 固定官方 Ascend/CANN 镜像 digest，或使用维护者已验证的宿主 CANN；记录驱动、CANN、
  Python ABI、编译器、设备型号和拓扑。加载 CANN 后在实际 worker 中验证身份、NPU
  可用性和设备绑定；仅 `npu-smi` 可见或 import 成功不构成执行证据。

## 输入与状态隔离

显式设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。模型与 adapter、tokenizer、
数据、初始参数及 buffers 均离线锁定并保存摘要。优化器与调度器配置、随机种子、精度、
全局 batch、梯度累积、固定 rank 数据划分及有效 token/样本数必须在两侧一致。

当前 FP32 公开训练参考显式锁定 `fp16=false`、`bf16=false`、HF32 关闭和
`full_determinism=true`；仅设置模型 `torch_dtype=float32` 不足以禁用训练混合精度。
确定性 API 必须读回实际后端状态，并用恢复轨迹检验数值效果。CANN 9 的 HCCL 初始化
接受 `HCCL_DETERMINISTIC=true/false/strict`，不能把 Transformers 某版本写入的 `1`
直接提前放进 communicator 初始化环境；记录实际初始化顺序及 worker 起止环境，不能静默
覆盖下游后写值或因此免除严格恢复比较。

环境、wheel、模型缓存、编译缓存、临时目录、日志、快照和 checkpoint 均放在
`$JITTOR_LAB_ROOT/_state/<topic>/<run>/` 或用户明确指定的任务 state 根目录；短临时路径可放作业本地 /tmp，均不进入仓库。
运行目录不能覆盖已有证据；复用缓存时保留来源和兼容条件。

为每个运行键、runtime、rank 分配独立的 `JITTOR_HOME` 或 `cache_name`，并隔离 `TMPDIR`
和 `XDG_CACHE_HOME`。首次 JIT/扩展编译串行完成后才启动并行验证；单测与 benchmark
不能共享正在编译的缓存。分布式 checkpoint 恢复必须使用新的 worker 进程，不能靠同一
Python 进程保留的对象证明恢复成功。

## 内容寻址运行键

运行键至少哈希：同步 SHA、相关 dirty diff、测试与 runner 源码、依赖锁、解释器 ABI、
驱动/CANN、设备型号、轨道/阶段、主机与 rank 拓扑、后端及 fallback 策略、模型/adapter/
数据/初始权重摘要、tokenizer、精度、优化器/调度器、全局 batch、梯度累积、固定数据划分、
随机种子、步数和推理生成配置。checkpoint 恢复追加 checkpoint 摘要、保存步、恢复步和
父运行键；性能追加预热、同步、测量与统计协议。

键相同且日志、manifest、快照和结果完整时直接复用，不因重启会话重复原样计算。
输入或相关实现改变就生成新键，仅复验受影响范围；旧结果保留原基线，不重标为新提交。
源码未变的未受影响证据可引用原键并说明关联，不拼成一次不存在的完整运行。

公开数据预处理可能创建 AF_UNIX socket。为作业本地短 TMPDIR 使用已有 `JITTOR_TORCH_KEEP_TMPDIR=1`，避免 shim 默认改写成长缓存路径；各并行角色仍各自独占临时目录。

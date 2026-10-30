# Torch Compatibility Release Acceptance: 2026-10-26 Day 2–5 Continuation

## 2026-10-30 R14 Qwen held-out generation 指标补充（部分完成）

本项使用 R14 已训练的 seed821 LoRA adapter，不重训、不做 preflight/resume，也不重跑 CUDA core 冷编。源码绑定 HEAD e4a3894065953efa50ea23a0e52b45511b835f59、fingerprint 89157afb357abbb882057230ecb0e86586deeef7fb010a7cc51b3577eea054a6；base 为 Qwen3-0.6B revision c1899de289a04d12100db370d81485cdf75e47ca。JT adapter SHA256 8669a5fb3f0b738573809ede36c77a0758eb26aa555a6cc5d29f75ee9329257b，PT adapter SHA256 d48da15c6e4574d91f8474e7e16a010a000f3c69542fb681ab43377f31ea2b36，两边 adapter config hash 相同 363e7f31496a81fcaa6ed24f15c906ae7bf3ec8dac14c5015d1673bc88cf2051。训练与生成评估分别使用相同的 train hash 3836597b696a3b79b81ac9f34bc79ce9a4d06cf6e8bf204aabf48dc5d924ecf0 和 eval hash d8318478c8991e75ba5cac474e1f3dfe95c0478da49ad8b2a8909c72fb72f84f；seed821、2 epochs、batch=1、gradient accumulation=2、BF16 训练、最大输入96。eval 17个 source ID 与训练选定的101个 ID 不交叉；来源仍是 Alpaca train split，不是独立 benchmark 测试集。R14 的 gradient-checkpointing shim 仍直接执行目标函数，未证明 activation recompute 或显存节省；这不影响本次关闭梯度后的生成协议，但训练差异须保留在解释边界内。

生成协议是 prompt=instruction 加非空 input、newline 拼接、明确排除参考答案；tokenizer 原始编码、最大输入96、FP32 权重、无 autocast、greedy、num_beams=1、max_new_tokens=48、use_cache=true。每侧单卡单进程，目标 A800 UUID GPU-ecb44f7d-870b-b9a4-a3d6-10301bae6068，对外 logical cuda:0。分数为每例宏平均 exact match、regex-token precision/recall/F1 和 token ROUGE-L F1；这是小样本描述性指标，不设发布阈值，也没有人工判读。

PyTorch attempt torch-02 自然 rc=0，17/17 预测非空，真实模型参数和输出 token 均在 cuda:0；PyTorch 2.6.0+cu124。宏平均 exact match=0、token precision=0.264501、recall=0.379311、F1=0.212716、ROUGE-L F1=0.164849，诊断耗时25.94秒。exact match 为零及较低 recall 受48-token输出上限和样本量影响；这既不是质量 PASS，也不单独说明训练退化。PyTorch 首次 attempt 因启动环境遗漏共享 runner 的 PYTHONPATH，在模型加载前 rc=1，原件保留。

Jittor 侧 attempt-01 使用复制 cache 后触发 jit_utils reload guard（rc=3）；attempt-02 在复制后的不同 JITTOR_HOME 上由 JITTOR_NO_BUILD=1 正确拒绝缺失的 core build stamp（rc=1）。两者均未加载模型。attempt-03 改用 R14 原始 JITTOR_HOME、相同 cache/build flags，JITTOR_NO_BUILD=1 下成功复用原 core，确认其 SHA256 85de0553b81d8a1b0124704e95d0e8ed198f06b70be7ded7302063e8eaa2ecbe 与 build stamp SHA256 3db74ae63e5475eda46ff7011e7e59721c8ddec7ecd2bcc603a21a093fd1d4bf 未变。进程 PID 2854331 绑定前述 GPU UUID，观测到设备内存2756 MiB，并进入生成循环；300秒上限到达时没有 result.json，最终 fallback counter 未落盘，故 JT 分数及最终 fallback 状态均为证据不足。timeout 后进程、锁已退出。原始 cache 清单由2820项增至2895项，仅新增75项，没有已存在项改写或删除；前后清单 SHA256 分别 95fa8a73644191191987b4ad1d49b9ba1cc2614f8fb34c5a20e9e012ccb3c691 与 004dcc11da5c8f9179cb0265d0a2d5a513ba8cdabb018ba9b357dae29fa5a238。这次 Jittor 启动和长模型加载不作性能结论。

执行包 /data6/cgq/jittor-lab/_state/release-day23/20261030-r14-qwen-heldout-gen-r1-prepared/：evaluator SHA256 99b427052ec229f570c031784518360043b6df72691a5331aba7113b2750a732，PyTorch result.json SHA256 9322562094de3c1a7aace74dc15e6325ca38ffa232886e43ebfc44ea8c7b7bed，终审 terminal.json SHA256 7e0ed47258f310320ec9092517efcb1a8766dee5e1aaed207e9a164d15b3d69b。结论为部分完成：补有 PT 小样本任务分数，但 Jittor 侧没有完整 generation quality 结果，因此不能宣称两侧生成质量接近。R14 原有训练/续训实用证据保持有效；整体 Day2–5 与 Day4–5 正式性能/显存验收仍未完成，release 继续 Not complete。


## 2026-10-30 R14 Qwen LoRA 实用训练与新进程续训通过

本次验收绑定隔离快照 `/data6/cgq/jittor-lab/worktrees/torch-compat-practical-r14-main-snapshot-20261030`，HEAD `e4a3894065953efa50ea23a0e52b45511b835f59`，源码指纹 `89157afb357abbb882057230ecb0e86586deeef7fb010a7cc51b3577eea054a6`；执行前后指纹一致。实际 CUDA 冷编产物为 core `85de0553b81d8a1b0124704e95d0e8ed198f06b70be7ded7302063e8eaa2ecbe`、utils `d8a93374c62de296dcc55bdc7e4f4419330ce0f468a4ddaf01178dbd99a8179f`、stamp `3db74ae63e5475eda46ff7011e7e59721c8ddec7ecd2bcc603a21a093fd1d4bf`，CUDA key `cu12.4.99_sm_80`。CPU 冷构建 227 TU，CPU focused 30 passed/6 CUDA-fixture skipped，维护 core10 为 227 passed/3 skipped；CUDA focused 为 47 passed、无 skip/fail，均未见 fallback。该候选是在隔离快照中执行，不能误称为已合并或已提交的干净源码版本。

seed821 Qwen LoRA full 两 epoch完成 102/102 optimizer updates、0 skip，202 个 microbatch 的 CUDA FP32 梯度与 loss 均有限；112/112 LoRA tensor 相对初始 adapter 改变，冻结 base 未变。full 汇总 train loss JT 2.3359577714、PT 2.3366093262；按 epoch 计算的微批次均值 JT 1.2054693076→1.1536167642，PT 1.2056800938→1.1540639769，均呈下降趋势。相同固定验证集 eval/predict loss 为 JT 2.4555516243、PT 2.4566440582，相对差约 −0.0445%；未记录 held-out generation 质量分数，不据此宣称生成质量或普遍优势。

随后由独立新进程从 `checkpoint-51` 恢复，最终 `global_step=102`、本进程完成 51 次更新、0 skip。模型/optimizer/scheduler 恢复签名精确一致，930 项状态值检查通过；337 项模型/optimizer 张量设备映射与 `cuda:0` 观测一致，112 个 optimizer scalar `step` 仍逐值检查但不再错误要求 tensor device map。Python、NumPy、CPU、CUDA RNG 检查全通过，续训 101 个 microbatch 的 sample ID 顺序精确匹配 full run 的第二 epoch；112/112 adapter tensors 相对 checkpoint-51 更新。最终 eval loss 与 full JT 完全一致，保存/加载/merge 生命周期通过；实际 UUID `GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d`，fallback=0。

首次 `resume` attempt-01 在任何训练更新前被旧比较器中止，原始 returncode=1、日志与 cache 收据保留：checkpoint state signature 相同、930 项值检查全通过，唯一失败条件是将 112 个 optimizer scalar step 的 dtype 条目误作必须存在设备映射的 tensor。修正仅限执行包中的状态审计 helper，并使用新目录 `resume-attempt-02` 手动续跑；未覆盖 attempt-01，也未自动重试。终端证据：full review `reviews/full-821-terminal.json` SHA256 `f8521274bcfdf78a3728ca3457af5b52b99239b70eab0b739b463f950fc4d382`；attempt-01 failure review `reviews/resume-01-terminal.json` SHA256 `2e740cf0a3c3a7b1c3469545003e7e013e5533c672b853ecc3629a904b0421b7`；attempt-02 resume review `reviews/resume-attempt02-terminal.json` SHA256 `eebcda6a2f97dae4de8f6da7fb3f3864e0c9c15bdb89fe0518865737d0ae349c`；step102 checkpoint manifest SHA256 `12c1e228f76af70c61306041d0675024f751b673e506b783d81fcdc6aa05b5a0`。完整 package manifest SHA256 `c5ad13321df99209948142f0edc97669dd70d0c9362b38435cfeb8b84a088e90`。

严格数值/互操作边界独立保留：stock PyTorch `torch.load` 无法读取 JT 自己生成的 plain-pickle `optimizer.pt`（原件 `reviews/checkpoint-51-pytorch-audit-failure.json`），而受限 Jittor portable-pickle 对原始 checkpoint state 的审计及同一 Jittor 新进程续训通过；这属于跨框架序列化互操作限制，不推翻同运行时功能恢复。旧 strict comparator/FAIL 不回写为 PASS。gradient-checkpointing shim 没有执行 activation recompute，显存/性能证据不足。Day4–5 的正式配对性能与显存门及更广泛 Day2–5 发布验收仍未完成，因此仅判定本配置实用训练和保存恢复通过，整体 release 仍 Not complete。

## 2026-10-30 跨 CUDA 设备 Tensor.to 功能回归验收

此前双卡环境的生产 `Tensor.to` 跨 ordinal 传输曾出现读回 NaN，而调用返回成功；独立 host staging 标量往返正确。这里是环境相关的失效证据，不足以确定所有系统上的唯一根因。修复限于 `compat/torch/types.py::_make_cuda_resident`：当输入 Var 已位于 CUDA、且源目标 ordinal 不同时，使用 `_make_cpu_resident(v).to_device(index)` 完成可微分的 D2H/H2D 两段迁移；同卡、CPU→CUDA和未放置Var仍用现有路径。

R7绑定执行 HEAD `54fd59269bc9f0dba7651947fff111e71897f031`、工作源码指纹 `29e687e3d100f3e35fcac4ef13c1465c2bcabf94a7f27d07fb79aa2eb577c47b`（2815个源码文件）；双卡冷编 CUDA core SHA256 `e65333a9c7cf11d9865bc8ae8ba415afab130014b47e3592b2c9c14edf735b3c`。受限 systemd worker 在 GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701` 与 `GPU-b6f59824-d1a3-cd6f-8100-9149728fc8eb` 上运行单一节点 `/home/cgq/jittor/compat/tests/torch/test_native_tensor_placement.py::test_explicit_copy_roundtrip_and_gradient_keep_source_placement`，pytest `1 passed in 11.45s`。cuda:1→cuda:0→CPU 结果精确为 `[2., 3.]`，反传到源张量的梯度精确为 `[4., 6.]`，placement断言通过；worker记录两个GPU均由该进程持有、设备可见数2、fallback计数前后均0。

终态审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-torch-cross-cuda-stage-r7-test-hold/terminal-audit.json` SHA256 `6ba61de0e06aa0962cf1a254ba01205f6d92cc42ac82e9df89c1af186b56597f`，`result.json` SHA256 `90c7d30cffea95d18c5f42974abf5e341c1aeea3c5381dcbf58267658e0bcbe8`，pretest audit SHA256 `b4f4f5f6a8f5e72b702addf998560709f1e183efda01e5ee5f0f9bd706b1c345`。独立核验通过：2815文件源身份不变，core/utils/stamp及libcudart映射与hash一致，3323个受保护缓存未变；candidate cache无删除或已有内容改写，仅锁文件变化并新增该用例的JIT/torch-shim缓存。R7服务自然退出，cgroup为空，Jittor锁不再持有。

执行历史按次保留于 `_state/release-day23/20261030-torch-cross-cuda-stage-r{1..7}-*`：R1运行时工具模块需新进程重启，未进pytest；R2原子结果写入错误，测试结果未知；R3双卡配置缺少对应core，pytest未启动；R4隔离双卡冷构建成功；R5测试环境误带旧cache_name，在pytest前停止；R6 pytest因相对节点路径未解析，报告`no tests ran`且测试体未运行；R7修正cache_name、仓库cwd和绝对节点路径后完成唯一有效测试。R1/R2/R3/R5/R6均不计作测试通过或测试失败；R2结果明确未知。

本结果只证明这一输入、两张卡和反向梯度路径的功能性，不构成性能、capture、通用驱动兼容或长周期训练证据。Day2–5整体继续未完成，Day4–5性能/显存门仍待正式验收，之前的strict FAIL不变。


## 2026-10-30 e7d171 Qwen LoRA checkpoint51→102 实用训练验收

本次绑定主树HEAD 9ffd0884ebc79f1e3674f091cc569de672b22219、源码指纹e7d1711ffcb67363b63a0ffb0743ec152eed418edbb0150bbfab52e754ab8c90。实际使用当前Python shim与原位DF45 CUDA core/utils，core SHA256 8774718332e0b0bddd2f3f72a26c448ac60484266a3e200207032d20d23a3020、utils SHA256 d83b331385c81f3ffa8b70ba2665acee73fe6ba7e5c300089af43170f80eebaf；属混合身份，不是e7原生冷编。实际GPU UUID GPU-2be51dc7-d99a-7307-869b-30ed5888e701，fallback=0。Qwen3-0.6B LoRA，seed821、BF16、gradient accumulation=2、每设备batch=1、eager eval；本次只覆盖既有checkpoint51之后的step52–102。

恢复时global/scheduler step及112个optimizer scalar step均为51；337个模型及optimizer状态张量实际CUDA0，Python/NumPy/CPU/CUDA四类RNG通过。功能恢复通过，strict hash仍因两个额外initial_lr元数据项FAIL，旧FAIL保持。之后完成51次更新、101个微批次、51次AdamW返回及对应更新分支；每步112项unscaled FP32梯度均finite且无缺失，optimizer skips=0。首次续训step52有112/112个LoRA参数变化；后续不声称逐参数逐步均变。BF16/autocast仅在前两个微批次观察。checkpoint shim实际被调用但直接运行目标函数，不证明activation recomputation或节省显存。

51个更新step等权loss JT 1.1684247986，PT full/resume均1.1688747537；逐步mean-abs差0.0021948391，max差0.0068603158。JT 101个微批次等权loss 1.1536167642。分段loss有起伏，step102为epoch尾部单微批次。固定17行验证集两次eager评估，最终JT eval loss 2.4555516243，PT为2.4566440582，差−0.0010924339（约−0.044%）。PT使用不同checkpoint；sample ID及shape匹配不证明token/label字节相同。结果支持此配置下的实用训练和固定集loss接近，不支持严格对齐或普遍收敛。

checkpoint102共18文件、约30.47MB。adapter、optimizer、scheduler、RNG、TrainerState及release-state签名/值/设备文件hash已核；112个保存optimizer scalar step及Trainer global_step均102。没有新进程重载step102，未运行predict，没有accuracy或性能验收。工件目录为 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-e7d171-qwen821-resume51to102-executor-r4-hold/attempt-01/qwen_resume/；practical-report.json SHA256 04e24040e445a35434da85b4edb58275c608d18998283210fb2e5fede277e022，comparison.json SHA256 9de9f20b1074e1edb9ad0c762f1d6dfa18b28aaa245f6860fdec9e98ea0ac552，controller receipt SHA256 56b76651012724d2ceb2cf8c92bc1e0508bc3eaa859eaa64400304beb978c767，worker result SHA256 605b0a01f847f23e57b3e60a4018000713742235cb2b3909debd32bc93fe7f27。两份执行包静审SHA：1f475ec4e62d30c0b877a8735c39c2ce6064f8b445fd4abdff67fa1bb2fe10be、2b22ab01aeea1e8e28121221a68075e40262960996c405b559822814132d2018；终态审计SHA：6341d8ca21ab5d342afdb4f90036c329a0cd541af27131ed730c1540b8334a27（运行/产品/缓存）、e60b5a26a411361b4229acd583f87b0e9d295db8a0dacfb96d6aa0955bac2e9c（训练/恢复/评估）。unit自然成功、worker rc0，保护cache一致、candidate cache旧内容无改删且新增0、五锁无内核持有记录。/proc有1883次权限拒绝，外部进程与FD可见性有限。

实用训练判断：本次e7d171 Qwen续训通过；strict initial_lr哈希失败继续保留。性能/显存独立未验收；predict、step102 fresh reload、通用长训均未由此证明。

## 2026-10-30 BERT checkpoint100 新进程加载与评估

在上文 step50→100 的真实续训之后，独立新进程实际通过 Trainer 加载该 checkpoint100 的模型、优化器、调度器、TrainerState 与 RNG：12 项 checkpoint 文件身份核符，模型/优化器/调度器/scaler 状态签名与已保存值严格一致，201 个 optimizer scalar step 均为100，603 项参数及moment设备为CUDA0，四组RNG恢复一致。此为新进程加载和评估，未执行后续 optimizer 更新、训练或数据游标续接，不把此前50步续训误作此次加载后的训练。

固定评估向指标函数实际交付33条预测和标签，loss 0.4756673276、accuracy 0.7878787879；此前训练末 loss 0.4761481881、accuracy 0.7878787879，loss绝对差 0.0004808605，原因未知，不宣称位一致或将其归因于checkpoint错误。评估前后模型/优化器/调度器/scaler签名一致；TrainerState原102条日志前缀保留、仅追加一条step100评估日志，global_step不变，其他TrainerState字段未全部比较。四组RNG评估后相同；no_grad、禁止 optimizer.step 与 model.train(True) 是该runner的执行约束，并非全局backward探针。

两份独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-bert821-checkpoint100-load-eval-r4-cuda-replay-independent-terminal-r1/audit.json SHA256 764e6cb139669be45c90ede38c294dee4ddbc97db541a7ed449267553092f9fd；/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-checkpoint100-load-eval-r4-resume-next-terminal-r1/audit.json SHA256 bf8312ccb89a45943a8f5db65bd3149ff8523834b5874ee5b9ff1ffd4ebc8a33；原 receipt SHA256 4a5ef00d93ce7885a9fe4c5eead196ae6e91cc005f5a754d4289453676d8e37e。实际e7 Python与已审df45原位CUDA core为混合身份，非e7原生冷编；三份保护cache不变、功能cache未新增，五锁内核记录和可见FD holder为空，unit自然成功/cgroup空，但FD拒读和端点可见性限制保留。结果只证明该单次加载与固定评估，不覆盖新进程后续更新、逐样本ID验证、predict、长期训练或性能。


## 2026-10-30 恢复后 BERT checkpoint50→100 实用续训

任务已恢复 active。下方 peer-enabled host staging 直接 CUDA 双向同步/异步中转的四进程八阶段结果仍有效，旧 no-auto-peer 显式 peer 复制失败与未执行项继续保留。当前主树 Python 源指纹 e7d1711ffcb67363b63a0ffb0743ec152eed418edbb0150bbfab52e754ab8c90、HEAD 9724b453，复用原位已审 df45 CUDA core/utils（混合身份，并非 e7 原生冷编），从历史 df45 checkpoint50 单次完成 seed821 BERT 的 step51–100，显式将两组 AdamW 设为 nonfused/foreach=False。恢复模型/优化器/调度器签名严格一致，201 个 scalar step=50、603 个参数与 moment 的实际 CUDA placement、Python/NumPy/CPU/CUDA 四组 RNG 核验通过。

50 次续训更新均有 201 个有限 CUDA FP32 梯度，201 个参数每步实际变化；99 个 microbatch 的 sample ID/形状与历史同种子 PT 报告一致，未比较逐 token 字节。两条不同训练 microbatch 的 query projection 观察到 CUDA BF16 输出和 autocast，不代表全程或内核内部精度。续训 50 次更新的平均 loss 为 0.2081167603，历史 PT 为 0.1965655518；单步有正常非单调波动，例如 step51/52 为 0.28931/0.31714、step99/100 为 0.55750/0.08252，不能据此声称逐步单调下降。PT 使用不同 checkpoint 与优化器路径，比较仅为描述性单种子结果。

固定 heldout 的实际预测/标签各 33 条，loss 0.4761481881、accuracy 0.7878787879；PT 对照 loss 0.4836055934、accuracy 0.7878787879。checkpoint100 六个主要文件已保存并重新核 hash，终态模型为 step100、best metadata 为该 checkpoint；尚未新进程反序列化或验证再次恢复。无 predict、native EventQueue、性能或多种子结论。两份独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-e7d171-bert821-resume51to100-r2-cuda-replay-independent-terminal-r1/audit.json SHA256 0981a708aac6b497e5ecd15bc0812819da096807e80a109e046f2b0da820cd60；/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-e7-resume50to100-resume-next-terminal-r1/audit.json SHA256 39a3dde39b3d1128a541b12fc14ab427203ea85594152c5b6f8f381dad31a0ff。保护 cache 不变，功能 cache 新增4项且旧项内容不变；五锁内核记录和可见FD holder为空、unit自然成功，FD拒读与端点快照限制保留。下一步若要验证持续恢复，应独立设计 checkpoint100 的新进程加载与状态核验；跨卡生产修复仍需独立验证。


## 2026-10-30 任务恢复与 peer-enabled host staging 验证

任务已从下方历史暂停状态恢复。peer-enabled host staging r1 单次实际运行四个独立进程，覆盖两张 GPU 的双向传输及同步→异步、异步→同步两种次序，共八阶段；每次目标对源启用 peer 返回 0。实际 CUDA 指针和设备 ordinal、两张 GPU 的进程 UUID、portable pinned host、源流与事件及目标流顺序均已核验。源、host 和目标四个 FP32 值均为 [1.5, -2, 5, 7.5]，consumer 为 [2.5, -8, 13, 20.5]，原始字节匹配且设备读回两次一致。各进程分别编译 PTX 并执行 GPU driver JIT，driver/runtime 活体映射前后相同。

独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-peer-host-async-resume-next-terminal-r1/audit.json SHA256 4c7f11e128f12ce6f812f686fe069530b1cf4a6a7c5de0145c448f56575ea97c，原 receipt SHA256 d7d7ebdb6c474fb34d71ab1b30b318fb3eccffb1f4bcc139a0ed0b4319d0d7e9。六份缓存的前后及终审快照一致，六锁无内核持有记录，unit 自然成功、cgroup 空及本任务进程退出；进程和 FD 可见性、端点快照之间的瞬态变化仍有限制。本实验是四个连续 FP32 元素的直接 CUDA 中转；记录和同步读回会干预执行，异步提交顺序不证明实际重叠，也不覆盖 Jittor Tensor.to、capture、反向、通用布局或性能。此前 no-auto-peer 八案例的显式 peer 复制 FAIL 和未运行项仍保留。


## 2026-10-30 按额度条件暂停

用户设定的 Codex 周额度暂停条件已触发（剩余 19%），任务暂停；现场可见范围内无正在运行的本任务 unit、主 PID 或 cgroup 进程，历史 active/exited 与 failed unit 保留。六份缓存锁无内核持有记录、可见 FD 无 holder；进程和 FD 有权限可见性限制，结论为当前时点快照。HEAD 为 7c078c39056c872316235421d53b0ac89e97babd，原 dirty 和历史产物保留。

peer-enabled 双向同步/异步 host staging 诊断包 /data6/cgq/jittor-lab/_state/release-day23/20261030-peer-enabled-host-async-sequence-runner-r1-hold 的 manifest SHA256 e7617bf62086d5021d2fc8eb3716db43ae738bc226dee9ff806faf36f4e99d77 已经两份静审，尚无现场 binding、未创建 attempt，也未运行该包 GPU 或 JIT；恢复后应重新核现场资源、源码及锁并生成新鲜绑定。该包是直接 CUDA 实验，不是 Jittor 修复或 capture 验收；八案例原 FAIL 和未运行项继续保留。暂停快照 /data6/cgq/jittor-lab/_state/release-day23/20261030-peer-staging-pause-handoff-r1/pause-snapshot.json SHA256 1bdcd3defb4de0c0148c1bdadfc2b02cd7e700439c2d66286f4699bda13d0597。


## 2026-10-30 no-auto-peer 隔离候选跨卡八案例：四项通过、第五项失败

在已审私有冷编候选上，五个独立进程实际调用 Torch Tensor.to。未由测试脚本显式启用 peer 的双向直接复制及 producer→copy→consumer 四项，四元素连续 FP32 源、目标及 pending 中间值按位匹配，实际 CUDA 指针 ordinal/dispatch 和两个物理 UUID 均核对，四项 fallback 0；此项不证明 peer 实际处于关闭状态。第五项在目标 cuda:1 对源 cuda:0 显式启用 peer 返回 0 后，0→1 直接复制的目标两次读回均为 16 字节 ff，源前后保持原值；记录数值不符并首错停止，后续三个显式启用案例未运行。失败案例未执行最终 fallback 和映射复核，原 FAIL 保留。

独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-no-auto-peer-eight-case-r3-resume-next-terminal-r1/audit.json SHA256 dca45d7f0525d5b4737a7f7fc76eb9470543f6ed70682417767d43275b479ef4，原 receipt SHA256 fd5cca369841a9779b426d29d963a320abd0f1563bc17e80b99687aaf87bc625。五份受保护 cache 前后与终审快照一致，候选 cache 新增 8 项且旧项未变；六锁内核记录为空，unit 以失败状态结束、cgroup 空，已记录进程退出。进程和 FD 的权限限制及快照之间的瞬态变化无法完全排除。每案为新进程，未显式启用不等于 peer 关闭；pending 构造期间无 Python 显式同步不证明内部未同步，读回及强同步会干预执行。此结果仅覆盖该隔离候选、两方向和四个连续 FP32 元素，不证明原 CUDA 700 根因，未覆盖 capture、反向或性能，不是主树修复验收。


## 2026-10-30 no-auto-peer 私有 CUDA 冷编及 CPU sentinel

有序跨 CUDA copy 路径移除主动 `accelerator_peer` 调用的单行隔离候选，在独立 JITTOR_HOME 实际完成 CUDA-enabled 原生冷编，unit 自然 rc0，约 381.3 秒；它基于当时主树工作树 `e7d1711ffcb67363b63a0ffb0743ec152eed418edbb0150bbfab52e754ab8c90`，不是主树生产补丁。日志记录 core 编译进度 `228/228`；stamp 列 231 个编译文件，独立审计区分 228 个 core 文件与 3 个 CUDA backend 文件，并核对 817 项 stamp 源输入。实际 private core／utils 的进程映射、路径、hash 与 stamp 核符，CPU sentinel 得到 `[3.0, 5.0]`、fallback 0。首次独立审计有一条将绝对路径误判为不合条件的审计断言失败，已保留；修正审计未重编。

两份独立终审分别为 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-no-peer-cold-r2-independent-terminal-r1/audit.json` SHA256 `07e9ed58b4cdb4acf46c348ad149eb0d52286c3860b06e14b3a0d34ff124c6eb`，以及 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-no-peer-private-cold-r2-independent-terminal-r1/audit.json` SHA256 `396b39d6cb0178afb87de5881f67bb75d3d72fbc574132c38b34a1b35bf7aaac`。五份受保护 cache 前后相同，含私有 cache 的六锁终审无内核记录，unit 成功退出、cgroup 空；快照和进程／FD 观察有可见性限制。

此次只运行 CPU sentinel，尚未执行 GPU tensor、跨卡复制或后续因果测试，不能证明复制正确、原故障根因或性能／capture 行为。移除该路径的主动启用不会撤销已启用的 peer，其他路径也可能启用；旧直接复制及 capture 失败和预算账目保留。


## 2026-10-30 接续状态与跨卡 peer 因果候选

以下为冷编前的历史准备记录；实际冷编状态见上文。

Checkpoint 控制参数任务代码已提交 `0d1030f8cc802e0ef1bbb27a8849aa39c23c2a7d`，三文档测试记录已提交 `7fd69c59bd2134787608486868f09f19d97939a4`：主树 CPU 代表五节点通过；隔离候选 13 节点实测通过后控制器节点过滤导致 unit 严格 FAIL、旧 smoke skip 和历史 PT 全套终态未知均保留。实现仍直接调用目标函数，不重计算 activation、不提供 checkpoint 内存节省，未实现的非默认控制显式拒绝。

此前实际完成的 Qwen checkpoint 51→102、BERT 非 fused 两步、双向手工 host 中转及其各自独立审计见下文：它们使用记录时的源码／checkpoint 身份和干预，不能自动外推到当前提交树、完整模型或其他配置。Qwen 恢复严格 hash 的 `initial_lr` 额外项、BERT 历史 checkpoint-100 严格差异、原跨卡直接复制与 capture 失败均保留；手工中转不构成生产修复。

针对 peer enable 前后同指针跨卡复制结果变化，正在独立准备一个有序跨 CUDA copy 的因果候选：只在隔离源码中移除该路径主动调用 `accelerator_peer` 的一行，保留原 copy／stream 流程；候选补丁 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-no-peer-copy-private-cold-r1-hold/candidate.patch` SHA256 `cff08b726e5b4c1bb995318fcc84e8b591072acfdedae506a798a346b98eaa19`。此为尚在准备的假设检验，当前未构建或运行、未应用主树，也不证明原故障根因或复制正确。该改动不会撤销已经启用的 peer，也不保证其他路径不会启用 peer；跨设备性能、其他环境及 capture／replay 语义未知。后续需冻结源码与资源、独立冷编并按实际条件验证，旧实验和预算不重置。


## 2026-10-30 checkpoint 控制参数任务代码提交

任务代码已提交 `0d1030f8cc802e0ef1bbb27a8849aa39c23c2a7d`（`修复checkpoint控制参数转发`），提交仅修改 `compat/torch/installers/data.py` 的 checkpoint 控制参数处理：默认控制不会误传目标函数，非 reentrant 目标 kwargs 可按约定转发，不支持的非默认控制明确拒绝。它仍是直接调用近似，不提供 activation 重计算或内存节省；已记录的主树 CPU 五节点及隔离候选 13 节点实测各有各自源码／执行范围，不能外推完整 PyTorch、GPU、模型或内存验收。先前隔离运行 pytest 成功后 worker／unit 严格 FAIL、旧 smoke skip 和 PT 全套终态未知继续保留。


## 2026-10-30 主树 checkpoint 控制选项 CPU 代表测试

候选改动已应用到主树工作树，执行源码 fingerprint `e7d1711ffcb67363b63a0ffb0743ec152eed418edbb0150bbfab52e754ab8c90`；直接导入实际主树 `torch.utils.checkpoint.checkpoint`，五个代表节点的 15 个 setup／call／teardown 阶段均通过，测试前后主树函数身份、实际 CPU core／utils 映射与 fallback 0 均已记录。覆盖 `use_reentrant=False` 下默认 RNG 控制与函数 kwargs 转发、`preserve_rng_state=False` 和自定义 context 的显式拒绝及警告只发一次；这是五节点 CPU 小样本，不等于已实现完整 PyTorch checkpoint 语义。该实现仍直接调用目标函数、不重计算 activation、不节省相应内存，其他不支持的非默认控制按实现显式拒绝。

先前隔离候选运行实际 pytest 13 节点／39 阶段通过，但旧 smoke 实际 skip，worker 节点前缀误判导致原 unit 严格 FAIL；原失败及映射／fallback 未落盘限制均保留。此次直接主树五节点通过不能抹去该失败或把旧 smoke 标为执行通过；先前 PyTorch 全套 pytest 的终态仍未知。没有重跑 PT 全套、GPU、模型或 checkpoint 内存测试。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-checkpoint-controls-direct-cpu-independent-terminal-r1/audit.json` SHA256 `a989c48f0feeab9cf5dcb11dd162b958fac29ffc64cd35558045b6a5ed6ac0fc`。unit 自然成功退出；CPU cache 端点无增删且仅锁文件变化，四份保护 GPU cache 不变，五锁内核记录为空；缓存快照和进程／FD 观察存在可见性限制。


## 2026-10-30 checkpoint 选项隔离候选 CPU 实测与控制器失败

CPU 单次 pytest 对隔离 `data.py` 候选实际收集 13 个新节点，保存的 setup／call／teardown 共 39 阶段均为 passed，pytest 汇总 `13 passed, 1 skipped`、返回 0；覆盖控制参数消费、reentrant 目标 kwargs 拒绝、非 reentrant kwargs 转发、反向梯度、警告及不支持选项的显式拒绝。旧仓库 checkpoint smoke 节点在 setup 实际 skip，pytest 保存的理由为 `needs deployed torch-shim + nvcc`，因此旧节点未执行，不能算通过。另有测试插件对该外部测试文件显示 0 executed 的路径记账差异；上述 13 项范围由逐节点阶段记录独立重算。

pytest 返回后，worker 按 `test_checkpoint_options.py::` 前缀过滤，但实际新节点 ID 以 `::test_...` 开头，导致误判收集不匹配并以 rc1 退出，控制器和 unit 亦严格 FAIL；原 receipt／日志保留，不以独立重算改写为整包 PASS。按执行源码控制流，到达 pytest 前的 overlay 路径／hash、公开 checkpoint callable 和 CPU core／utils 活体映射检查未抛错，但失败路径没有持久化当时映射详情，也没有记录终态 fallback，不能声称独立取得它们。先前 PT 全套 pytest 的终态未知；该候选仍是直接调用近似，没有 activation 重计算或节省内存，未覆盖 GPU 或模型。

独立重算审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-checkpoint-controls-cpu-r2-independent-terminal-r1/audit.json` SHA256 `642969eb0efb7abf4ef5b94789f20547764a0421d75d6b1e12501a747867cf52`。CPU cache 端点新增 36 项、无删除，14 个原 shim 文件仅 mtime 变化且内容 hash／大小不变，另锁文件变化；四份保护 GPU cache 不变，五锁内核记录为空，unit 以 exit 1 失败结束且 cgroup 已空。端点快照和进程／FD 可见性有限。


## 2026-10-30 graph capture owner／close CPU fake 原型

独立 Python 状态模型及 12 项标准库测试通过，覆盖 unsupported backend、先分配失败、确定／未知 begin、空图／非空图关闭、设备切换及恢复、嵌套与跨线程 owner 拒绝、未知 end 时保留 hold 并拒绝再次 capture、select／restore 失败及首错保留、已关闭 graph 在恢复失败时保留，以及 fake caller 对 poison 的传播。测试中的两次 `checked_caller` 调用是同一个 fake 函数的重复，不是两个真实调用点的集成验证。原始 10 项测试记录和扩展至 12 项的结果都保留。

真实源码只读 AST 核对显示 `step_capture.py:597` 和 `graph_replay.py:757` 的 capture 调用路径捕获宽泛 `Exception` 后返回、不重新抛出；这说明若未来 native unknown-close 以普通异常传播，必须另外设计调用侧 fail-closed，当前没有运行真实调用路径验证。现有 BackendOps ABI 3 表要求版本和结构尺寸精确匹配，真实 end 的 null 不能区分失败和已关闭空图，fake 的明确状态并非现有接口；模型中的锁、owner token、poison 和 held list 也不能证明真实 allocator 全局状态、线程生命周期、CUDA custom context、外部 backend 或非空图 launch／release 的正确性。unknown 不代表进程仍可安全执行其他 GPU 操作。

这只是 CPU 行为原型与静态源码审阅，没有生产补丁、Jittor 集成、真实 backend 修复或 GPU／JIT 测试；历史 capture 906 与原两次干预观察保留。独立审阅 `/data6/cgq/jittor-lab/_state/release-day23/20261030-capture-fake-prototype-independent-review-r1/audit.json` SHA256 `ae72c4673e9f5f90fe7757ff124fd540b20a7a4cc18c53ce20c88be50026c679`；原型 manifest SHA256 `c27b6bcc746ee465c01f6da846848a39e76cfc9fb1c460b2f769e532c32b1fe4`。


## 2026-10-30 手工 host 中转双卡一阶梯度诊断

主树工作树 fingerprint `4ed94ce1a836ef782dc8d47199a9b0cb86e1eb73752b1de919c6c3cb2ffb35d3` 使用实际 CUDA core，在两个独立进程分别测试 logical GPU 0→1 和 1→0：显式启用目标到源的 peer（两次返回 0，之前隐藏状态未知），对四个连续 FP32 元素手工执行 `x._copy_to_cpu().to_device(target)`，而非改变生产实现。两方向的源前后、host 中间值和目标值均为 `[0.25, -1.5, 2.0, 3.25]`，backward 得到源端梯度 `[2.0, -0.5, 1.0, 0.25]`、loss `4.0625`，与独立 CPU PyTorch 数学参考一致。host 的 public device／placement／location／dispatch 分别为 CPU／0／CPU／`cpu:-1`；源、目标和梯度的实际 DEVICE 指针与对应设备 ordinal、两个物理 GPU UUID、实际 core／utils／runtime／driver 映射及 fallback 0 已核。CPU PT 的原生库映射未采集。

另以独立 tensor 双向分别变更源和目标，并读回两端，观察到两次变更实际发生且另一端值保持，支持该特定样本的复制隔离。该实验含 peer 启用、显式同步和多次读回，范围仅限该四元素连续 FP32、一阶梯度及单线程；不覆盖直接跨卡 `.to()`、非连续布局、其他 dtype、非阻塞、capture／replay、性能或生产修复，也不能推翻历史直接跨卡错误及 CUDA 700／906 严格失败。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-manual-host-autograd-r2-independent-terminal-r1/audit.json` SHA256 `9ae248862e156f253a8a5c5915884c19a3fb6cb7342a3ad41f6f8f26c468e4c3`。unit 自然成功退出；主 GPU 功能 cache 端点新增 12 项、无删除且旧项仅锁文件变化，三份保护 cache 不变，五锁内核记录为空；快照和进程／FD 观察存在可见性限制。r1 未运行，静审发现错误的 CPU dispatch 索引断言，r2 修正并独立审阅后执行，原审计保留。


## 2026-10-30 主树 4ed94 BERT 非 fused 两步恢复诊断

单 seed 821 的真实 BERT 在主树工作树 fingerprint `4ed94ce1a836ef782dc8d47199a9b0cb86e1eb73752b1de919c6c3cb2ffb35d3` 和实际主树 CUDA core 上加载历史 df45 checkpoint-50，完整状态签名严格一致，201 个 scalar step 与 trainer／scheduler 均为 50，201 参数及 402 个 moment 的设备记录为 cuda:0，Python／NumPy／CPU／CUDA 四组 RNG 恢复检查通过。随后有意把两个 optimizer 组的 `foreach`／`fused` 从 `None` 设为 `False`，这是改变原配置的诊断干预；保留原 100 步 scheduler，实际只续训第 51、52 两步，0 skip、fallback 0。每步 201 个梯度均有限 FP32 CUDA，201 个参数均发生变化；每步真实 Python 逐参数 AdamW 更新与 optimizer scope 各 201 次、step 分别为 51／52，记录的 scope 进前／内部／退出 dispatch 均为 cuda:0。该 Python 观察不证明 kernel 内部执行或其他设备上下文行为。

四个 microbatch 的样本 ID／形状及两步 LR 与历史 PyTorch 报告匹配，未逐 token 比较；第 51／52 步 loss 分别为 `0.28930664`／`0.31713867`，历史 PT 分别为 `0.27783203`／`0.27587891`。PT 使用不同 checkpoint 且 optimizer 路径可能不同，差异仅为描述性对照，不证明同状态数值等价或收敛趋势。运行配置为 BF16，记录未建立全程及 kernel 内部精度；同步、逐参数检查和包装会干预执行。此次未做 eval、predict 或 lifecycle，也未生成新的 checkpoint-52；先前 full／resume checkpoint-100 严格状态差异与其他历史 FAIL 保留，不能用这两步取代。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-4ed94-bert-nonfused-r2-independent-terminal-r1/audit.json` SHA256 `249fc6695b642cc297a6549b6c4687a66381218ee2c7d478cb93933d2ae3ff9f`。unit 自然成功退出；主 GPU 功能 cache 端点新增 8 项、无删除且旧项仅锁文件变化，三份保护 cache 不变，五锁内核记录为空；端点快照及进程／FD 可见性有限。r1 静审发现异常处理依赖缺失，r2 修复并完成模拟后才运行，原审计保留；本次不作性能验收。


## 2026-10-30 主树 4ed94 Qwen LoRA checkpoint 51→102 续训

单 seed 821 新用途在主树工作树 fingerprint `4ed94ce1a836ef782dc8d47199a9b0cb86e1eb73752b1de919c6c3cb2ffb35d3` 和实际主树 CUDA core 上加载历史 r10 checkpoint-51，按原 102 步日程实际续训至 102：51 次更新、0 skip，每步记录 112 个有限 FP32 CUDA 梯度，首个续训更新中 112 个 adapter 参数均变化，fallback 0。恢复功能比较通过，112 个 optimizer scalar step 与 trainer／scheduler 均为 51，337 个已记录的 adapter／moment／buffer 设备为 cuda:0，Python／NumPy／CPU／CUDA 四组 RNG 恢复检查通过；337 不涵盖冻结底座全部参数。恢复严格 hash 比较仍为 FAIL：多出两个值为原始学习率的 `initial_lr` 字段，功能比较仅对该已限定差异作例外，不改写严格结果。历史 checkpoint 来自旧 r10，并非本工作树从头训练。

101 个续训 microbatch 的 sample ID／形状和 51 步 LR 与历史 PyTorch full／resume 报告匹配；未逐 token 比较。续训更新窗口等权平均 loss `1.16842480`，两个 PT 参考逐步 loss 平均／最大绝对差分别 `0.00219484`／`0.00686032`；固定 17 样本 eager eval loss `2.45555162`，PT 两个参考均 `2.45664406`。新 checkpoint-102 的 112 个 adapter 均与两个 PT checkpoint 不同，最大绝对差 `0.000506258`、整体差值 L2 `0.0237648`（PT 参考 L2 `8.6498742`）；这只是 adapter 数值诊断，没有 PT optimizer 状态一致性或长期收敛结论。checkpoint 文件和旧 checkpoint 原件身份均已核；另存 `final-state.npz` 的最佳模型与末态 optimizer 组合未证明为同一原子 checkpoint。

训练首个续训更新的两个 microbatch 各观察到首层 q_proj 输入／权重 FP32、输出 BF16、autocast 开启及 CUDA dispatch；不证明全程或内核内部精度。51 次更新均观察到 fused AdamW Python 分支调用，逐参数 `adam_update` 调用为 0，故本次不覆盖非 fused 参数 scope 路径；Python 计数不证明 kernel 内部执行。公开 `jt.fetch` 包装观察到两次调用并返回，但未直接观察 EventQueue。predict／lifecycle 明确未运行，旧严格失败及评估限制保留；同步、读回与包装均有干预，不作性能结论。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-4ed94-qwen-resume-r2-independent-terminal-r1/audit.json` SHA256 `1a1451fe83152e01f8702716961b6c68be4a1cf4503b28a0b82142361b91543f`。本次独立 3000 秒预算内 unit 自然成功并退出；主 GPU 功能 cache 端点增加 428 项、无删除，旧项仅锁文件变化，三份保护 cache 不变，五锁内核记录为空。缓存为端点快照，现场进程／FD 可见性有限；历史预算和原失败不重置。


## 2026-10-30 Qwen checkpoint 51→102 新用途准备

以下为运行前计划的历史记录，实际运行与终审结果见上文。

正在准备在主树当前工作树上进行 Qwen LoRA checkpoint 51→102 的独立续训新用途，预算单列为 3000 秒；这是拟议运行，不是已完成的训练或通过结果。原有训练、续训和严格失败以及原预算账目保持不变，不因新用途而重置。runner 尚未冻结，实际源码与产品、PyTorch 对照、checkpoint／数据身份和现场资源仍须在可执行方案中绑定并审阅；尚未以本计划运行 GPU、JIT 或模型，不能据此推断新工作树的训练或恢复成功。

## 2026-10-30 graph capture backend 契约只读审阅

隔离 r4 的历史观察显示：在设备 0 的 per-thread stream 开始 capture 后，跨设备操作拒绝时当前设备变为 1；沿当前设备结束时 end 返回 401、后续读取返回 906，而另一独立进程显式恢复设备 0 后 end 返回 0、源字节匹配。GDB、同步及设备恢复均为干预，结果不能认定原 CUDA 700 的唯一根因，也不是通用 capture／replay 通过。

只读审阅发现现有 backend 表 ABI 为版本 3 并要求结构尺寸精确一致；新增字段会影响旧表兼容，原回调改签名同样改变二进制调用契约，外部 backend 与消费者未穷尽。当前 end 返回 null 既可能是结束失败，也可能是确认关闭后的空图或不可实例化图，不能据此证明 recording 已关闭；持有分配的 owner、起始设备与线程及非空图 launch／release 的设备归属都需要明确验证。发生关闭状态未知时，是否保留 held allocations、阻止继续 capture 并把致命条件穿过现有 Python 宽泛异常处理，仍是待设计和验证的问题；custom context、并发与错误恢复亦未解决，CoreX 和外部 backend 不可直接外推 NVIDIA 实验。审计仅提出受限方向与后续 CPU fake-backend／真实 CUDA 验证要求，没有补丁、冷编、GPU 实验或合入。审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-capture-backend-contract-readonly-handoff-r1/audit.json` SHA256 `e56b922245228e975e46347cb71981c8ce0008f657b029c3fc134ed4e2da75db`；原 906 严格失败和两次干预观察均保留。


## 2026-10-30 Adam 选项任务代码提交

任务补丁已提交 `672d1fcf47726c6b5a7b546f14aaa5c61b6c4244`（`支持Adam显式关闭批量与融合选项`），只修改 Adam 的前端选项、参数组选项保存和非 fused step 前校验：接受 `foreach`／`fused` 的 `False` 或 `None`，对 `True` 明确拒绝，其他错误类型拒绝；不声称实现 foreach 或 fused。此前主树工作树 fingerprint `4ed94ce1…` 的直接 CPU 七节点及三个新进程序列化恢复通过，隔离候选的 CPU／CUDA 固定梯度四步 PyTorch 对照也通过；这些各自带同步／读回的证据范围与工作树 dirty 身份保留，不等同于该提交树的完整模型、GPU 恢复或性能验收。原控制器入口错误、观察器错误、同进程状态共享及续步严格失败均不改写。


## 2026-10-30 主树 Adam 选项直接 CPU 恢复验证

使用当时工作树源码 fingerprint `4ed94ce1a836ef782dc8d47199a9b0cb86e1eb73752b1de919c6c3cb2ffb35d3` 与主树真实 CPU core、直接导入主树 Adam 模块（无 overlay），七个节点的 21 个 setup/call/teardown 均通过、无跳过。三个不同 `foreach` 组设置（缺省、`None`、`False`）各自保存 checkpoint 并由新进程加载，再进行一步更新；恢复前后两个参数的参数值、m、v、step 与同次父进程记录逐值一致，step 从 1 到 2，`None` 与 `False` 实测各自保留。另测 `foreach=True`／`fused=True` 显式拒绝、组修改和 closure 前置检查、加载后不支持组的拒绝及 AdamW 小样本回归。父子进程实际 core／utils 映射和模块身份已核，fallback 均为 0。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-adam-options-direct-cpu-independent-terminal-r1/audit.json` SHA256 `aa5b6c34182fd8f168e9e38ff2ed79338b6f843c8b01d6e7356c6d16ce4c1ab8`；原 receipt SHA256 `ea5ffdbe1f0ab1ea8cdfa9c96b1c8f938e2985e80ce0858abaafda7a997684fc`。CPU 缓存端点仅锁文件变化，四个受保护 GPU 缓存端点不变；锁内核记录为空、unit 自然成功且终态无活动 cgroup，现场进程与文件描述符可见性有限。此前启动错误、观察器错误与同进程直接状态共享的记录均保留。本次为带同步／读回的小样本 CPU 恢复测试，没有独立 PyTorch 数值对照，也不代表 GPU、模型训练或性能验证。


## 2026-10-30 隔离 Adam 选项候选 CPU／CUDA PyTorch 对照

使用主树原生 CPU／CUDA core 和隔离的两份 Adam 选项候选 Python 模块，两端显式 `foreach=False, fused=False` 构造 Adam；各以四参数固定梯度运行四步，与已保存的对应 PyTorch 参考逐字段比较通过。两端参数和梯度最大绝对差均为 0；m 的最大绝对差 CPU `7.4506e-9`、CUDA `2.9803e-8`，v 两端均 `1.8626e-9`，参数 step 和 14 次逐参数更新已核。CUDA 参数实际在 logical cuda:1、ambient dispatch 为 cuda:0，端点 DEVICE ordinal 与两端 core／utils、CUDA runtime／driver 映射已核，fallback 0。

每端对 `foreach=True` 和 `fused=True` 的顶层与参数组构造以及 step 均观察到显式拒绝；step 拒绝前 closure 未调用，所观测参数与 step 未变。这不代表 True 路径已实现。测试使用同步和读回，范围是固定小样本而非完整模型、性能或无同步训练；PyTorch 原生库映射未采集，候选尚未合入主树。同进程直接状态共享和先前失败均保留。

CPU cache 与 CUDA cache 端点分别保持 1146 和 3033 项、无增删，保护 cache 端点一致，终态锁无内核记录；两个 unit 成功结束，记录 PID 在终审时不在现场。CUDA 运行期间三份按该协议可变的状态文档因补写 capture 记录发生变化，执行源码 fingerprint `f110…` 保持，不能称完整工作树前后字节不变；缓存快照和进程观察亦有可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-oracle-cpu-cuda-r1-independent-terminal-r1/audit.json`，SHA-256 `10f736c562abca446a55d876b4e14554e7d0fed224a246d1fcf58eea27dabe3b`。

## 2026-10-30 隔离 r4 capture 906 的 GDB 双分支观察

两个独立进程均使用隔离 r4 core 与主树 optimizer overlay，先做 CUDA0→1 的 257 元素 warm copy，再观察 capture 拒绝。两分支实际 begin 均返回 0；拒绝后当前线程 runtime／driver device 从 0 变为 1，该线程的 PTDS capture status 为 0。A 沿当时设备结束：end 前 device 1、status 0，GDB 关联实际 end 返回 401，之后源 driver DtoH 返回 906，源后字节未能验证。B 在 end 前显式恢复 device 0：该线程 status 为 1，实际 end 返回 0，之后源双读与原始期望字节一致。两次都记录预期的跨设备 staging 在 graph recording 中拒绝信息。

进程 rc0 表示诊断记录完成，A 的读回仍失败，不能把两分支标作 capture 功能或 replay 通过。GDB 断点、当前线程查询、同步、读回和 B 的设备恢复均有干预；两个分支是不同进程，结果不能单独证明所有线程／流、生产安全或此前 CUDA700／906 的唯一根因。实际 begin/end 的线程、时间窗口、PTDS stream、返回码及 runtime/core 映射已核；原 remaining16 capture 严格失败保留。隔离 r4 cache 端点均为 982 项、无增删且仅 lock 文本变化，三保护 cache 端点一致，锁无内核记录；unit 成功结束、任务 PID 终审时不在现场，快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-r4-capture906-gdb-compare-r2-independent-terminal-r1/audit.json`，SHA-256 `e90092e8f7a8216593ba5d74e9a9cc65710ac64a5f1c81df292ec1724ba56d3f`。

## 2026-10-30 Adam 选项候选 CPU 序列化新进程恢复

隔离两文件 Adam 选项候选的 CPU checkpoint r3 实际完成 8 个节点及 24 个 setup／call／teardown 阶段。新增组 `foreach` 缺省、显式 None 和 False 三种情况下，分别真实 `torch.save` checkpoint 并在三个独立新进程加载、恢复和续步；文件 SHA、进程身份、CPU core／utils 活体映射及 fallback 0 已核，恢复前后两参数的值、m/v 和 step 与原进程对应状态按 `atol=rtol=2e-6` 一致。其余节点覆盖不支持 True／类型的显式拒绝及 closure 前检查、加载不支持组选项的下一步拒绝，以及 AdamW／native Adam 单步回归。

这是 CPU 固定小样本、隔离候选的序列化测试，没有独立 PyTorch 数值、GPU 或模型训练对照；磁盘序列化新进程通过不证明同进程直接 state_dict 加载的状态独立。此前直接加载时 moment 对象共享观察、续步严格失败及观察器失败均保留。CPU cache 端点 1122 → 1146、新增 24 项无删且旧项仅 lock 文本变化，四份保护 cache 端点一致、终态锁检查通过；unit 成功结束，记录 PID 终审时不在现场，快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-checkpoint-r3-independent-terminal-r1/audit.json`，SHA-256 `81f6dd171843ed4d230c2582c08b7d93cbe70d900e260e8939aa5650c485cb94`。

## 2026-10-30 Adam 直接状态共享观察与序列化观察器首错

隔离候选 CPU checkpoint r2 的首个节点实际观察到，同进程将 `state_dict()` 直接加载到另一优化器后，源与恢复优化器的 m 和 v 分别具有相同 Python 对象 id；源优化器单步后，两侧 m/v 数值同步改变，m 的 dispatch 均为 CPU。这支持该特定直接加载工况的状态共享，但不单独证明此前续步比较差异的唯一原因，也不代表序列化或新进程恢复行为。

第二节点已写出 `checkpoint-missing.pt`，随后观察器假定 step 支持 `.item()`，实际取到 float 而报 AttributeError；新进程未创建，checkpoint 未加载或续训，其后 7 个节点未执行。unit 按严格失败退出，原 r1 入口错误及此前续步失败保留；失败 worker 未记录终态映射或 fallback。CPU cache 端点 1110 → 1122、新增 12 项无删且旧项仅 lock 文本变化，四保护 cache 端点一致，终态锁检查通过，记录 PID 不在现场；快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-checkpoint-r2-independent-terminal-r1/audit.json`，SHA-256 `0f5b69678978119a408a1cbc1e3bb390c14160c5f5385e3a42d580b64e9f1a8b`。

## 2026-10-30 Adam checkpoint 后继控制器入口失败

用于观察直接状态别名及序列化新进程恢复的 CPU checkpoint r1 控制器在入口调用 `predecessor.verify()` 时，因未导入 `predecessor` 报 NameError，unit 以 exit 1 失败。错误发生在 attempt 创建和 worker 启动之前，计划中的 9 个 pytest 节点与 checkpoint 子进程均未运行；不能将该结果称为框架验证或推断先前续步差异原因。原 journal 和绑定保留，不自动重试；此前静审未识别该未定义名称。

终审时 CPU、主 GPU 功能、隔离 r4、性能与封存 SDPA 五份 cache 快照与入场审阅相同；五锁无内核记录或可见 FD holder，控制器 PID 不在现场。1755 个 `/proc` FD 目录拒读，端点快照不能排除瞬态活动。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-checkpoint-r1-independent-terminal-r1/audit.json`，SHA-256 `22b2b5fc78c055ea268d489a57a7957a0755827af948ddc5aeafc308deb2c878`。

## 2026-10-30 隔离 Adam 选项 CPU framework 恢复续步首错

隔离两文件 Adam 选项候选的 CPU framework r2 收集 12 节点：四种顶层 None／False 构造和参数组保存组合通过；第 5 项在新增组缺省选项的 state_dict 保存和加载选项断言通过后，同进程依次对原优化器和恢复优化器续步，参数比较失败：原为 `[0.9812166, 2.0200136]`，恢复为 `[0.9801547, 2.0216746]`。按首错停止，余下 7 节点未执行；这不是候选整体通过，也没有独立 PyTorch 或 GPU 对照。原 r1 控制器入口错误及此次失败均保留。

源码显示 state_dict 直接引用 moment Var，恢复时 `_to(copy=False)` 对已在 CPU 的 Var 可能原样返回并安装，因而存在两个优化器共享 moment 的可能；本次没有保存对象身份或各阶段 moment，不能确定它就是失败原因。失败 worker 未保存终态映射或 fallback；CPU cache 端点 1098 → 1110、新增 12 项无删且旧项仅 lock 文本变化，四份保护 cache 端点一致，终态锁检查通过；unit 以 exit 1 失败、记录 PID 终审时不在现场，快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-framework-r2-independent-terminal-r1/audit.json`，SHA-256 `19b0f157598fbe1cdb1d8df5ef69d4e4e3b4422a316f1f0caef5b7844c107c11`。

## 2026-10-30 Adam 选项候选 CPU 控制器入口失败

隔离 Adam `foreach`／`fused` 选项候选的 CPU framework r1 控制器在入口引用不存在的 `source_guard.COLD` 时抛出 AttributeError；实际模块导出的是 `GUARD_COLD`。unit 以 exit 1 失败结束，错误发生在 attempt 目录创建和 worker 启动前，12 个 pytest 节点均未执行；这是控制器准备错误，不能判定候选框架行为或将其标为测试通过。原 journal 和绑定保留，不自动重试。此前静审未识别此属性不匹配。

终审时 CPU、主功能 GPU、隔离 r4、性能及封存 SDPA 五份 cache 快照与入场审阅一致，五锁无内核记录或可见 FD holder，控制器 PID 不在现场；1751 个 `/proc` FD 目录拒读，端点快照不能排除瞬态活动。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-options-cpu-framework-r1-independent-terminal-r1/audit.json`，SHA-256 `d750b4f3abaf50872de641d2d88dadfcca78a866945451c2ad237e8b98785d38`。

## 2026-10-30 隔离 r4 剩余跨卡复制与 capture 首错

基于隔离 r4 host-staging 原生 core／Python 源及显式加载主树 optimizer 模块的一次有界后继中，原 suite 未执行的前 14 个复制节点完成：两方向各四种 dtype 的 17 元素，以及 FP32 两方向在 8 MiB 前、恰好和后各一个元素的边界。每项的源前、目标、源后三份原始字节和 driver 双读均与独立重建的确定性期望一致，实际两卡 DEVICE ordinal、UUID、映射和 fallback 0 已核。同步、指针物化和读回均为干预，不能外推到所有大小、模型或原主树跨卡实现。

第 15 项 CUDA0→1 capture 的 warm copy 字节符合期望，并记录跨设备 host staging 在 Jittor graph recording 期间不支持的拒绝信息；随后 source-after 的 driver DtoH 报 CUDA 906，进程 rc1，未完成源后数值验证、终态 fallback 与映射检查。因此该节点严格失败，不能称 capture 通过或 replay 已验证；第 16 项反方向 capture 未执行。原 CUDA700、隔离候选异常／资源生命周期及 context 等限制仍保留，不能由这些结果确定原故障根因。

隔离 r4 cache 端点 970 → 982、增加 12 项且无删除，旧项仅 lock 文本变化；三份保护 cache 端点一致，终态锁无内核记录。unit 以 exit 1 失败结束、已记录任务 PID 在终审时不在现场；缓存快照和进程观察有可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-r4-devicecopy-remaining16-r2-independent-terminal-r1/audit.json`，SHA-256 `b033b8be360c91828331f2a361014a14c5fd95c013a4338c38977380d1cd6ce0`。

## 2026-10-30 Adam 更新作用域代码提交

任务代码已提交 `c1569515961a5c4b9739c8b4ab2916bb531d9f22`（`修复优化器更新的参数设备作用域`），该提交仅包含 `compat/torch/optimizer_api.py` 的 Adam／AdamW 非 fused 逐参数更新 scope 改动。真实主树工作树 f110 身份下的 CPU 与 CUDA 固定梯度四步对照已完成，独立审计 SHA-256 `e9c2ea3011d87abc7cda5a5bb7b9db951b0bb172706bb6083be80b625c38fa51`；完整工作树还包含既有 dirty 内容，不能把该工作树身份当成提交树身份。测试包含同步和读回，不证明完整模型、无同步运行或跨设备复制；原 Adam `foreach` 构造失败、历史 CUDA 700 及候选限制继续保留。

## 2026-10-30 主树 f110 Adam／AdamW CPU 与 CUDA 固定梯度对照

主树 f110 Python 源直接加载已集成 scope 的 optimizer，复用已审主树 df45 的原生 CPU／CUDA cold core：CPU 两个 Jittor 阶段使用已保存的 PyTorch Adam／AdamW 参考，CUDA 新生成两个 PyTorch 参考后运行两个对应 Jittor 阶段，六阶段均自然 rc0。每种算法各以四个 FP32 参数、固定梯度运行四步，逐字段的参数、梯度、m/v 和 step 与对应 PT 参考通过预注册容差；每个 Jittor 阶段记录 14 次逐参数更新，fallback 0。CUDA 参数在 logical cuda:1（UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`），ambient 为 logical cuda:0；最终 CUDA 参数及状态的 driver DEVICE 指针和 ordinal 已核。

主树 Adam 调用省略不支持的 `foreach`／`fused` 参数，AdamW 两侧显式设为 False；先前 Adam 对 `foreach=False` 的实际构造失败和相关 API 缺口仍保留，不能由此通过结果覆盖。每步均显式同步和读回，观测会干预调度；这是固定梯度小样本，不是完整模型训练、收敛、性能、无同步或跨设备 copy 验证。实际主树模块及 core／utils 和 CUDA runtime／driver 映射记录已核；PT 的原生库映射未采集，CPU 物理设备仅按公开分类。

CPU cache 端点 1090 → 1098、CUDA cache 3029 → 3033，分别新增 8 和 4 项、无删除，已有项仅 lock 文本变化；各自保护 cache 端点一致，终态锁无内核记录。两个 unit 成功结束，记录的任务 PID 在终审时不在现场；端点快照和进程观察有可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-adam-scope-post-r4-independent-terminal-r1/audit.json`，SHA-256 `e9c2ea3011d87abc7cda5a5bb7b9db951b0bb172706bb6083be80b625c38fa51`。

## 2026-10-30 主树 Adam scope 集成后 CPU 首错

主树 f110 源码的直接 CPU 小样本对照中，独立 PyTorch Adam 与 AdamW 固定梯度参考各完成四步、进程 rc0；随后的主树 Jittor Adam 接受 `foreach=False, fused=False` 的构造调用在进入 `step` 前报 `TypeError: Adam.__init__() got an unexpected keyword argument 'foreach'`。堆栈显示 Torch frontend 将参数传至不接受 `foreach` 的 native Adam 构造器，这是该调用实测的 API 失败；尚未运行 Adam scope 更新，不能据此推断 scope 的更新正确性或失败原因。按首错停止，Jittor AdamW 和后续 CUDA 对照均未运行；保留原失败，不自动重试。

失败进程已记录实际主树 optimizer 路径和 CPU core／utils 运行中映射，因构造失败没有产品映射后记录或 fallback 结果；PyTorch 新参考未采原生库映射。CPU cache 端点均为 1090 项、无增删且仅 lock 文本变化，四保护 cache 端点一致、终态锁无内核记录；unit 以 exit 1 失败结束、记录任务 PID 终审时不在现场，缓存快照和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-adam-scope-post-cpu-r2-independent-terminal-r1/audit.json`，SHA-256 `a7850f486d736b5a514611b2a46602f811f0de9a326832d0dabad8d0472a4e41`。

## 2026-10-30 隔离 Adam scope CPU 异常恢复单例

在主树 df45 CPU core 和预导入的隔离 Python scope 候选下，通过真实 AdamW.step 在 `adam_update` 调用内注入异常。进入时观察到 Python placement 与 native TLS 均为 CPU；异常后恢复到事先设置的合成 CUDA1 外层标记，退出外层后恢复测试前初值。这个 CUDA1 只是 TLS 标记，没有真实 GPU 操作；step 在异常前已增至 1，测试不证明优化器状态事务回滚。frontend result-type token 和 runtime device／flags 没有单独验证，异步 GPU 错误和部分 moment 更新亦不在该单例范围。

候选模块与主树 helper 的加载身份、CPU core／utils 活体映射、stamp 和 fallback 0 已核。CPU cache 端点均为 1090 项、无增删且仅 lock 文本变化，GPU 功能及另外三个保护 cache 端点一致；unit 成功结束，记录任务进程在终审时不在现场，终态锁检查通过，进程和端点快照存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adam-scope-exception-cpu-r3-independent-terminal-r1/audit.json`，SHA-256 `4bbb0d9df5c43df6d085c88bbccdbef893ae1add396b99512017c4918d84d7cf`。

## 2026-10-30 主树 core 与隔离 Adam scope Python 候选定向对照

在 df45 主树 Python 和原主树 CUDA core 上，预导入加载仅 `optimizer_api.py` 的隔离 scope 候选，一次串行 suite 的 13 个独立阶段自然 rc0：1 个新 PyTorch 双卡分组固定梯度参考与 12 个 Jittor 候选阶段。CPU+CUDA、双 CUDA 同组、双 CUDA 分组各四步及其 checkpoint 第 2 步新进程恢复至第 4 步均通过预注册逐字段对照；恢复初态和后续结果也与各自 Jittor full 第 2 步和尾段对照。两设备的进程 UUID 以及 CUDA 端点参数、梯度、m/v 的 driver DEVICE 指针和 ordinal 有记录；CPU 端点的设备为公开分类，不是 driver 内存证明。实际 overlay、core、utils 和 CUDA runtime 映射前后身份及 fallback 0 已核。

固定梯度场景还覆盖四步中间无显式逐步读回、手工梯度 AMP、FP16、BF16、capture 拒绝及 fused：fused 真实选择和调用各 4 次，逐参数 fallback 0；capture 拒绝不表示 replay 已验证。无逐步显式同步不证明底层无内部同步，其他阶段的观察和端点指针检查会同步并物化。各项按预注册的 dtype 对应容差比较；新双卡 PyTorch 参考未采原生库实际映射，历史 PT 参考按保存工件绑定，恢复并非 PT checkpoint 续训对照。这是四参数小样本功能实验，不覆盖真实模型训练、收敛、性能、完整 AMP／GradScaler 或所有设备路径，Python 候选尚未合入主树；历史 CUDA 700 和先前严格失败保留。

主功能 cache 端点 2985 → 3029 项，新增 44、无删，旧项仅 lock 文本变化；隔离 r4、性能及封存 SDPA 三份保护 cache 的端点快照相同。unit 成功结束，记录的任务进程在终审时不在现场，终态锁无内核记录；缓存端点和进程观察存在可见性限制。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-adam-scope-overlay-functional-r2-independent-terminal-r1/audit.json`，SHA-256 `958186907811568d5909d95fda64c8a653dafc6ac0eb1895c03372cb6fa6051c`。

## 2026-10-30 主树 df45 core：临时 scope 的 Adam 单步对照

在真实主树 df45 Python 源及其原主树 CUDA cold core 上，以独立进程对 ambient CUDA 0、参数位于 CUDA 1 的场景执行一次 Adam 更新。测试在调用期间临时包装 `adam_update`、`optimizer_state_scope` 和 `_optimizer_arithmetic` 并观察实际分派；p0、p1、p3 三个有梯度参数参与更新，p2 无梯度。与保存的 PyTorch 参考相比，梯度精确一致，参数和 m/v 在 `atol=rtol=2e-6` 下通过，step 已检查，fallback 为 0。实际 core 和 CUDA 库的映射身份以及目标参数的 DEVICE 指针、ordinal 1 均有记录。

逐参数观测中，scope 前、内、后的 driver context 和 runtime device ordinal 均为 0，而 dispatch 为 `cuda:0 → cuda:1 → cuda:0`；实际分支记录为 `native_or_other`。测试未证明 driver context 切换到 1，也不能确定先前 CUDA 700 的根因。观察器的临时包装、指针物化及 scope 外强同步改变了执行条件；这只是受干预的单步对照，不是无干预多步训练或永久产品修复。此前隔离候选 suite 的 CUDA 700 原始失败继续保留。

主功能 cache 的端点记录为 2981 → 2985 项、增加 4 项且无删除，既有项中仅锁文件文本变化；隔离 r4、性能和封存 SDPA 三个保护 cache 的端点快照一致。systemd unit 自然成功结束，已记录的任务进程退出；锁与进程审阅受可见性限制。独立终审：`/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-adam-control1-scoped-r2-independent-terminal-r1/audit.json`，SHA-256 `4dfee9f053d4d13ddfb80cb686acff683cccdd694b8bfb754cadce11513a5d58`。

## 2026-10-30 隔离 r4 Adam 环境与 placement scope 对照

使用隔离 r4 冷编产品，两个独立进程分别执行 CUDA1 参数的单步 Adam 对照：ambient CUDA1，以及 ambient CUDA0 且仅在 Adam 调用中进入参数 placement scope。两项均完成 p0、p1、p3 三个有梯度参数的更新；梯度与保留的 PyTorch 参考精确一致，参数和 m／v 在预定 `atol=rtol=2e-6` 内一致，fallback 0。

在 ambient0 scoped case 的三个参数上，依次采样的 driver context ordinal 和 runtime current device 均为 0，scope 前／内／后 ambient dispatch 为 CUDA0→CUDA1→CUDA0；记录的算术分支为 `native_or_other`。这些读数并不表示实际 CUDA context 已切到 CUDA1，也不证明唯一算术路径或先前 CUDA 700 的根因。scope、指针物化和强同步改变调度；这只是隔离候选上的单步小型对照，并非原主树或完整模型验证。此前 suite 的 CUDA1 Adam CUDA 700 和其他未执行项保留，候选异常恢复、非 primary context 和 capture 并发限制仍未解决。

候选 cache 端点新增 4 项、无删且既有项仅锁文本变化，三份既有保护 cache 端点一致；unit 成功结束、任务 PID 不在现场，阶段间任务 cgroup 核验仅 controller；端点快照和 `/proc` 可见性受限。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-r4-adam-control1-scoped-r2-independent-terminal-r1/audit.json`，SHA-256 `67c7a44f28cb9b4971cc74ce4b34650cb44e82a7b54b158eee228fbb6551d853`。

## 2026-10-30 隔离 r4 staging 候选定向功能首错

基于 r4 私有冷编产物的一次定向 suite 在首错停止前完成五项：FP32 scalar 的 CUDA0→1 和 CUDA1→0 原始字节复制，两方向各 257 元素的 producer→copy→consumer 链输出逐字节符合独立期望，以及 CUDA0 上四参数、三个有梯度更新的 Adam 单步与保留 PyTorch 参考在预定容差内一致。pending 链没有显式 Python 同步，但 staging 后端本身会同步，观测也会物化和同步；这五项实际 CUDA 设备和 core／utils／runtime／driver 运行时映射前后记录核符，fallback 0。

第六项 CUDA1 Adam control 的首个参数更新结果 `sync(True, False)` 期间，设备查询 `cudaMemGetInfo` 报 CUDA 700／illegal address，worker 以 rc70 停止，首更新未完成；launch ring 仅提供候选调用，不能证明肇事 kernel 或先前 CUDA 700 的根因。该失败 worker 仅有映射前记录、没有终态设备 PID／UUID 记录。按首错停止，后续 16 项未执行，包括 8 MiB 边界、多 dtype 和 capture；不能将前五项外推为完整跨卡、模型或错误恢复通过。

候选仍仅在隔离源和私有 cache，未知完成状态下的设备分配生命周期、非 primary context、capture 并发及异常恢复限制未解决，不能直接合入。候选 cache 端点增加 20 项、无删且既有项仅锁文本变化，三份既有保护 cache 端点一致；unit 按失败退出、任务 PID 不在现场，终态锁记录核验通过；端点快照和 `/proc` 可见性存在限制。原失败均保留，不自动重试。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-staged-copy-candidate-functional-r2-independent-terminal-r2/audit.json`，SHA-256 `ff90f05d2cf42b31c9fec2b66a78e518dbe2e94d8e0cd1174175ac23217efbe9`。

## 2026-10-30 隔离 r4 跨设备 host staging 候选 CUDA 冷编

在独立源副本及专用 functional cache 中，ordered 跨不同 CUDA 设备的 pageable host staging r4 候选完成一次真实 CUDA 冷编：日志记录 core 228/228 与 CUDA backend 3/3，stamp 列出 231 个编译单元，controller 自然 rc0、约 393.38 秒。源副本与 df45 执行清单的区别仅为候选 driver，38 个忽略 pyc 按已审资源绑定；实际私有 core／utils／stamp 的磁盘路径、内容与身份已核。逻辑 CUDA0 对应 GPU UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`，单卡基础算术 sentinel 得到 `[3, 5]`、fallback 0；没有单独执行 CPU sentinel。

该冷编和单卡算术不验证跨卡 staging 功能、双卡模型或错误恢复，也不能作为可合入结论。未知完成状态下源／目标设备分配的生命周期、非 primary context 身份与恢复、capture 并发、失败资源累积及设备范围同步的影响仍待处理或验证；原 CUDA 700 和所有失败证据保留。构建进程运行时的实际共享库 maps 未采样，已记录 readelf 与当前库磁盘身份，不能据此冒称实测映射身份。三份既有缓存端点一致，候选及既有锁现场无记录且可见 holder 空，unit 成功结束、任务 PID 不在现场；1792 个 `/proc` FD 目录拒读，端点快照和采样存在可见性限制。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cuda-staging-r4-private-cold-independent-terminal-r1/audit.json`，SHA-256 `35a01578d7e70853a741521b2a9c2e45f6a92701b2e060b2ea1d91e20181ffea`。

## 2026-10-30 启用 peer 后 driver 与 host 中转对照

在单个独立进程中，显式 target 1→source 0 peer enable 返回 0 后，针对逻辑 CUDA0→CUDA1 的同一源／目标指针对固定依次执行 C→H，传输一个四字节 FP32 scalar。C 使用目标 context 下 driver `cuMemcpyPeer`，目标双读 `ffffffff`（NaN，MISMATCH）；H 实际从源做两次 driver DtoH，得到 `e286013d` 的 host buffer，再在目标 context 同步 HtoD，目标双读 `e286013d`（MATCH）。每步源前后为 `e286013d`，目标事先重置并双读 `a5a5a5a5`，设备、context、线程、地址、返回码及次序记录已核；进程 rc0 表示记录完成，不能将 C 的错误结果标为通过。

这是先 C 后 H 的单次、单方向、四字节观察；host 中转、重新设置 sentinel、同步和双读均会改变执行，不能证明 H 对其他大小、方向或长期运行可用，也不是通用 fallback 或性能结论，不能单独确定此前 CUDA 700 的根因。实际 core／runtime 映射、两卡 PID／UUID 和 fallback 0 已记录；功能缓存无增删且仅锁文本变化，两份性能保护缓存端点一致，unit 成功结束、任务 PID 不在现场、终态三锁无内核锁记录；端点快照与 `/proc` 可见性受限，不能排除瞬态活动。此前失败与对照保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-peer-enabled-host-stage-r1-independent-terminal-r1/audit.json`，SHA-256 `8e698bd03ce26eb08cd50f1f48f907eafe67142983cbb75c2eb6dd269b565a5d`。

## 2026-10-30 异步复制显式 peer enable 前后对照

在独立单进程、逻辑 CUDA0→CUDA1 的同一源／目标指针对上，使用同一个目标 nonblocking stream 和同一对 events 执行两次四字节 `cudaMemcpyAsync`：D0 目标双读为 `e286013d`（MATCH），随后显式 target 1→source 0 peer enable 返回 0；D1 目标双读为 `ffffffff`（NaN，MISMATCH）。两次复制及两设备同步均记录返回 0，每步源前后双读 `e286013d`、目标预写 sentinel 双读 `a5a5a5a5`；同一线程的指针、context、event／stream 次序与结束后资源清理已核验。进程 rc0 表示观测完成，不能当作复制功能通过。

D0 前源码路径没有显式 peer enable，但资源创建或框架可能改变隐含状态，且固定先后顺序、sentinel 重置、事件、同步及读回均会干预；该单 scalar、单方向观察不能单独确定唯一原因或此前 CUDA 700 的根因。实际 core／runtime 映射、两卡 PID／UUID 与 fallback 0 已记录；功能缓存 2981 项无增删、仅锁文本变化，两份性能保护缓存端点一致，unit 成功结束、任务 PID 不在现场，终态三锁无记录；终态前现场检查的可见 holder 为空、约 1787 个 `/proc` FD 目录拒读，端点快照和采样仍有可见性限制。此前失败与对照保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-async-peer-transition-r1-independent-terminal-r1/audit.json`，SHA-256 `ab2344668889cfd31c9a4a4f96b5bf30b50bd2e987b225637166366067c3cb0c`。

## 2026-10-30 同指针对显式 peer enable 前后对照

在单个独立进程中，逻辑 CUDA0→CUDA1 的同一保活源／目标指针对，第一次在目标 context 执行 driver `cuMemcpyPeer` 后目标双读为 `e286013d`（MATCH）；接着显式执行目标 1 到源 0 的 `cudaDeviceEnablePeerAccess`，返回 0 且清除状态返回 0；再以同一指针对执行第二次 `cuMemcpyPeer`，目标双读 `ffffffff`（NaN，MISMATCH）。两次复制 API 均返回 0，每步源前后双读 `e286013d`、目标 sentinel 事先双读 `a5a5a5a5`，两次复制的指针、设备、context 和线程顺序已核；进程 rc0 表示观测完成，不表示该复制行为通过。

第一次复制之前，源码可见路径没有显式 peer enable，但原有隐含 peer 状态及第一次复制对状态的影响未知；sentinel 重置、同步、读回及固定先后顺序都会干预，不能单凭这次前后观察确定唯一原因或此前 CUDA 700 的根因，也不能推断其他方向、大小和配置的结果。实际 core／runtime 映射、两卡 PID／UUID 和 fallback 0 已记录；功能缓存无增删且仅锁文本变化，两份性能保护缓存端点一致，unit 成功结束、任务 PID 不在现场，三锁与可见 holder 空；约 1784 个 `/proc` FD 目录拒读，端点快照与采样仍有可见性限制。此前失败与对照保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-peer-enable-transition-r1-independent-terminal-r1/audit.json`，SHA-256 `417b63b0e1f4e93b6690f5a83acba4ca36caa4bfdef92308a470a74b0fa0f073`。

## 2026-10-30 同一指针对三种跨设备复制顺序对照

两个独立进程分别按 C→D→P 和 P→D→C 执行逻辑 CUDA0→CUDA1 单 scalar 控制，其中 C 为目标 context 下 driver `cuMemcpyPeer`，D 为 runtime `cudaMemcpyAsync` 默认方向，P 为 runtime `cudaMemcpyPeerAsync`。每进程内三步使用同一已核验的源／目标指针对，且每步源双读 `e286013d`、目标预写 sentinel 双读 `a5a5a5a5`；三种操作之后目标均双读 `ffffffff`（NaN），源仍双读 `e286013d`，六个观察均为 MISMATCH。两个进程 rc0 表示记录完成，不能作为复制功能通过；两进程的实际分配地址彼此不同。

两进程各自先启用目标到源 peer 访问，后续 D/P 均重新执行 runtime prepare／peer enable，记录返回依次为 0、704、704；D 两次均处于第二步。目标 sentinel 写入、同步、事件链和 driver 读回均会干预执行，顺序或已有状态可能影响结果，不能由这两种顺序推断完整排列或此前 CUDA 700 的根因，也不能将该结果泛化为所有复制。实际设备、context、线程、指针和 core／runtime 映射及 fallback 0 已记录；功能缓存无增删、仅锁文本变化，两个性能保护缓存端点一致，unit 成功退出且任务 PID 不在现场，三锁及可见 holder 为空；约 1775 个 `/proc` FD 目录拒读，端点快照和离散采样仍有可见性限制。此前失败与对照全部保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-same-pair-sequence-r1-independent-terminal-r1/audit.json`，SHA-256 `b1b0c6bab4f4141bd6b9d38dbb2a061cd45cdcb54c735388d8ee797d011d9ab7`。

## 2026-10-30 CUDA runtime 目标指针先读对照

在独立单次 DS 控制中，目标 sentinel 事先双读为 `a5a5a5a5`，从逻辑 CUDA0 向 CUDA1 的四字节 `cudaMemcpyAsync`、事件链及两设备同步均记录返回 0。完成后，源码可见路径在显式目标 Var sync／地址 getter 之前，保活原目标并以已核验的 device/context/地址范围直接做两次 driver 读回，均为 `ffffffff`；随后执行目标 Var sync 和 getter，地址仍为 `0x78819f425600`，核验后再次双读仍为 `ffffffff`，源前后双读均为 `e286013d`，分类为 `BOTH_MISMATCH`。rc0 表示观测采集完成，不能当作复制正确。

这排除了该次记录中“显式目标 Var sync／getter 之后才第一次出现错误值”的狭义解释；同步 DtoH、地址属性查询、事件和设备同步本身均可能干预，地址相同也不能证明没有同址写入，框架或 CUDA 内部隐含行为仍未知。该单 scalar／方向结果不能确定此前 CUDA 700 根因或代表所有复制。实际 core／runtime 映射、两卡 PID／UUID、fallback 0 已记录；功能缓存无增删、仅锁文本变化，两个性能保护缓存端点一致，任务 unit 成功退出，三锁及可见 holder 为空；约 1793 个 `/proc` FD 目录拒读，端点与采样存在可见性限制。此前六 case、driver、GDB 与原失败保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-pointer-first-r1-independent-terminal-r1/audit.json`，SHA-256 `2927aff26f2e548897337c6e34828b1fe3e7366c83e877098ca29ccb035a4ae8`。

## 2026-10-30 CUDA runtime 跨设备 scalar 六控制结果

六个独立进程分别对逻辑 CUDA0→CUDA1 的同一已知 FP32 scalar 执行四字节 CUDA runtime 控制：`cudaMemcpyAsync` 默认方向推断与 `cudaMemcpyPeerAsync` 各测无事件链、显式事件链以及事件链加目标 sentinel 的变体。六个 case 的原始记录均为源迁移前后各两次 driver 读回 `e286013d`、目标迁移后两次 `ffffffff`（NaN）；两个 sentinel 变体在复制前已双读确认目标为 `a5a5a5a5`。每个 case 的复制调用和两设备同步返回 0，记录的指针、设备、context、单线程与事件／stream 顺序通过对应校验；六次进程 rc0 仅表示采集结束，六项观察均为 MISMATCH，并非复制功能通过。

这些是同一方向、同一数值和特定构造／初始化条件下的有限实测；显式物化、同步、读回、事件链及 sentinel 都可能改变执行时序，runtime 与 driver 对照仍不能单独确定此前 CUDA 700 的根因，也不能推断所有跨设备复制的行为。六个独立 worker 的实际两卡 PID／UUID、core 与 CUDA runtime 映射和 fallback 0 已记录；功能缓存前后均为 2981 项、无增删且仅操作锁文本变化，两份性能保护缓存端点一致，任务 unit 成功结束、任务 PID 不在现场，三锁无记录且可见 holder 为空。约 1789 个 `/proc` FD 目录拒读，端点快照与离散采样不能排除不可见或瞬态活动。原 mixed、分段、driver 两 case 与 GDB 观察及失败均保留。

独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cudart-cross-device-control-r3-independent-terminal-r1/audit.json`，SHA-256 `1b41b0b4481e8af0e81d05e4150009d0f531887359f9af40fdf605b7ae09fb77`。

## 2026-10-30 跨设备 scalar 控制与 GDB 调用记录

在两个独立进程、同一已知 FP32 scalar、逻辑 CUDA0→CUDA1 的单次诊断中，driver host-staged 和 driver peer 控制各自记录源迁移前、目标迁移后、源迁移后均为 `e286013d`；另一个进程的 Jittor `.to` 记录源值前后仍为 `e286013d`，目标两次 driver 读回均为 `ffffffff`（NaN）。两个 case 进程 rc0 表示记录和分类完成，不代表 Jittor 跨设备迁移功能通过；不同方法使用各自构造的 scalar，不能称所有复制均有同样问题。

另一个受控 GDB 单例中，实际同线程 `DeviceCopyOp` 与 `backend_copy` 记录 4 bytes、源 device 0 到目标 device 1、ordered；`cudaMemcpyAsync` 记录相同源／目标指针、4 bytes、kind 4 及目标 stream，入口对应返回 0。事件记录／stream wait 和目标设备同步的返回也记录为 0，但目标仍读为 NaN、源值保持 `e286013d`。异步 API 返回 0 不证明目标数据正确，也不能单独确定此前 CUDA 700 的机制；GDB 断点、强同步、设备初始化和读回均改变时序，该单 scalar、单方向结果不构成一般硬件或完整 optimizer 结论。

两项诊断的源码、core 与保护缓存端点核符，功能缓存无增删、操作锁文本变化，任务 unit 自然完成且 cgroup 清空，三锁与可见 holder 空；`/proc` 权限拒读、离散采样和端点快照限制仍在。两 case 独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-cross-device-scalar-two-case-r1-independent-terminal-audit-r1/audit.json`，SHA-256 `2f10a6b93d0afa2ef21b0b9bb6f6c2cd2a6c7f7ebde5e6d06cfbc9cb835f0afc`；GDB 独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-device-copy-gdb-production-r2-independent-terminal-audit-r1/audit.json`，SHA-256 `5428b142a46c6459f8ff9c3f4595d543765d3ebc6cebdddf99ba0a046311998d`。


## 2026-10-30 跨 GPU scalar driver 读回校准失败

在显式同步、driver context push/pop 和两次 DtoH 读回的单次正向校准中，逻辑 CUDA0 的 native scalar 在 clone 前后均为 FP32 位模式 `e286013d`，与 CPU PyTorch 参考一致；显式 `.to(cuda:1)` 后返回值经强同步，所观测指针为 CUDA DEVICE memory、ordinal 1，两次 driver 读取均为 `ffffffff`（NaN），未匹配参考。该次复制前后指针与属性、context 恢复端点检查通过，两张物理 GPU 的本进程归属均有记录。

这是干预下的一次迁移后异常，不能单独证明 backend 哪一环出错或此前 CUDA 700 的根因；迁移后源 scalar 尚未复读，反向设备顺序及后续操作未执行。此前 mixed、分段及本次读回的严格失败原件均保留，没有自动重试。功能缓存无增删、锁文本变化，源码和 core 产品及两处保护缓存端点一致，unit 失败退出且 cgroup 已清空，三锁与可见 holder 空；`/proc` 权限拒读、离散观察及端点快照不能排除瞬态变化。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-gpu-driver-readback-calibration-r1-independent-failure-audit-r1/audit.json`，SHA-256 `fb89b51dba181a90987f1fbc14935754c24f55b10f3a22582c0bdf0aca73b99f`。


## 2026-10-30 双 CUDA 基线错误的分段定位

在显式同步和指针物化的单次分段诊断中，control0 的 3 个有梯度参数完成首步，与已保存 PyTorch 首步对照最大绝对差为 0；21 个所观察指针均为 CUDA DEVICE memory、ordinal 0。control1 将全部参数置于逻辑 `cuda:1`、ambient dispatch 为 `cuda:0`；首个参数的参数、梯度及两个 moment 指针均为 DEVICE memory、ordinal 1，而采样的当前 CUDA context ordinal 为 0。随后首个 Adam 更新结果的 `sync(True, False)` 报 CUDA 700 `cudaErrorIllegalAddress`。

这缩小了有同步干预的观察范围，但不能证明 context 差异导致错误，也不能根据有限近期 launch 记录认定肇事内核；显式同步与指针物化改变原始异步调度。失败 worker 未到终态 PID→UUID 查询，逻辑到物理映射仅有 `CUDA_VISIBLE_DEVICES` 与指针 ordinal 支持；mixed、初始快照和设备顺序反转的后续三个阶段未执行，未自动重试。此前 r4 因启动命令绑定路径占位符未替换而在 worker 前退出的操作者错误及原件同样保留。

功能缓存新增 4 项、无删除，锁文本变化；源码、core 产品和两处保护缓存端点一致，unit 失败退出且 cgroup 已清空，三锁与可见 holder 空；`/proc` 权限拒读、离散观察与端点快照限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-two-gpu-segment-r6-cuda-replay-failure-audit-r1/audit.json`，SHA-256 `a6767e10196dfe5caf3eb594494d80ea31b5be41addee7607439279d6c329904`。


## 2026-10-30 隔离 AdamW 候选 mixed-device 与双卡基线失败

CPU 与单 CUDA 同组固定梯度 full 及各自从第 2 步 checkpoint 在新进程恢复的 PyTorch、Jittor 基线和隔离候选共六阶段通过；候选 full helper 为 14 次调用、4 次命中，resume 为 8 次调用、2 次命中。该覆盖限于小型 optimizer fixture 与实际设备状态，不代表完整模型训练或性能。

PyTorch 双 CUDA 同组 full 阶段通过；Jittor 基线在双 CUDA 同组首个更新期间，于逻辑 device 1 同步时检测到 CUDA 700 `cudaErrorIllegalAddress`。前三个参数的更新调用有记录，但无完整步骤；双 CUDA 候选及后续阶段未运行。日志近期 launch 候选不能证明故障内核或原因，也不能归因于尚未运行的候选；严格失败原件保留，未自动重试。功能缓存新增 8 项、无删除，锁文本变化；源码、core 产品和两处保护缓存端点一致，unit 失败退出且 cgroup 清空，三锁及可见 holder 空，`/proc` 权限拒读和离散观察限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-cuda-fallback-mixed-r2-cuda-replay-failure-audit-r1/audit.json`，SHA-256 `55f9a20bb993d6d87a2a52ea730eebd91fe4fa3287eeeda83e748f585e5fca4d`。


## 2026-10-30 隔离 AdamW 候选单 GPU fallback 验证

单 GPU 固定梯度实验以 PyTorch、Jittor 基线和隔离候选分别运行 AMP optimizer 上下文、FP16、BF16 和 capture 拒绝四类 fixture，各 4 步，共 12 阶段通过。AMP、FP16、BF16 候选 helper 均为 14 次调用、0 次缓存命中；capture 阶段活跃捕获的 3 次 helper 调用无缓存命中且发生预期拒绝，replay 为 0，非活跃调用另有 6 次命中。基线与候选的对应值、dtype 和设备精确对照；PyTorch 低精度 moment 使用参数 dtype，Jittor moment 为 FP32，跨框架按预先登记的对应 dtype 容差比较，不能称为位等。

这是固定梯度下的 optimizer 局部检查，不覆盖完整 AMP 训练、GradScaler、capture replay、mixed device 或性能；候选尚未合入主树。功能缓存新增 44 项、无删除，锁文本变化；两个保护性能缓存端点一致，systemd unit 自然成功、cgroup 清空，三锁与可见任务进程/holder 为空。`/proc` 权限拒读和离散观察限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-cuda-fallback-single-r1-cuda-replay-terminal-r1/audit.json`，SHA-256 `02e047a7f874dda7c2279305edcfdb3aa4112949be20f0133f300e91c78bff61`。


## 2026-10-30 隔离 AdamW 候选 BERT 48 预热 A/B 严格失败

基于此前 30 步池轨迹，新用途 A/B r2 在双方采用固定 48 次预热并保持原 1 MiB 热态稳定门槛。baseline0 完成；candidate0 预热末稳定，但第 6 个热样本 reserved 增长 90 MiB，超出门槛，allocated 恒定且热态 JIT 快照不变。基线在第 4 和第 30 次预热各扩张 90 MiB，候选在第 26 次预热扩张 90 MiB；其具体分配机制仍未知，不能认为延长预热可以保证下一个窗口稳定。

严格 FAIL 原件保留，其余四阶段未运行、没有自动重试。单个不稳定候选／基线热样本中位比约 `0.9803` 仅供探索，不能作为正式配对速度比或收益结论；candidate 未生成最终 result，因此没有该阶段完整终态 fallback 和 source-after 证据。不再以单纯延长预热作为后继方案。unit 以失败退出且 cgroup 已清空，源码与 core 产品、两处保护缓存端点一致，私有性能缓存无增删，锁与可见 holder 空；`/proc` 权限拒读、离散采样和端点快照限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-bert-performance-ab-warm48-r2-cuda-replay-failure-audit-r1/audit.json`，SHA-256 `2a3add79ae0670e22f2a948329aa26322ffb18b6645c6689c99e3502111b3db2`。


## 2026-10-30 隔离 AdamW 候选 BERT 双侧 allocator 轨迹诊断

在已审真实 CUDA core、相同 BERT base B8/S256 FP32 配置下，隔离基线与候选各执行首步、16 次预热和 30 次带同步及观测的更新。基线预热后 reserved 为 `4266655744` bytes，第 14 个诊断更新升至 `4361027584`；候选预热后为 `4172283904`，第 10 个诊断更新升至 `4266655744`，各增长 90 MiB，之后到第 30 步未再增长。两侧 allocated 恒为 `2184966656` bytes，32 份 JIT 快照在各自进程内一致；候选首步 helper 调用 197 次、命中 196 次。

这是单次有限的池聚合轨迹；逐步同步、扫描和记录会扰动执行，不能确认具体分配位置、长期稳定或性能收益。此前 A/B r3 的严格 FAIL 仍然保留，不能由此改判。unit 自然成功并清空 cgroup，源码与 core 产品、功能及封存 SDPA 缓存端点一致；独立性能缓存 1227 项前后无增删，两个探针文件仅 mtime 变化、锁文本变化，三锁及可见 holder 空。`/proc` 权限拒读、离散采样和端点快照限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-bert-allocator-trajectory-r3-cuda-replay-terminal-r1/audit.json`，SHA-256 `cd443b4bfc8116fb0b316356138dc62ae80636df22615ff6481ccf4039cb8a63`。


## 2026-10-30 隔离 AdamW 候选 BERT A/B 热态诊断失败

新用途诊断 r3 保持每侧首步、固定 16 次预热及 10 次热更新的原严格稳定门槛，并在计时结束后逐样本保存 allocator 与 JIT 快照；新增扫描和写盘可能扰动后续样本。baseline0 完成，candidate0 首步 helper 197 次调用、196 次命中，预热末两次稳定、10 个热更新 loss 均有限，但热态 reserved allocator 前九次为 `4172283904` bytes、第十次为 `4266655744` bytes，范围 90 MiB，超出原 1 MiB 门槛；allocated 恒为 `2184966656` bytes，12 份热态 JIT 快照一致。baseline 的同一 90 MiB 扩张发生在预热第 4 次后并随后稳定；具体分配机制及长期行为未知。

该次严格 FAIL 保留，未产生正式配对速度比，也不能反推此前 r1 合并断言失败的原因。unit 失败退出且 cgroup 已清空；三处缓存锁无内核持锁记录，源码、产品和两处保护缓存端点一致，私有性能缓存无增删，仅探针 mtime 与锁文本变化。FD 权限拒读及离散采样限制现场结论，端点快照不排除瞬态活动；未自动重试。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-bert-performance-ab-r3-independent-failure-audit-r1/audit.json`，SHA-256 `1673299b3f176a114d31e33248f7dcf20bb3b511f87b21ebcc36cff01061e9e4`。


## 2026-10-30 隔离 AdamW 候选 BERT 性能 A/B 单次失败

在当前 df45 源码和已审独立性能 CUDA core 上，按 AB/BA/AB 计划运行 BERT base B8/S256 FP32、eager、nonfused 的基线与隔离候选性能对照。baseline0 完成首步、16 次预热、10 个热样本及各 3 次独立 memory/attribution 更新；candidate0 首步 helper 197 次调用、196 次命中，16 次预热末两次达到稳定门槛，10 次热更新的 loss 有限，但随后因合并断言 `hot JIT or allocator unstable` 失败。热样本时序、热态 JIT 变化及 allocator 明细未在异常前落盘，不能确定是哪一条件触发；candidate 结果与配对速度比均未产生，后续四阶段未运行，失败原件保留且没有自动重试。

该实验不能证明候选性能收益或正式性能门槛通过。systemd unit 失败退出且 cgroup 已清空；私有性能缓存新增 4 个同一 JIT 算子相关文件、无删除，两个数学探针文件仅 mtime 变化、锁文件文本变化；源码及 core 产品和两处保护缓存端点保持一致，三锁无内核持锁记录。进程/FD 权限拒读和离散采样仍限制现场观察，端点快照不能排除瞬态活动。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-bert-performance-ab-r1-independent-failure-terminal-r1/terminal-audit.json`，SHA-256 `a970ea0741db18b1c2d48793959a1fde389f98711030118cc4f0f5d5de28a8fa`。


## 2026-10-30 隔离 AdamW 候选：固定 BERT 梯度单步对照

在真实 CUDA0 上，将此前 BERT base B8/S256 FP32、nonfused 且 foreach=false 的 r3 基线保存的同一组 197 个梯度分别注入隔离基线和候选 overlay。逐名核实 199 个参数槽及 197 个梯度对象在 `adam_update` 入口被实际消费，另外两个 pooler 参数没有梯度；两端各执行一次 optimizer 更新。独立逐数组审计确认 197 个梯度、199 个更新后参数、197 组一阶／二阶 moment 的 FP32 位模式精确一致且有限，step 均为 1；197 个参数实际变化。候选 helper 调用 197 次、复用命中 196 次，实测 CUDA dispatch，fallback 为 0。

此对照固定使用已有梯度，未执行 forward/backward、PyTorch optimizer 状态对照或性能测量，不能证明训练连续性、长期数值一致或收益；候选尚未合入主树，mixed device、AMP 等未覆盖范围不因本结果消除。此前 r3 单步严格 FAIL 原件保留，该实验不覆盖或改写其结论。独立审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-injected-r2-independent-numpy-audit-r1/audit.json`，SHA-256 `1c1108aea2ca6591305a9859ded98d0231106b852c1ba75051aecb45050edc66`。


## 2026-10-30 隔离 AdamW correction 候选 BERT 单步与严格差异

隔离的双模块 Python overlay 在真实 CUDA 上分别执行 BERT base B8/S256 FP32、eager、nonfused 且 foreach=false 的基线和候选各一次更新；候选 helper 实际调用 197 次、命中 196 次，命中记录核对了 CUDA0 dispatch、FP32 frontend 及 AMP=0。两侧保存的 197 个梯度、199 个更新后参数和 197 组 moment 均有限；197 个参数实际变化，199 个更新后参数的 FP32 位模式全部相同。

严格 comparator 仍为 FAIL：197 个梯度中 196 个及 197 组 moment 中 196 组位模式一致，唯一差异来自 `embeddings.token_type_embeddings.weight` 梯度及对应 m/v。梯度在 optimizer.step 之前已经不同：711/1536 个位模式不同，最大绝对差 `8.881784197001252e-16`、相对 L2 约 `3.8524e-7`；m 最大绝对差约 `9.0206e-17`，v 约 `8.3304e-28`。这是单步且更新后参数完全相同的直接证据；差异的具体机制与长期训练影响仍未知，不能因这一次而宣称长期一致或性能收益。该实验未执行新的 PyTorch 梯度/optimizer 状态对照，包含大量同步和读回，也不构成计时结果。独立逐数组审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-scalar-r3-independent-numpy-audit-r1/audit.json`，SHA-256 `0ae999b3e8c480f896dc65fa95fc6c131821644aeac41b101ee8044a17cf3ca2`。

此前 r2 在权重加载、forward 及 step 前因配置观察器直接比较内存字典与历史 JSON 字典而 FAIL；失败时保存的 JSON 配置与历史参考相同，本地源码及模拟表明 JSON 的整数键转字符串足以解释误判，但原进程内存类型未保存，不能认定当次唯一原因。r2 和 r3 两个原严格失败均保留，r3 使用全字段严格规范化及类型记录；候选尚未合入主树，未证明默认 fused/foreach 命中、完整训练或性能改善。

## 2026-10-30 隔离 AdamW correction 标量复用候选 CUDA 小型验证

尚未合入主树的两个 Python 模块 overlay 在 df45 真实 CUDA core 上，使用单 GPU、五参数、FP32、nonfused 且 foreach=false 的合成 fixture；PyTorch、Jittor 基线和候选分别完成 full、checkpoint 新进程 resume 及无逐步显式同步的 async，共 9 个阶段。基线与候选按 FP32 位模式精确对照通过，与 PyTorch 的对照在 `atol=2e-6, rtol=2e-6` 下通过；候选 helper 在 full/resume/async 分别记录 18/10/18 次调用与 8/4/8 次实际复用命中。核验了真实目标 GPU UUID、参数和 moment 的 CUDA dispatch，Jittor fallback 为 0；恢复 step 核对的是数值，不能据此宣称 PyTorch step 标量本身位于 GPU。

async 脚本没有逐步显式同步或读回，在最终一次显式同步后检查四步 loss；底层是否发生内部同步及其完整异步生命周期未证明。该小型实验不覆盖 mixed device、AMP/BF16、图捕获、BERT 实际命中或性能收益。终态功能缓存新增 68 项、无删除、既有项仅操作锁文本变化，独立性能缓存与封存 SDPA 缓存端点不变，锁与可见本任务进程/缓存 FD 为空；权限拒读与端点观察限制仍在。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-cuda-r2-independent-terminal-r1/terminal-audit.json`，SHA-256 `6c832ca93c7639c584371fc1e28f447b16db0cc68802df56a6e1d0b524ec640f`。

## 2026-10-30 隔离 AdamW bias correction 标量复用候选 CPU 证据

隔离候选在当前 df45 工作树上以两个 Python 模块 overlay 验证，尚未合入主树。r4 小型 CPU fixture 分别运行真实 PyTorch、Jittor 基线和候选各 4 步，并从第 2 步 checkpoint 恢复后再执行 2 步；候选标量 correction 记录 28 次调用、12 次实际复用命中。独立复核中候选与基线的 FP32 位模式一致；与 PyTorch 的对照在 `atol=2e-6, rtol=2e-6` 下通过，观测最大绝对差为 `6.103515625e-5`，差值不能单独代替相对/绝对组合阈值判断。r4 的 comparison 进程自然 rc0 且结果文件 PASS，但控制器未在活体扫描中观测到短命进程身份，原整组终态仍为 FAIL 并保留，不能回写为通过。

独立后继 r5 仅执行原本未开始的三个 CPU focused 节点：连续与 checkpoint 恢复对照、低精度不缓存、native 与 AMP 不缓存；三个节点的 setup/call/teardown 共 9 阶段全部通过、0 skip，systemd 单元自然完成。r4 已有轨迹未重跑。审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-r4-independent-partial-terminal-r1/audit.json`，SHA-256 `f174f60be5d79f1b31caf7b3d0eb05709ccd6c552b5f3b6a4c622fef049c3dab`；r5 独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-adamw-focused-r5-independent-terminal-r1/audit.json`，SHA-256 `910b7ec1ff9e71c9abe721e97db3c8a11e23fb77810a54e12879ee96619adb8c`。这是隔离小样本 CPU 证据；CUDA、多设备、异步图生命周期、真实 BERT 是否命中以及性能收益均未验证。

## 2026-10-30 BERT B224/S512 近容量单步校准保护停止

在实际 81920 MiB GPU 上，BERT base FP32、B224/S512、合成输入及 AdamW 的一次有界校准首先运行 PyTorch；单步期间 NVML 采样最低可用显存为 2326.75 MiB，目标进程采样峰值为 78700 MiB，低于自设的 6144 MiB 全局余量后控制器仅向本任务 worker 发送 SIGTERM，进程 rc -15，Jittor 阶段未启动。200 ms 离散采样及停止延迟不能保证余量始终不低于阈值，也不能将进程采样峰值当作 allocator live 高水位。原运行没有 result 或 allocator sidecar，不能确认更新完成、实际 allocator 峰值，也未记录 OOM 异常；不能称为 OOM 或宣称该尺寸的双端完整能力。

自设 80% allocator live 与 6 GiB 余量属于本次单步校准，不是原正式门槛；完整 near-capacity 三轮性能和显存验收仍未完成。失败原件保留，不自动换尺寸或重试。systemd 单元 failed/exit-code 1，cgroup 消失；源码、原生产品、私有性能缓存与两受保护缓存端点快照不变，相关锁和可见任务进程/缓存 FD 为空，权限拒读进程的状态仍未知。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-capacity-r4-independent-terminal-r1/terminal-audit.json`，SHA-256 `ef52b815d966addf6b30734a654ff287a5c928ceffe8fb710abc3bebd0ad1732`；另一独立审计 `/data6/cgq/jittor-lab/_state/release-day23/20261030-bert-capacity-r4-independent-failure-audit-r1/audit.json`，SHA-256 `81b8f566cb0a1199c13abf40f6136a97865f47082398ba03555d7e76b955f143`。

## 2026-10-30 df45 BERT 代表格三轮性能证据

基于 df45 工作树及独立 CUDA 性能缓存，r18 完成 BERT base、B8/S256、FP32、固定合成 token/mask、实际 AdamW 的三轮交替配对；每端每轮固定 16 次 warmup、10 个同步热样本，另做独立的 3 次 memory 与 3 次 CUDA attribution 更新。六阶段均有 33 次更新后的有限 loss 记录；加载时逐项核对 199 个参数和 2 个 buffer，197 个梯度有限且位于实际 CUDA，fallback 为 0。该配置不是容量边界或真实下游任务质量实验。

三轮共 30 个热样本的 pooled 中位耗时：PyTorch 118.458 ms，Jittor 155.516 ms，JT/PT 为 1.31283，超过原 1.10 的时延诊断阈值；逐轮中位数比为 1.4556、1.2137、1.2680。pooled p95 分别约 130.296/166.738 ms，标准差约 13.350/19.875 ms。分离测量的 allocator live 峰值为 PT 3,343,963,136 字节、JT 3,613,674,496 字节，比值 1.08066，低于 1.10 的该代表格诊断阈值；其为各框架所报告的 allocator 指标，不能当作总设备显存或整个发布验收。采样的进程 NVML 峰值分别为 3996/4544 MiB。

这项新协议明确使用 eager、固定 math 设置和 16 次 warmup，memory/attribution 与热计时分离，与历史固定 4 次预热等协议并非同口径；加载和首步含观测开销，热样本间写记录可能影响采样。五次稳定入场与阶段 NVML 为离散观察，无法排除全部瞬态负载或权限不可见进程；未证明厂商 kernel 内部算术精度。原 Day4–5 全范围性能与 near-capacity 显存验收仍未完成。旧 r7、r11、r16 失败及其原因不确定性均保留，不回写成通过。

六阶段及 controller 自然 rc0；独立 systemd 单元终态成功且 cgroup 消失。执行前后功能缓存与已封存 SDPA 缓存快照一致；私有性能缓存既有 math probe 两个文件仅 mtime 变化且内容、身份与大小经核验，操作锁文本变化，终态锁和可见任务进程、缓存 FD 均空；端点快照不证明期间没有瞬时写入。独立终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-r18-independent-terminal-r1/terminal-audit.json`，SHA-256 `08d7958485ff53880819f91418125b44a6e503354c82c9dcd084bbb5c9fb18e8`。

## 2026-10-30：df45 BERT 性能 r16 单对诊断及失败终态

r16 的固定 B8/S256 FP32 配置仅完成一对 PyTorch/Jittor：10 个热样本中位数分别为 96.51/154.35 ms，JT/PT 为 1.599；独立三步 memory 阶段分别记录 PT allocator peak 3,343,963,136 B 和 JT pool peak 3,613,674,496 B，比值 1.081，两端分配器指标口径存在差异。两阶段各自 rc0，配对条件检查通过；该结果仅作单对诊断，未完成正式三轮验收，也不是近容量结果。逐步有限性、设备、fallback、采样和数学条件的具体证据及限制见独立审计。

控制器终态为失败：性能缓存前后快照无新增或删除，两个既有探针 tmp 源文件内容哈希与大小不变、mtime 变化，既有锁文件内容与 mtime 变化，保护缓存端点快照不变；端点快照不能排除瞬态写入。控制器在 finally 中用缓存元数据失败覆盖先发异常，故未继续到第二对的直接原因未知，不能推断为特定负载或编译问题。锁及进程的陈述以各次实际审计为准，权限或采样不可见范围不外推。旧 r11 失败和本次失败均保留，未自动重试。独立审计：`/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-r16-pair-independent-audit-r1/terminal-audit.json`，SHA-256 `ed6b0f377c9c9de05b62c518f3447fdc9a91dac003200d42a14b4b55bf0fb4db`。


## 2026-10-30 df45 BERT B8/S256 配对性能 r11 失败保留

在工作树源码指纹 df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3、执行时 HEAD e92681d291272db1f1b9d0e3fe18cf35775a4de2 下，r11 的 PT0 阶段完成并保存 10 个热态样本；JT0 的首次更新及 4 次 warmup 按控制流返回后，合并的 JIT 缓存或 allocator 稳定性断言失败。该次未保存 warmup 各次缓存差异与 allocator 值，无法判断是哪个条件触发，也不能声称已测出具体不稳定原因；JT 热态样本数为 0，没有完成 PT/JT 配对性能比值。失败保留，不自动重试。

独立终审确认独立性能缓存端点新增 277 项、没有删除，既有项仅锁文件文本变化；受保护缓存和绑定产品前后未变。锁终态无内核记录及可见持有者，但进程/FD 权限拒读与离散采样仍是边界，缓存端点对比也不能证明期间没有瞬时改写或确定各 warmup 的 JIT 活动。该结果不覆盖近容量格、完整性能验收或其他版本。独立终审：/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-r11-independent-failure-terminal-r1/terminal-audit.json，SHA-256 1f1541c3c850494b213acd9891c966e449fff823846547b87ff839e914284cba。

## 2026-10-30 df45 独立性能冷编与 BERT 代表格首阶段失败

为隔离旧功能缓存和已封存的 SDPA 性能缓存，基于 df45 工作树执行源码指纹 `df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3` 在新私有缓存真实冷编 CUDA：231 个编译单元，约 400.13 秒，自然退出成功；CUDA sentinel 实测数值 `[3, 5]`，实际 dispatch 为 CUDA 设备 0，fallback 前后为 0。旧功能缓存与 SDPA 封存缓存的冷编前后快照一致，不能由端点快照推断期间不存在瞬时写入。独立冷编终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-cold-independent-terminal-r1/terminal-audit.json`，SHA-256 `6fd06492e5eb57e7f07e48661c1e0c9f16d1ee8bf8041fda8bb78760d577224b`。该产物来自工作树，不等同于纯提交树产品。

BERT B8/S256 FP32 合成输入代表格 r7 的五次稳定采样 READY 后，首个 PyTorch 阶段在热态及 CUDA attribution 后因映射观察器遇到 deleted 映射而以 rc1 结束；未记录触发的映射路径，原失败原因的具体映射类型未知。代码独立查证显示观察器在筛选目标库之前对任何 deleted 路径报错，无关的 memfd 映射也可能触发；不能据此认定本次实际触发项属于该类。Jittor 阶段未启动，未产生可用双端配对比值，不构成 Day4–5 性能或显存验收通过；单次失败原件保留，不自动重试，现准备限定该观察缺陷的独立后继。失败终审 `/data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-performance-r7-independent-failure-terminal-r1/terminal-audit.json`，SHA-256 `6ebcd8885b8ca30079b88def29c10a4162e3fecef5a235d866b78f2a77bacb7e`。NVML 是离散采样，受权限和采样间隙限制。

## 2026-10-30 Qwen 历史 checkpoint51 在 df45 工作树的单步续训核验

历史 Qwen checkpoint 51 加载到 df45 工作树（执行源码指纹 df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3），仅执行一次更新至 step 52，不能称为完整训练或纯提交树运行。112 个 adapter 梯度在 CUDA 上有限、112 个 adapter 参数发生更新；两组学习率由 1.59375e-5 变为 1.5625e-5，fallback 前后均为 0。设备核验的 337 项仅包括 112 个 adapter 参数、224 个 Adam moment 和 1 个 buffer，不涵盖冻结的 base 模型。

两个真实训练 microbatch 的首层 q_proj 输入及权重为 FP32、输出为 BF16，autocast 开启且记录实际 CUDA dispatch；未证明 kernel 内部算术精度。恢复时对照 Python、NumPy、Torch CPU 和 CUDA 四种 RNG，以及样本 ID、输入和标签形状；没有逐字节比较 token。严格恢复比较仍有两个新增 optimizer group initial_lr 字段，不能称为严格相等。两个 microbatch 的平均 loss 为 1.3973119258880615，与历史 Jittor 对照相同，历史 PyTorch 为 1.3938316106796265；该单步没有新固定评估、predict、完整训练或性能结果，观测同步及物化也不支持性能推断。历史原 strict 失败与 Qwen predict 限制继续保留。

缓存端点快照记录新增 304 项、无删除、旧非锁项未变；操作性锁文件内容变化，锁 inode 保持且终态释放。端点快照不能证明期间没有瞬时写入，权限拒读的进程与 FD 仍属未知。第二独立终审纠正了第一版将操作性锁文本变化误算作普通缓存变动的 false 结论，原审计保留，未重跑。证据：独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-qwen-step52-independent-terminal-r1/terminal-audit.json（SHA-256 83600fd6dddb9b5693e0411bf20a00d65886510ac2beeb4afcfd41b7d5989e80）；第二独立终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-qwen-step52-independent-terminal-audit-r2/terminal-audit.json（SHA-256 b393b5448bb0dbe2d1319ac015cff824543a833a12a41ec6c2d56855f7142f15）；其前一版 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-qwen-step52-independent-terminal-audit-r1/terminal-audit.json（SHA-256 b08906628c4caf9e8567f7ae5ee9db54f0bbc674dd58059d2e6614fb6461e42c）；包内终审 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-qwen-step52-preparation-r1-hold/attempt-01/terminal-audit.json（SHA-256 f1c0a1ed23db4dd715681ed74e8258433fc30080b04bfe7be7009b1b3aecc6e2）。

## 2026-10-30 df45 源码范围与四卡 NCCL 默认传输复核

只读内容检查确认：当前工作树的 df45 执行清单 2815 项均与原清单内容一致；当时 HEAD 05630e82 对应有 79 个已跟踪路径不同、24 个清单路径未跟踪。此检查未重新盘点完整工作树、未重新构建或验证当前运行，因此工作树训练证据不能称作纯提交树同源。审计 /data6/cgq/jittor-lab/_state/release-day23/20261030-postcommit-worktree-source-recheck-r1/audit.json，SHA-256 72650d4a2c3e0f1bfb3772748d41254474353c67eae72ac788ef75042dcedffc。

历史默认四卡 NCCL Trainer 在 PyTorch 阶段超时，Jittor 阶段未启动；独立 PyTorch 原生四 rank all_reduce 默认配置同样在 60 秒内未返回，另一次禁用 P2P 后四 rank 通过并观察到 SHM 通道。证据支持有界传输配置诊断，未定位根因或唯一故障链路；不能据此断言默认 P2P 可用或归因 Jittor。历史四 rank tiny BERT LoRA full/恢复通过绑定 NCCL_P2P_DISABLE=1 的特定配置，不能外推默认配置；历史源码身份及其他边界不变。审计 /data6/cgq/jittor-lab/_state/release-day23/20261030-four-rank-nccl-default-readonly-audit-r1/review.json，SHA-256 06060640e972e1bdc4c9cad3c7442a11513ded34e66156d9c6334524a6bb3736。


## 2026-10-30 历史T5 BF16续训功能证据复核

历史候选两seed checkpoint50的model/optimizer/scheduler/scaler共2088项加载签名在新进程on_train_begin比对通过；续训50→100各50更新、0 skip、99 microbatch样本ID/shape与各自full后半一致、50组LR一致，梯度missing/nonfinite均0、参数cuda:0/fallback0。第二epoch记录loss均值 full/resume：821 `1.48445840/1.48436376`，823 `1.49720019/1.49706295`；固定eval loss full→resume分别 `3.06211638→3.06272078`、`3.05675101→3.05602074`，exact match均0。仅支持历史该配置的有界功能续训和验证loss接近，不能断言任务生成质量或当前main适用。

两seed原strict各771/2088 FAIL，均为257模型+257一阶+257二阶moment；最终257 FP32模型tensor均可读有限，full/resume模型全局相对L2分别 `8.8914e-7/8.8376e-7`、max abs `0.00032434/0.00022142`，moment数值差幅未知。加载签名未包含RNG，原有RNG文件存在不等于四类实际恢复匹配，moment加载后的真实设备也未逐项核验。两seed的JT/PT full checkpoint50所需模型/optimizer/scheduler/RNG/签名文件此处实际存在，resume目录引用full的checkpoint不是缺失；历史其他22文件缺口本次未复核。只读终审 `_state/release-day23/20261030-t5-historical-resume-readonly-audit-r1/audit.json` SHA `2b46c2674051ee643897fd20000fba6ddeac12d0959d04f1792958428375164c`；原strict与预算 `51.29068389162421s` 均保留。


## 2026-10-30 Vision r25 五格历史实用训练与恢复证据审阅

五格历史r25实际训练均完成100更新、两epoch样本加权训练loss下降：CNN FP32 seed821 JT `2.312916→2.217066`、seed823 `2.322480→2.221158`，CNN BF16 821 `2.313635→2.218671`，ViT FP32 821 `2.299955→2.261349`，ViT BF16 821 `2.300069→2.261586`。固定65样本eval JT/PT loss与accuracy依次：CNN FP32-821 `2.190925/2.190382`, `9/65 vs 11/65`；CNN FP32-823 `2.184424/2.183609`, `7/65 vs 10/65`；CNN BF16-821 `2.189611/2.191783`, `7/65 vs 10/65`；ViT FP32 `2.211886/2.211886`, `21/65 vs 21/65`；ViT BF16 `2.211897/2.211983`, `21/65 vs 21/65`。小验证集CNN accuracy差异如实保留，不能由loss接近推断无差异或外推收敛。

五格resume 62–100样本ID/输入摘要/LR与各自full均39/39匹配，PT/JT full样本/输入100/100一致、fallback0；代码加载optimizer/scheduler与四类RNG，但未有加载前后optimizer状态及RNG独立对拍，恢复功能范围因此有限。两格CNN FP32同运行时strict exact通过、PT严格仍分别FAIL160/177；ViT FP32/CNN BF16/ViT BF16同运行时exact分别FAIL100/245/143，后续跨框架strict未运行，不能称全部门禁通过。只读原报告/NPZ及hash审计 `_state/release-day23/20261030-vision-r25-five-cell-offline-audit-r1/five-cell-summary.json` SHA `16c3ec54fafae6b0e896c7743aad4220edd6ccfce7cd0c115c1f0ce9dc6813f3`；绑定历史r25 source `1be46e54…`，非当前工作树或提交树，原310.095694191秒诊断预算未动。


## 2026-10-30 任务范围中文提交及保留的工作树

已提交 `cf5be28b618b6d99c89af8856d67dab618a053f3`（修复 Torch 兼容训练与恢复），parent `3dedae001372b6be106ae6700bcf10ebc7faebac`，tree `657d6e1da328fd064aba6735be29a0e8f2eb76d0`；仅由独立审阅的59路径任务patch暂存，cached diff --check通过，未push。提交前备份全部144个dirty路径与原空index；提交后逐项验证除3文档有审定的可逆摘要插入外，其余文件内容/模式不变，index为空，保留的dirty相对新HEAD仍存在；后审 `_state/release-day23/20261030-task-only-commit-postcheck-r1/postcheck.json`。候选代码与此前r12仅两处空白、types.py两空行和文档变更，结构单节点已验证前一同代码tree；工作树与候选22条路径不同，旧dirty相关依赖重合已披露，不能将working tree的df45源码身份自动转成提交树训练结果。更改HEAD后旧执行fingerprint包含旧HEAD，未来执行必须新绑定，不能把原缓存/历史证据重新贴标签。性能测量未取得稳定窗口、BERT续训strict差异原因及影响仍未完全解释，原两seed strict FAIL、T5/Vision预算与历史失败均保留；发布验收未完成。


## 2026-10-30 任务证据更新（历史记录按原身份保留）

HEAD-only r12 两卡实际训练与恢复使用 **r10 原生 core 和 Python 根目录，加 r12 两个 scheduler Python 模块 overlay**，并非完整 r12 冷构建。FP32 tiny BERT LoRA Trainer 两 rank 完成 full 4 次更新、checkpoint 2 新进程续训至 4；两卡真实 CUDA、fallback 0，有限梯度及参数更新被检查。JT/PT 两侧各 rank 平均 microbatch loss 分别为 0.34503534/0.34503534、0.34377843/0.34377844；固定 16 样本短程证据不能推及长程质量、AMP 或多于两卡。独立终审 fc2fc52657c9e6a31ef5ae9d8c2be617833018ffdd4ee24e3579ca599436fc17。

主树 df45 是另一源码身份：BERT seed821 配置 BF16 从头完成 full100，记录的参数和优化器状态为 FP32；训练前 initial evaluation 显式 autocast 下的 CUDA logits 为 BF16。projection dtype 和 autocast 状态按跨阶段集合记录，未标记训练时段，因此训练期间 BF16 覆盖未独立确认；历史 PT 报告同样未分阶段。训练结束后生命周期保存/重载对照使用 keep_fp32_wrapper=False 移除 forward 包装；两 epoch 平均 loss JT 0.354515→0.208091、独立历史 PT 0.354172→0.196566，固定验证 loss/accuracy JT 0.477709/0.787879、PT 0.483606/0.787879。新进程从同源 checkpoint 50 恢复至 100，恢复时 603 命名 CUDA 张量、RNG 和状态通过检查，后续 50 次更新有限、fallback 0；验证 0.478523/0.787879。最终 201 参数和各 201 一阶/二阶 moment 的严格哈希共 603 项不同，201 step 相同；相对全局 L2 分别约 4.46e-5、0.0505、0.00580，原因未定。原 strict 失败和原两 seed 15.425953905% 对 15% 门槛的失败均保留。

补充的独立训练阶段诊断采用同源 df45 与历史 PT 结果：两侧各两个 training_step microbatch 的首层 attention query projection 均记录 CUDA autocast 开启、FP32 输入及权重、BF16 输出；JT 同时验证 native CUDA dispatch 和 fallback 0。两侧各 201 梯度有限且参数更新；首个 optimizer step 前的零学习率人工改为 6e-6 后执行一次更新。该诊断未覆盖原 full100 的每一步或其他 projection，未追踪内核内部算术与累积精度，也不能作为性能基准。终审 SHA-256 c5cc3bb5a5c84d555753e0ee22d9f1e564f526274a29cf0d35bc6b59ddb9d044，路径 /data6/cgq/jittor-lab/_state/release-day23/20261030-main-df45-bert-bf16-training-phase-jt-shim-successor-r1-hold/independent-terminal-audit.json。原 full100 跨阶段集合的观察边界不变。

同源 checkpoint100 同批单步诊断比较联合状态差异：两侧都真实 CUDA、201 梯度有限及参数更新、fallback 0；重置 seed、将 checkpoint 的零学习率改为 3e-5 且不推进 scheduler，更新向量相对 L2 差约 0.0861、cosine 0.9963。参数、moments 与梯度起点均不同，不能归因单一 optimizer 或视作自然续训；其对长期训练影响未知。

隔离性能 cache 的 SDPA 合成同形输入 A/B 正确性通过（输出及梯度对照、11 个边界节点），并非整模型性能。唯一 60 秒配对 readiness 全 13 次采样未达到空闲显存、利用率及无外来进程门槛，未启动计时；无加速或性能验收结论。Day4–5 性能/显存仍未完成。原性能预算与历史失败保留。

本候选结构节点在 3068da63… tree 上实际 1 项通过、无跳过：原测试方法读取完整候选源码，使用桩处理无关框架导入；不是完整结构套件。后续候选只修改三份文档，代码 blob 相同，训练及测试仍按各自原混合或主树身份解释。旧 HEAD 802/800 结构失败、旧执行错误和用户已有 dirty 均保留。

### 可复核终审

- r12 两 rank 模型及恢复：/data6/cgq/jittor-lab/_state/release-day23/20261030-headonly-r12-bert-overlay-alias-successor-r1-hold/model-independent-terminal-audit.json，SHA-256 fc2fc52657c9e6a31ef5ae9d8c2be617833018ffdd4ee24e3579ca599436fc17。
- 主树 df45 full100、50→100 续训、同批单步：分别为 20261029-main-df45-bert-bf16-learning-trend-plan-r1-hold/independent-terminal-audit.json（96f990e7895121b37db7e5b15b728c97839a88883517203e0f129bb1f4821265）、20261029-main-df45-bert-bf16-freshresume-plan-r1-hold/independent-terminal-audit.json（f3edc6ba4b820d0d163d70b7f99b37552188211fafaac64c6961e1cc32a2729f）、20261029-main-df45-bert-checkpoint100-matched-batch-update-plan-r1-hold/independent-terminal-audit.json（ed9df0163d615db6d084329cb1dc1033af9ff1dc7aa5a8918d6b83a29495a456）；相对路径均位于 /data6/cgq/jittor-lab/_state/release-day23/。
- SDPA 正确性终审 20261029-sdpa-r6-shared-cache-correctness-terminal-audit-r1/terminal-audit.json（eecbd81183e3487871ee7e769813f03d8e4de46d7100b4cc0940f021ed40b5e1）；readiness timeout 终审 20261029-sdpa-r6-readiness-timeout-terminal-audit-r1/terminal-audit.json（2c64f1686c65d14dd72f2002b1d83da07a03ab4c4e7c85b30447fb0cff23ad3b）；同上级目录。
- 单节点结构终审 20261030-headonly-r12-structure-whitespace-terminal-audit-r1/terminal-audit.json（43d7e58e63ac3eee1e8cbdaaa63d1a1160622a32ba465db79d11b0b6f0798108）；同上级目录。


## 2026-10-30 df45 BERT真实训练阶段BF16补充验证

针对完整full100历史observer跨阶段混合set的问题，另做相同配置、真实Trainer训练阶段标记的两microbatch单步：PT/JT首层query projection两次输入及weight均FP32、输出BF16，CUDA autocast=true；JT三类张量native dispatch均cuda:0，指定物理GPU、fallback0。两侧样本及输入/标签hash相同，201梯度有限及201参数实际变化；两侧均将原首步零LR人工设 `6e-6`，观察sync与RNG历史差异明确保留。PT scaled losses `[0.4248046875,0.3701171875]`、JT `[0.427276611328125,0.34375]`，跨框架随机流未证明一致，不能声称loss逐步对齐。独立终审 `_state/release-day23/20261030-main-df45-bert-bf16-training-phase-jt-shim-successor-r1-hold/independent-terminal-audit.json` SHA `c5cc3bb5a5c84d555753e0ee22d9f1e564f526274a29cf0d35bc6b59ddb9d044`；首个observer读取shim不存在的 `torch.__file__` 造成输出未保存的失败原件保留。只证明两个训练microbatch首层投影的输出精度，不能外推原full100每步/每层或kernel内部计算及累积精度；以下对full100跨阶段observer的范围更正继续有效。


## 2026-10-30 df45 BERT训练精度记录范围更正

上文df45从头full100与同源续训均使用 `mixed_precision=bf16` 配置，真实CUDA、FP32 master参数/optimizer状态和有限梯度及记录loss/验证指标证据成立；但投影dtype/autocast观察器在训练前initial eval注册，只保存贯穿训练、评估与lifecycle的集合而无阶段标记。initial eval明确在accelerator.autocast内、CUDA logits为BF16；该汇总不足以证明训练阶段BF16覆盖，亦不能断言训练全程FP32。Accelerate源码有native_amp forward包装路径，但该运行未按训练阶段独立证实。`keep_fp32_wrapper=False` 移除forward包装仅发生训练结束后的lifecycle保存/重载对照，不应写成训练期已移除。历史PT报告的投影汇总亦不区分阶段，因此先前数值比较为同训练配置下的描述性对照，不能凭现有集合声称训练期两端混合精度覆盖相同。已准备补充阶段标记的有界单步验证；原产物、loss、严格失败和验收边界均不改写。


## 2026-10-29 SDPA隔离A/B正确性通过，稳定窗口未取得

本次新用途使用df45源码真实独立performance JITTOR_HOME冷编core，与功能cache隔离；CPU／CUDA原冷编路径及初始两次controller/驱动入场失败均保留，成功cold `_state/release-day23/20261029-sdpa-r5-independent-cold-terminal-audit-r1/terminal-audit.json` SHA `522b1bb2502073af1fae872d44cd0972dae655f25e3926ebbc8eae81a32862e0`。原A/B尝试因导入后修改sealed cache_path而失败，改为共享同一独立performance JIT cache后真实验证A原模块/B独立shuffle模块：物理seq-major合成输入，float64 output/dQ/dK/dV参考及三次重复通过，A/B max差 `0/1.19e-7/0/0`，B侧11节点无skip，实际源码/so/key不同且marker相符，双方CUDA/fallback0；终审 `_state/release-day23/20261029-sdpa-r6-shared-cache-correctness-terminal-audit-r1/terminal-audit.json` SHA `eecbd81183e3487871ee7e769813f03d8e4de46d7100b4cc0940f021ed40b5e1`。driver cache环境由controller设置但worker未独立记录，不能称实际路径已观测。

60秒13采样单次readiness因目标GPU利用率92–100%、每次有3个外来PID而未达稳定窗口，计时未启动、无速度结论、不再自动排队；终审 `_state/release-day23/20261029-sdpa-r6-readiness-timeout-terminal-audit-r1/terminal-audit.json` SHA `2c64f1686c65d14dd72f2002b1d83da07a03ab4c4e7c85b30447fb0cff23ad3b`。历史BERT性能ratio1.735仍是既有测量及其共享负载边界，不能由本合成正确性测试证明优化收益或根因；旧诊断预算未重置。


## 2026-10-29 df45 full与resume最终状态同批单步实际诊断

针对checkpoint100 603项strict hash差异，离线静态bias-corrected m/(sqrt(v)+eps)方向续训相对full整体L2差10.096%、cos0.994923；保存lr实际为0，因此该方向不是下一步真实更新。一阶moment相对L2差5.052%、二阶0.5799%及模型4.4625e-5继续保留。

独立同批压力诊断分别在新进程加载两份同源checkpoint100，固定原训练两个microbatch `[14,136]`/`[171,66]`、BF16/GA2/clip1，人工将两组lr设为3e-5并重设seed821，分别真实AdamW step101；这是额外干预，非原自然schedule。两侧checkpoint恢复签名/603 CUDA状态、201有限FP32 CUDA梯度及201实际变化参数核验、fallback0；两microbatch scaled loss均 `0.076171875/0.150146484375`，裁剪前梯度相对L2差 `1.943%`，裁剪后 `1.838%`，实际参数delta相对L2差 `8.6095%`、cos `0.996302`；更新后模型相对norm差由 `4.4625e-5` 增至 `5.0940e-5`，一阶moment相对差由 `5.052%` 至 `4.461%`。两侧起点模型、moment和梯度都不同，此为联合效应，不能单归因或认定长期无影响。独立终审 `_state/release-day23/20261029-main-df45-bert-checkpoint100-matched-batch-update-plan-r1-hold/independent-terminal-audit.json` SHA `ed9df0163d615db6d084329cb1dc1033af9ff1dc7aa5a8918d6b83a29495a456`。原strict 603项差异、原两次离线解析失败及双seed strict FAIL均保留；实际full/resume固定验证均26/33且loss接近，分别作为独立功能/任务表现证据。


## 2026-10-29 统一main df45 BERT BF16同源checkpoint续训

主树 `df45a565…` 从本源码full50的真实checkpoint新进程续训至100：checkpoint50加载签名、603 CUDA模型/moment状态、201 step及Python/NumPy/CPU/CUDA四类RNG一致；50续训更新0 skip、每步201 FP32 CUDA梯度有限、首个续训更新201参数变化、fallback0。99个microbatch样本/形状及50步LR与同源full和历史PT一致；第二epoch更新窗口等权均值resume `0.20797455`、full `0.20809052`、PT `0.19656555`，固定验证loss依次 `0.47852302`、`0.47770923`、`0.48360559`，准确率均26/33；功能恢复在此配置通过。独立终审 `_state/release-day23/20261029-main-df45-bert-bf16-freshresume-plan-r1-hold/independent-terminal-audit.json` SHA `f3edc6ba4b820d0d163d70b7f99b37552188211fafaac64c6961e1cc32a2729f`。

最终checkpoint100严格状态603项hash不同（201模型、201一阶、201二阶moment，无缺失/新增，201 step相同）；离线可信本地产物数值分析确认dtype/shape相同且有限，其整体相对L2分别 `4.46e-5`、`5.052%`、`0.5799%`，full/resume单步loss最大差 `0.00354004`，差异原因与实际影响未定，不能以验证接近自动称无害或最终状态严格一致。数值后继 `_state/release-day23/20261029-main-df45-bert-bf16-freshresume-plan-r1-hold/checkpoint100-numeric-comparison-successor-r2.json` SHA `f14388795d5fe399d1880306709190225ea54922775d466eaccb8ce9f92fbd5a`；前两次离线比较因weights_only opcode和portable包装误解析失败均保留。此为单seed、sync/物化诊断干预，原双seed strict FAIL、非性能结论及EventQueue未直接观测保留。


## 2026-10-29 统一main df45 BERT BF16从头两epoch学习趋势

`df45a565…` 在真实CUDA单卡以固定seed821、197训练/33验证、B2/GA2从头完成两epoch/100次更新、0 skip；每步201个FP32梯度有限且实际CUDA，step2非零LR后201参数均变化，1/50/100步603模型与moment状态CUDA，fallback0。198 microbatch样本ID/形状及100步LR与已审历史独立PT一致；按每epoch 50更新窗口等权的mean_microbatch_loss口径，JT `0.35451462→0.20809052`、PT `0.35417236→0.19656555`；最终固定验证JT loss `0.47770923`、PT `0.48360559`，双方accuracy `26/33`，JT选checkpoint100。另一按99 microbatch数加权的epoch1记录均值约0.347694，两种口径不得混用。LambdaLR helper last_epoch=-1实际调用1次、step101次，serialization调用 jt.fetch 4次并返回；native EventQueue未直接观测。checkpoint50/100模型/optimizer/scheduler/RNG文件与签名已核，但未独立解码或新进程续训；梯度/参数检查有sync与物化干预。独立终审 `_state/release-day23/20261029-main-df45-bert-bf16-learning-trend-plan-r1-hold/independent-terminal-audit.json` SHA `96f990e7895121b37db7e5b15b728c97839a88883517203e0f129bb1f4821265`。单seed实际学习趋势与任务表现支持此配置的实用训练，但不覆盖新源码optimizer-only重建分支的模型验证、新进程续训、其他网络或原双seed 15.42595%>15% strict FAIL，后者保留；共享负载及诊断干预下不作性能结论。


## 2026-10-29 统一main df45 BERT BF16两epoch实训运行中

固定seed821、197训练/33验证、B2/GA2、两epoch 100更新的全新单卡BERT BF16训练用于验证统一main真实学习趋势及新scheduler/fetch路径；对应历史PT权重/数据/超参及23项引用已核。controller PID 2031669、child PID/PGID 2031672/start_ticks 189609483，单次cap3000秒，运行目录 `_state/release-day23/20261029-main-df45-bert-bf16-learning-trend-plan-r1-hold/`，绑定 SHA `7225df80…`。逐步梯度设备/有限性、参数签名和关键步optimizer状态会同步/物化，属于明确诊断干预；Python wrapper记录LambdaLR/helper/jt.fetch调用，native EventQueue未直接观测。启动时GPU UUID `GPU-2be51…` 空闲约51292 MiB，外来负载/利用率100%，仅正确性/学习趋势，不作性能结论。尚无当前训练或验证结果，接续须核PID/start/receipt、锁和实际状态，不重复；历史双seed strict FAIL保留。

HEAD-only r12从r11仅加入持久CUDA启动回归测试，完整HEAD patch SHA `250c43dffca948b5e027ea20f65b0e4141ecffc54b039f410398f6f6b69718c5`，fingerprint `2394be191ac3fbe00a8dc304c623f65c97c25339cadf2f66785c3d58c50c8349`，隔离index对2796文件/链接blob/mode核对，主index不变；静态审计 `_state/release-day23/20261029-headonly-r12-startup-regression-candidate-r1-hold/candidate-audit.json` SHA `8f7453bac4006ff97e3db73346b4c0a076bf9b174f3e0dbcdefbec976f3b8594`。原会话前相关dirty片段重合按r9审计披露；r12未单独实测，候选未暂存或提交。


## 2026-10-29 r11候选CPU调度器及tiny BERT训练/恢复

HEAD-only r11相对r10仅LambdaLR实现/helper/测试三文件，r10真实CPU core + 两个r11 scheduler overlay及r10其余任务模块：八个PT oracle场景、16个调度器/Accelerate节点通过、fallback0，独立终审 `_state/release-day23/20261029-headonly-r11-lambda-cpu-focused-terminal-audit-r1/terminal-audit.json` SHA `efea8cd356d3aa8d73a25bbd0a8fb34d1769d68cd6f30b9d07dc4a2cf72f3f2e`。其后合成tiny BERT CPU、dropout0、八步：25参数每步实际变化、梯度有限、参数/50优化器状态CPU、fallback0；step4保存并新进程恢复到step8，连续与恢复最终25参数最大差0，PT/JT训练loss最大差 `5.96e-8`。固定验证集两端趋势同向变差：JT loss `0.696845→0.767809`、accuracy `0.25→0.125`，不能称收敛或任务表现改善。终审 `_state/release-day23/20261029-headonly-r11-tiny-bert-cpu-model-terminal-audit-r1/terminal-audit.json` SHA `9118537ee24255fe75c3756e4f2070446f129a38cf843c633c6f66decb890239`。仅短程CPU混合身份，未证明GPU、长训或RNG恢复。统一main df45 100步BERT BF16从头学习曲线实验已独立准备，正在入场审阅，未运行；r11候选与main不可混同。


## 2026-10-29 统一main df45 CUDA冷编及功能专项通过

主树 `df45a565…` 同源CUDA真实冷编231 TU、38既有资源、全部产品/锁终审 `_state/release-day23/20261029-main-unified-fetch-lambda-cuda-r1-hold/independent-terminal-audit.json` SHA `783aeffb80ca67ea63e2534e4653a1c1dd94638b7005412f2172871b7e756add`。main core ELF init fetch 88/device 198，故本次pre-import通过不能独立证明device-first；该危险顺序在r10候选已实测。真实CUDA功能六阶段均通过、无skip：preimport `use_cuda=1/use_nccl=1`；持久startup、原fetch/GIL/EventQueue 4/4；32回调/cleanup/异常退出；placement 61/61；CUDA restore 11/11；LambdaLR GPU探针含两次实际参数更新、moment CUDA/有限、optimizer-only重建和缺字段。全阶段CUDA dispatch/fallback 0→0，终审 `_state/release-day23/20261029-main-unified-cuda-focused-r1-hold/independent-terminal-audit.json` SHA `3585f23ab5330b3a39960dfe5590ecf7f02e55dc863218abaa1e79217b77276f`。GPU Lambda中间数字仅断言未单独保存；本次不含NCCL collective/多GPU/超过20同时排队。d900的两rank模型与checkpoint证据及旧CPU BF16 strict失败均保留各自身份，不能转写为df45模型通过；任务候选r11本身与main不等同，待独立回归及模型学习曲线。


## 2026-10-29 统一main df45 CPU构建、调度器及fetch通过

统一主树 `df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3` 独立CPU cold自然rc0，227 TU、38既有pyc资源、源码/产品/stamp与锁终审 `_state/release-day23/20261029-main-unified-fetch-lambda-cpu-r1-hold/independent-terminal-audit.json` SHA `99244535aa8a4b46785dd63c1f1becbfc50395d812e7083fba1a8c200d2ba860`。随后当前main真实LambdaLR新增6和原scheduler/Accelerate 10节点共16/16、optimizer-only重建8案例与已审PT oracle一致、fallback0；终审 `_state/release-day23/20261029-main-df45-lambda-cpu-focused-plan-r1-hold/independent-terminal-audit.json` SHA `fd32291e46de58ea99b1c0a06db29498ed443c113090405479ce68400d944470`。原d900数值缺陷、CPU BF16 strict失败未覆盖；重建是内存state_dict、非新进程模型恢复。

同源码CPU fetch后继3/3无skip：原fetch、32提交回调值、正常cleanup及预期异常退出子进程，CPU dispatch/fallback0；终审 `_state/release-day23/20261029-main-unified-cpu-fetch-successor-r1-hold/independent-terminal-audit.json` SHA `845fbf58e45b45f42639b8c05cf0decb2e02fe8de438367bb62284843bd598b6`。32提交不证明排队长度>20。正在以三份同源CPU审计为前置准备统一main独立CUDA cold及startup/EventQueue/fetch、optimizer placement/restore、LambdaLR GPU测试；没有当前main CUDA结论。


## 2026-10-29 LambdaLR与fetch队列修复已移入主树，统一验证待运行

经先前隔离CPU/PT八案例和原10回归通过后，LambdaLR修复已最小移入main：保留/设置各组initial_lr并在last_epoch恢复构造时按字段建base、缺字段按PT报错，其他scheduler不改；新增覆盖多组、tensor、optimizer重建和文件checkpoint的测试尚未运行。整合前目标lr_scheduler干净、旧dirty和index均已保存，`_state/release-day23/20261029-main-lambda-integration-r1/integration-review.json` SHA `8259d6fced9bc17730e3b4b1ea2a6e2f23b22d52f186d111a180af0783be5d30`。隔离overlay终审 SHA `6d4e2f342e6b610fc3a947b5618b1549683ce1de928db3e9a64d2a635aa657d3`，原d900学习率错误未覆盖。

随后已审r10 fetch惰性队列四文件最小补丁移入main：目标与候选字节一致，原 `var_holder.cc` 82行既有dirty及主index保留；整合审计 `_state/release-day23/20261029-main-r10-fetch-queue-integration-r1/integration-review.json` SHA `b15cfde8c384c7523ac45596a14a6db270d104cd4d72de3ac90d6fa2c3079842`。仓库新增持久子进程回归 `tests/runtime/test_flag_env_and_setter.py::TestUseCudaAtProcessStartup::test_cuda_enabled_before_jittor_import`，明确导入前设置CUDA、实测dispatch与fallback；CPU-only依据capability skip，实际测试尚未执行。新统一执行源码 `df45a56585f68874e8f1c64a80c9fff276bfe795e80c4d079cd06a8e7e528fc3`，manifest SHA `46243b6f25ecf6e897a4e5e05a306cc5d6f71b207fcd249bf8660eebb6cf443d`。候选r10 CPU/GPU功能与LambdaLR overlay证据不可自动转成统一main通过，正在准备新身份原生产物及定向验证；原r10 CPU strict SKIP失败、初始化signal11和BF16 strict差异均保留。


## 2026-10-29 fetch初始化崩溃隔离候选GPU专项通过

r10 HEAD-only独立候选以惰性fetch队列修复原pre-import `use_cuda=1` signal11；真实CUDA cold 231 TU，实际ELF仍维持原危险顺序device静态初始化索引100、fetch索引145。原环境 `use_cuda=1/use_nccl=1` 进程启动、真实CUDA core加载和 `sync_all` 成功，dispatch CUDA/fallback0；原CUDA fetch、callback GIL和CPU跳过的原EventQueue节点均实际通过无skip，32回调、正常cleanup与异常退出子进程通过。GPU独立终审 `_state/release-day23/20261029-headonly-r10-fetch-queue-candidate-r1-hold/cuda-cold-hold/gpu-functional-independent-terminal-audit.json` SHA `4b30b1babe55b273519b6d9f1f7e172e303788a8397a502db185627cd9f6d590`；CUDA cold后继终审 SHA `318adf9184b61dd2ee4fc5df47e2d7efbccb0562ea901fcd52f871ceba9a6cbb`，先前审计对CPU/CUDA符号可见性相同的错误及原signal11均保留。NCCL环境已启用但未执行collective；32提交未证明队列并发长度曾>20，外部ABI及析构边界仍如实记录。修复尚未移入主树，正在准备与dirty基线兼容的最小移植；调度器独立候选先运行。


## 2026-10-29 LambdaLR重建实测缺陷与fetch崩溃候选构建

主树 `d90073…` 的独立PT/JT CPU实测确认：只恢复optimizer、按 `last_epoch=2` 重建 `LambdaLR`，PT base/LR为 `0.003/0.00075`，JT为 `0.0015/0.000375`；自定义 `initial_lr=0.012` 时PT为 `0.012/0.003`，JT仍为 `0.0015/0.000375`；缺 `initial_lr` 时PT报KeyError、JT成功。完整optimizer+scheduler恢复两端step3均为0.00075，因此不能误称已验证的Trainer恢复失败。内存state_dict独立终审 `_state/release-day23/20261029-main-d900-cpu-scheduler-reconstruction-jt-successor-r1/independent-terminal-audit.json` SHA `66a08bca04c6dd8a53254c6b167bc2a595cbaeff7988d6558f8ad397cf52ff53`，fallback0；初次JT遗漏已审隔离stamp导致导入失败的原件保留。其他scheduler部分使用链式公式，正在针对LambdaLR隔离设计修复，不外推所有类。

r10独立候选将fetch两个队列改为惰性构造以处理 `use_cuda=1` 静态初始化崩溃，主树未改；独立CPU cold 227 TU真实通过 `_state/release-day23/20261029-headonly-r10-fetch-queue-candidate-r1-hold/cpu-cold-hold/independent-terminal-audit.json` SHA `90f6da11c0575ca518c7c7aafb2cbcd137a9b08717a0ff4ca37a1693c3f59930`。CPU fetch严格阶段3 PASS/1 CUDA-only SKIP、原门槛FAIL保留，派生审计仅判三个适用项通过、GPU必须执行原跳过项；`cpu-applicable-successor-audit.json` SHA `5af589cfaf7a023276a087a4e7b2c11eff9e850fa32509cfe0b24a3ac3bbc96d`。旧Linux core队列变量为本地符号、275缓存扩展无未定义引用，未知外部/Windows仍有ABI及无cleanup时队列不析构的生命周期边界。r10独立CUDA cold controller PID 1987842/start_ticks 189307494、worker 1987844/start_ticks 189307558、cap900，目录 `_state/release-day23/20261029-headonly-r10-fetch-queue-candidate-r1-hold/cuda-cold-hold/attempt-01-cuda`；正在编译，恢复先核实际PID/receipt/锁，不重复。原pre-import signal11及GPU功能尚未验证修复。


## 2026-10-29 候选r9两rank模型与恢复后继通过

HEAD-only候选的真实运行身份为r2 native/root + 覆盖9个r9生产Python差异：两rank、16样本tiny BERT LoRA FP32，full4/stop2/新进程resume4以及已审PT对照均rc0；六个rank进程不同、真实GPU/core/NCCL匹配、fallback前后0，每rank六个参数实际更新，候选scope/Torch AdamW step/CUDA fused更新真实调用已记录。每rank step2参数/optimizer/scheduler共40字段恢复最大差0，四类RNG与样本衔接；full/resume最终参数差诊断0。PT共同数值字段allclose（梯度max `1.49e-8`、参数 `1.041e-7`、optimizer `3.73e-9`），resume的四个仅PT状态字段保留；8/9 overlay在训练中实际加载，distribution未用，不能声称完整r9根目录或冷编。sync与Python profile是诊断干预。独立终审 `_state/release-day23/20261029-headonly-r9-bert-wrapper-successor-r1/independent-terminal-audit.json` SHA `791be9292968706fb1e7974c96df289218feeacf962c651b28c77bca0beeb585`。

原full4在首步后观察器把Accelerate包装器误判为候选类型而失败，日志未保存当次包装器类型；分析还发现fused路径不会调用per-param adam_update，不能以此断言未更新。该失败 `_state/release-day23/20261029-headonly-r9-bert-model-successor-r1/independent-failure-terminal-audit.json` SHA `7b431099b185723493dddded63b4822043ce39343d99fa5d7f2cb0a76ffda75c` 保留；后继实际观察包装链及真实fused更新。原use_cuda=1 core初始化signal11仍为未修复问题：只读静态分析显示候选ELF device静态初始化先于fetch_op全局list，崩溃指令读该list且fault地址0x10，与未构造容器相符；需进一步验证、审查另一fetcher_to_free的生命周期，不能将相关假设当成修复通过。


## 2026-10-29 候选r9模型后继运行中

候选safe-init成功导入后原finder绝对第0 guard失败（未到NCCL），原件 `_state/release-day23/20261029-headonly-r9-nccl-safe-init-diagnostic-r1/independent-terminal-audit.json` SHA `b9a6dcff223dccff57f0d8d49cce7c22711ed3632096cebc6b5460d870e9f03f`。独立新诊断验证finder唯一、前置11钩子对9候选模块共99次不抢占，8/9实际加载并核hash（另1未观察）；单进程NCCL实际warm成功/fallback0，审计 `_state/release-day23/20261029-headonly-r9-nccl-finder-diagnostic-r1/independent-terminal-audit.json` SHA `ebba681d9ededee2c4d31b42efb65380a7295ecbea4774a9bf5868695f264fed`。首次core signal11与finder失败都保留，不能以成功后继抹去。

混合r2 native/root + r9九生产Python overlay的独立两rank BERT LoRA full4→stop2→fresh resume4模型后继已启动，controller PID 1955525、full launcher 1955530、rank 1955532/1955533，目录 `_state/release-day23/20261029-headonly-r9-bert-model-successor-r1`；full launcher cap1200秒，失败即停不自动重试。实际两rank已在真实launcher导入时初始化NCCL，首步前各6参数/12状态槽检查完成；当前算子编译、尚无更新和训练结论。检查有显式sync与Python调用profile干预；接续先核PID/start、receipt、实际JIT/锁/资源和source/product，勿从本状态外推完整r9冷编。性能A/B保持HOLD、不得与任务JIT并发。


## 2026-10-29 提交候选r9 NCCL warm在core初始化崩溃

任务候选需独立模型与恢复证据，因其与d900主树有45个生产文件差异；候选模型真实身份为r2原生/根目录加覆盖全部9个生产差异的r9 Python overlay，不能称完整r9根目录冷编。首个单进程NCCL warm在r2 CUDA core动态加载、NCCL setup前signal11，模型未运行，原失败独立审计 `_state/release-day23/20261029-headonly-r9-bert-runtime-r1-hold/warm-failure-terminal-audit.json` SHA `6f0f552673a51ea2ec18ffa3edf34d5e62eeed1bbfdd1e1fc48f06b5474f5b3a`；core/utils/stamp及双源仍匹配，无可见任务进程或锁holder。失败前环境导入前强设 `use_cuda=1/use_nccl=1`，已通过151节点的focused按 `use_cuda=0/use_nccl=0` 导入，栈符号指向 `init_use_cuda → setter_use_cuda → sync_all`；环境仍有其他差异，不能认定唯一根因，signal11实际失败不能抹去。独立safe-init warm诊断正在准备；成功也不能自动证明两rank launcher可用。

SDPA backward独立A/B方案仅静态准备、未运行：`_state/release-day23/20261029-sdpa-f32-backward-shuffle-ab-r1-hold/` manifest SHA `784f6f85f23776f462663268d660868275fb00eca55784f09525caafdd951443`，不重置旧性能预算。候选训练与诊断优先、JIT串行，性能收益与缓存隔离均未实证。


## 2026-10-29 d900两rank BERT LoRA保存及新进程恢复通过

主树 `d90073…` 真实CUDA/NCCL与固定16样本tiny BERT LoRA FP32，已有full4基线后新做stop2与不同PID的fresh resume4；两阶段两rank fallback均前后0、6个可训练参数实际更新。每rank checkpoint step2共40个参数/optimizer/scheduler字段恢复最大差0、Python/NumPy/CPU/CUDA四类RNG匹配，样本游标衔接；同源码full与resume最终参数最大差0。共同梯度/参数/优化器/调度数值字段与此前审定PT对照allclose；resume轨迹PT另有两个 `param_groups.initial_lr`、`scheduler._get_lr_called_within_step`、`scheduler.verbose` 四字段，不能声称state_dict字段完全一致。首步前与stop首步后state显式sync属于诊断干预。独立终审 `_state/release-day23/20261029-main-d900-bert-resume-successor-r1-hold/independent-terminal-audit.json` SHA `9154144a522dfd39914f59e2a8cc089d1f7787cfa6657b8e630c1f2d97abbab1`。旧CPU BF16 strict FAIL与首次full4超时保留；不外推17样本尾批、AMP、四rank或性能。


## 2026-10-29 d900 CPU优化器专项

主树 `d90073…` 绑定已审CPU core、按源码真实mtime隔离派生的stamp并显式 `use_mkl=0`：scope/异常2/2通过，restore 10/11，唯一失败为原BF16 AdamW恢复后下一步moment strict比对，最大绝对差 `6.676465e-05`、最大相对差 `0.00093548`；旧阈值与FAIL未修改。两个阶段fallback0，实际锁及可见holder空；审计 `_state/release-day23/20261029-main-d900-cpu-optimizer-focused-r1-hold/independent-terminal-audit.json` SHA `e9b7f0394c0b41eb17e9f97a24bff4e11b7154e7365c8dda815e23bec82024f9`。该数值差异长期训练影响仍未知，不能仅因数值小认定无害；旧GPU恢复11/11和新两rank full4功能证据单列。


## 2026-10-29 主树新fresh optimizer真实两rank训练通过

主树 `d90073…`、真实新CUDA core/NCCL、两rank tiny BERT LoRA FP32、16样本的full4后继自然rc0，每rank 4更新/8 microbatch，全部6个可训练参数实际变化；两rank UUID、样本序列、有限梯度及fallback前后0核实。每rank首步前及首步后的12个AdamW状态经显式sync物化为对应CUDA上的FP32 Torch Tensor，scope恢复一致；与已审旧PyTorch full4固定对照的梯度/参数/优化器状态最大绝差分别 `1.49e-8`/`1.041e-7`/`3.73e-9`，均在诊断容差内，loss最大差 `2.98e-8`。独立终审 `_state/release-day23/20261029-main-d900-fresh-optimizer-bert-successor-r1/independent-terminal-audit.json` SHA `615b1453353082158e74fd6e23741031e9c65391b69ef7c5f68fd6a83f79fefb`。原首个full4冷算子JIT 270秒超时已单独保留，新用途独立预算后继利用已产生缓存，约48.6秒完成；本次状态sync是诊断干预，四步短程不能证明epoch学习曲线、d900续训/尾批/四rank/AMP/性能。旧源码BERT/Qwen/T5历史曲线和对应保存恢复证据保留其身份，不能重标。


## 2026-10-29 主树fresh AdamW真实BERT首步前检查通过，训练超时

主树 `d90073…` 新冷编CUDA缓存先单进程NCCL warm rc0；后继两rank 16样本tiny BERT LoRA FP32 full4运行，按270秒launcher上限在首步算子JIT阶段rc124，首步、梯度、参数更新及PT对照均未完成，不能据此判断训练可用或失败。两rank均实际初始化NCCL，各自6可训练参数与12状态槽在首步前经显式sync检查为Torch Tensor/FP32和正确CUDA设备、构造前后scope一致；fallback仅训练前0，sync为诊断干预。相对较早双卡后继前快照新增78个jit .so，其mtime晚于本模型启动（该旧基线亦早于后续warm，归因按时间界定）。进程组结束、锁与可见holder空，部分外部FD不可读。超时独立终审 `_state/release-day23/20261029-main-d900-fresh-optimizer-bert-plan-r1-hold/terminal-timeout-audit.json` SHA `13447e7a50e6d3c09ac2e7ba8f82dd1f3133b08d3371c0c9d45bfdd6181b2847`；正在评估以明确新用途和预算完成实际更新，原超时不覆盖。


## 2026-10-29 主树optimizer设备专项通过，模型验证待执行

主树 `d90073…` 绑定同源码真实CUDA冷编产品，单卡placement 61/61、原18例AdamW/SGD混合设备冲突矩阵state无错放、原FP16/BF16失败节点2/2、主树restore 11/11；双卡后继2/2，driver核对cuda:0 `GPU-2be51…`、cuda:1 `GPU-b6f598…`，功能阶段均fallback0。首次双卡被控制器残留单卡数量断言在收集前阻断，原失败保留；后继仅运行尚未收集的双卡节点。独立终审 `_state/release-day23/20261029-main-d900-cuda-placement-dual-successor-r1-hold/independent-terminal-audit.json` SHA `b2affb58a8d89a65a925fbd7f67a658688090c9f7e88277ea5eab35a9d4171ec`；初始四阶段审计 SHA `9f23352e9240079bc19115740e80d9eb3ddb31e4af8c2014874e9939d081e6c7`。旧d08 CPU 2/2不能标作新源码CPU通过；旧BF16 strict差异保留。显式参数状态创建的前端类型改变可能切换算术路径，实际模型训练及更新效果尚待验证。

已准备两rank tiny BERT LoRA FP32、fresh AdamW实际状态位置和训练更新的有界方案 `_state/release-day23/20261029-main-d900-fresh-optimizer-bert-plan-r1-hold/plan.json` SHA `b5b75f81c9e5f738a0ceecd451e632bb7e376825b137e2f4107a363b4579cf84`；逐参数显式同步状态属于诊断干预，不能冒称无干预曲线。现准备新产品/NCCL绑定和必要预热，尚无该身份训练结果。


## 2026-10-29 主树新CUDA冷编与候选打包通过

主树 `d90073…` 在新独立缓存单次真实CUDA cold自然rc0；独立审计确认231 TU（194源码/37生成）、817源码签名、core/utils/stamp与libcuda_extern实际产品及新缓存RUNPATH、导入后 `use_cuda=1`，锁与可见holder终态空；无张量功能测试。审计 `_state/release-day23/20261029-optimizer-state-cuda-cold-release-r5/independent-terminal-audit.json` SHA `7f9081857286dd547819a29f2dcd4cda6d8ed054ebe1a4992bd690cc16c157d7`。后续须绑定该实际产品运行设备矩阵、双卡及恢复，不能以构建通过替代。

候选r9隔离源码副本的root/compat两套wheel及sdist均实际构建rc0；新增state placement helper、storage header、optimizer restore和DDP hooks逐字节在包内，wheel全部RECORD路径、hash、大小通过。终审 `_state/release-day23/20261029-headonly-r9-packaging-isolated-r1/terminal-audit.json` SHA `a67ae633c3c22673c7b450739fdbb968f26236118486040c7c161ca70f5cd371`；未测安装、模型及从sdist重建wheel。任务候选中与暂停前相关dirty重合的三处stop_grad及九行iadd逻辑已单独记录，原始字节不齐的路径不能声称已穷尽证明没有旧dirty；审阅 `_state/release-day23/20261029-headonly-r9-commit-readonly-audit-r1/review.json` SHA `b5b3dbf0ee88d4aea22011917bd6237e466ce8d0ba8f310e29ba0342df614a97`，尚未提交或暂存。


## 2026-10-29 候选r9定向通过、主树真实CUDA冷编运行中

HEAD-only候选r9真实身份为r2 CUDA core + r9九个Python生产模块/测试overlay；单次九阶段151/151通过：placement单卡61、双卡2、optimizer restore14、alias22、write1、cat14、RMS2、pow1、factory34。逐阶段真实UUID/dispatch/fallback0、源码产品、锁和cache已审，factory绑定原PyTorch oracle；终审 `_state/release-day23/20261029-headonly-r9-cuda-targeted-overlay-terminal-audit-r1/terminal-audit.json` SHA `c65204378ab7d8f18dbbd71da754f38ccc0de74bdb7fea184ab9e7e69cd494d3`。修正测试辅助代码后的CPU frontend恢复2/2亦通过，原r8 3pass/2fail及历史BF16 strict FAIL保留；混合overlay不是完整r9冷编或真实模型训练。候选helper与主树 `d90073…` 逐字节相同，其他身份不同，不能直接转移主树结论。

主树独立真实CUDA cold已单次启动，controller PID 1916153、child PID/PGID 1916169/start_ticks 188769085，cap900、无自动重试；新缓存及产物路径 `_state/release-day23/20261029-optimizer-state-cuda-cold-release-r5/attempt-01-cuda/`。入场审阅 `_state/release-day23/20261029-optimizer-state-cuda-cold-release-r5/release-admission-review.json` SHA `2c46cc5b93652beeca2ea7e0f40921adcc55a1db45a183faaf6191042fa1c4e6`。接续时先核PID/start、receipt、锁、产品与源码；尚无主树新构建或设备回归通过结论，其他用户不可读FD不作绝对断言。


## 2026-10-29 optimizer add_group实际错设备及后继修复待验

HEAD-only r7 的 r2 CUDA core + r7九Python模块实际回归：原optimizer restore 14/14通过（含此前FP16/BF16两失败）；后续55节点24通过/31失败，双卡及后续6阶段未启动。独立终审 `_state/release-day23/20261029-headonly-r7-cuda-targeted-overlay-terminal-audit-r1/terminal-audit.json` SHA `c75ee23fff9874ee2a0fa7c7c6a423a2b6a1a4df165b46c8368e38b2c2727467`。其中26项源于测试工厂只设置Python ContextVar、原生参数实际为FollowRuntime；独立探针在真实CUDA下确认其余五种Torch optimizer的新增CPU参数组，其CPU参数实际CPU、而新增state实际CUDA（native Var backend=-1、dispatch CUDA、sync后location=device），非仅显示差异。探针 `_state/release-day23/20261029-r7-cuda-add-group-location-probe-r1/` receipt SHA `45e14b2b91f22217cf6832d2700b02fdc3e221346a27631e23378f422255aad3`；两层placement token嵌套异常恢复经探针验证。显式CPU native Var的 `.device` 可能显示runtime CUDA，需按backend/dispatch/实际location判断，Torch state另测公开device与类型。

主树新源码 `d900732c0398909488f9525be5a11d29371d538c7af8f9bed3dd9fe8dd72857f` 将显式参数的Python placement、native TLS、frontend result type同时覆盖并逆序恢复，FollowRuntime保留原路径，修正原生测试工厂并增加类型/公开state/异常测试。相应HEAD-only r8 candidate fingerprint `4067dead1617683710defaf44b8e9901e5859f4548a5f98d76bbc0a36cc99f9d`，helper与主树逐字节一致；两者实际回归尚未运行，frontend类型变化的数值影响待验证。主树独立CUDA冷编包 `_state/release-day23/20261029-optimizer-state-both-placement-genuine-cuda-cold-r4-hold/` 仍HOLD，待r8终态和串行资源释放后放行；旧stamp、旧错误和历史构建全部保留。


## 2026-10-29 optimizer placement CPU通过、CUDA扩展stamp阻断

主树 `d08f9e…` 使用独立派生core stamp、`use_mkl=0` 完成CPU scope/异常恢复2/2；派生仅更新两个内容已按SHA核实不变的C++文件的真实mtime，核心及完整ingredients运行时检查通过，非新冷编。pytest部署14个shim文件被保守缓存检查拦停，逐项比对均与受审源码一致，无JIT源码或二进制变化；CPU终审 `_state/release-day23/20261029-main-optimizer-placement-fix-regression-r3-hold/independent-terminal-audit.json` SHA `27af5f3e263392eb9abfc33343d3bbd8fc56d87e6e0a734d2cf53b3be6cd6a89`。后续GPU尚未收集测试，独立 `libcuda_extern` stamp含相同mtime差异、无core那样的隔离路径覆盖，原产品和stamp与冷编快照哈希一致但当前完整运行时ingredients未验证；终审 `_state/release-day23/20261029-main-optimizer-placement-fix-regression-r4-gpu-hold/independent-terminal-audit.json` SHA `a2e0564a4599e3acabfed57252313a497975410fe734636600372645e47f0db5`。原签名和各次失败均保留，正在评估可审计的独立缓存，不可将GPU回归写成通过。

FollowRuntime混用原生参数和torch optimizer存在ambient placement冲突的代码路径，但目前无实际故障复现；直接固定设备会改变动态语义，不能以此假设修复。只读审计 `_state/release-day23/20261029-d08-followruntime-optimizer-readonly-review-r1/audit.json` SHA `663a7daff5a5b7c4351089ee24374a2d9abcf1936472e406251d8537ca6a00af`。


## 2026-10-29 optimizer placement回归受stamp检查阻断

主树 `d08f9e…` 单次串行回归在CPU测试收集前退出，CUDA/18例矩阵/双卡均未启动，不能记为通过。`JITTOR_NO_BUILD=1` 检测到core stamp过期；只读比对显示CPU和CUDA原stamp均仅有恢复后的 `py_tensor_frontend.h/.cc` 两个源码mtime差异，其大小和SHA与原冷编及当前清单一致。原生文件和已审产品未改，失败进程已终止、缓存锁与可见holder为空，其他用户不可读FD不作全机断言。独立失败审计 `_state/release-day23/20261029-main-optimizer-placement-fix-regression-r1-hold/independent-terminal-audit.json` SHA `a03d25bbc2050eb55087748c36cae2b97c5e3a7d4e386c43d0a4d1ebf0e75b2a`。后续仅在核实完整编译输入、生成器、工具链及产物等价后，评估独立派生stamp；原stamp及失败记录不覆盖，尚无测试通过结论。


## 2026-10-29 optimizer显式参数state placement修复待验

矩阵证实constructor ambient放置错误后，主树引入optimizer私有scope，针对已有显式placement的每参数临时覆盖上下文并finally恢复，覆盖zeros/adaptive FP32 cast及SGD contiguous；FollowRuntime参数直接走原路径，ambient冲突仍未解决。新指纹 `d08f9ec25b8669fc9d171c1e41274d4061f605e9e79c8a354090d412b64f80ff`，相对旧版仅4 Python/测试路径，曾尝试的native clear API已撤去且两个C++文件与事前备份逐字节一致；现有native产品能否复用需重审。静态review `_state/release-day23/20261029-optimizer-per-parameter-placement-fix-r1/static-review.json` SHA `03fa7cb604c82dc7202a7cae5c99728ea435039d52c363e218d2c39a554589c4`；新增CPU/CUDA混组、显式冲突、异常嵌套和公共factory语义测试，尚未运行，原30个state错设备证据保留。

## 2026-10-29 当前主树optimizer初始state实际设备缺陷

主树 `cbfa19…` 真实CUDA测试 FP16/BF16 AdamW新建moment公开device为CPU（原测试在native dispatch前失败），独立机制探针进一步证实同步后真实CPU location；手工zero/cast在CUDA，helper trace定位constructor期间 `tensor_frontend` 默认CPU placement覆盖了CUDA参数的 `zeros_like`，add_param_group上下文不同。18例AdamW/SGD×FP32/FP16/BF16×默认/显式CPU/显式CUDA，含CPU/CUDA混组，共30个初始参数state与参数设备不一致；新增组均匹配、数值有限、fallback0，异常路径未测。矩阵独立终审 `_state/release-day23/20261029-main-cbfa-optimizer-placement-matrix-r1-hold/independent-terminal-audit.json` SHA `1283b8d70c9baceac2f2df47e495f810c7b0094207d397604d7f4f6ac88e2259`；辅助helper trace SHA `09cd32f8a55e0169a999edb9a7c8f6e32ae106679531f8644c0c4d639fbaf953`。这是功能设备缺陷，非仅显示错误或末位数值差；正在设计每参数内部状态独立scope，同时保留公开工厂显式device与FollowRuntime语义，尚未修复。

## 2026-10-29 候选CUDA发现moment设备显示差异

HEAD-only候选实际 r2 CUDA core + 6个r6生产Python模块/测试：replay 6/6通过，optimizer 12/14，新增FP16与BF16 moment初始状态公开 `device=cpu`、测试期望cuda:0；断言发生在native dispatch检查前，不能断言物理存储在CPU。独立终审 `_state/release-day23/20261029-headonly-r6-cuda-targeted-overlay-terminal-audit-r1/terminal-audit.json` SHA `fd98594a1d069e3565cb4e180db9a17d2b6e7d038afc0dbb37729cbaff732f7b`，fallback0，余alias/write/cat/RMS/pow/factory未执行。原CPU BF16严格moment数值失败保留。当前main有相似半精度状态构造，正准备独立真实CUDA专项同时测公开device、dispatch/placement，不能预判是否受影响。

## 2026-10-29 HEAD-only候选CUDA cold通过，定向回归运行中

候选r2独立CUDA cold自然rc0、真实UUID及231编译单元核验，194源码/8生成器与r6逐项相同；独立审计 `_state/release-day23/20261029-headonly-r2-cuda-cold-independent-audit-r1/terminal-audit.json` SHA `010575095d68c3ec8638655c0856bc554ece59c98c0f716f3694fe10a4ff0579`。后继实际为r2 CUDA core+六个r6 Python模块/测试，单次八阶段replay/optimizer/alias/write/cat/RMS/pow/factory；controller PID/PGID 1868906/start_ticks 188472844、首worker PID/PGID 1868907/start_ticks 188472935，路径 `_state/release-day23/20261029-headonly-r6-cuda-targeted-overlay-r1-hold/`；锁/缓存/资源已核，GPU有外部负载，非性能测量。恢复先核PID/start/receipt/锁，失败停止不重试；目前无GPU测试/训练通过结论，原CPU BF16严格失败及结构基线问题保留。

## 2026-10-29 HEAD-only候选独立CUDA cold运行中

候选r6需要真实CUDA而非沿用main产品；以已审r2 native源码与CPU cold前置、r6 native 190编译源/777文件/8生成器逐项等价，建立独立CUDA cache，r6仅后续Python/测试overlay。单次cap900，controller PID/PGID 1863596/start_ticks 188427923，worker PID/PGID 1863659/start_ticks 188427981，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`，入场约76827MiB空闲且外部利用率59%，只作正确性；目录 `_state/release-day23/20261029-headonly-r2-cuda-cold-r6-overlay-r1-hold/`。恢复先核PID/start/receipt/锁，不重复；cold与后续定向测试尚无结论，不能把main GPU训练转移到候选。

## 2026-10-29 HEAD-only r6事务与DDP CPU overlay通过

任务独立提交候选r6完整patch SHA `6e7be1ab21e98642fd19e9a0b0f096169787ec77ded4afe019c96afd8be2c6a7`，将主树hook事务修复最小移植，HEAD transaction/context/fidelity依赖字节一致；r5及其故障原件保留。单次CPU测试真实身份为r2 framework/native core + r6六个Python生产模块/测试、use_mkl=0，事务8/8、hook16/16、owner20/20、原隔离故障注入2/2通过，CPU dispatch/fallback0；独立审计 `_state/release-day23/20261029-headonly-r6-hook-transaction-cpu-overlay-terminal-audit-r1/terminal-audit.json` SHA `16de455dd8df575f7fa76566b0b618c47a759c5a11e26323f426d4e0c8703fbc`。此为混合身份、不证明r6 native cold或完整shim安装故障，r4 BF16 moment严格数值失败、原HEAD types.py 802行结构失败、候选CUDA/多rank/训练未验证均保留；真实index未改，候选未提交。

## 2026-10-29 BERT热步间NVML间隔诊断（共享负载）

新主树 `cbfa19…` 的单次PT/JT各12步QNNQ交错诊断、每端6 query前置/6无query，模型/权重/config一致、实际197非空梯度和199参数有限、fallback0、loss最大差2.38e-7。PT两组更新median 87.33/85.33ms，JT 235.82/209.18ms；JT组内差+23.30/+44.73/-4.70ms波动显著，查询本身约105–110ms不计入更新。两阶段目标卡持续有外来PID 1849785、只有5/7离散负载采样且无等长sleep对照，不能区分查询、闲置间隔或共享负载的影响，不作为新source正式性能。独立终审 `_state/release-day23/20261029-main-bert-b8s256-nvml-gap-cbfa-r1/terminal-independent-audit.json` SHA `33aa1b06961b2d764d7ec773cc5ebfaf604483e42435d53fc4a85c041ac803fd`；原 `31857f…` 正式B8/S256 1.735倍结论及原负载监测限制继续保留。

## 2026-10-29 最新主树两rank Trainer full4通过

事务修复后 `cbfa19af2cc2d358d9d0777b03f186c188aa8d684c704ba8af04332155a575a7` 的单次两rank HF Trainer tiny BERT LoRA FP32 full4自然rc0，两个指定GPU真实source/core/NCCL、fallback0，均8 microbatch/4更新、16样本，副本参数一致；复用旧独立PT full，gradient/parameter/optimizer最大差1.49e-8/1.04e-7/3.73e-9，loss最大差2.98e-8（诊断）。独立终审 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-full-transaction-r2/terminal-independent-audit.json` SHA `82d04f818d1283ea5a9bf6a7635bb2207fd158096d87a9d79c69fac28e1ed616`。首次启动因旧helper hash在任何attempt/模型前被拦截，原失败保留；此次不含新source的resume或四rank，旧 `31857f…` 两/四rank续训结果按其身份保留。

## 2026-10-29 hook事务修复CPU回归通过

新主树 `cbfa19af2cc2d358d9d0777b03f186c188aa8d684c704ba8af04332155a575a7` 的五阶段单次回归：新增rollback故障注入8/8、hook16/16、Torch owner20/20、既有事务/边界27/27、结构24/24均自然rc0，CPU dispatch和fallback0；源码/产品与四份冻结文档一致、进程组及可见锁/holder空。独立终审 `_state/release-day23/20261029-main-ddp-comm-hook-transaction-cpu-focused-r1-hold/independent-terminal-audit.json` SHA `c3e238a4d1b35fbc6a852460894de5f9e15f659dd458e389b05ce50a9f90408c`。覆盖新建/预存namespace三个故障点及重复安装；原r5隔离事务2失败保留，CPU mock不等于真实多rank，之前2/4rank Trainer与性能结果继续按 `31857f…` 身份报告。

## 2026-10-29 hook事务故障注入失败与主树修复

HEAD-only r5 混合CPU测试 hook16/16、owner20/20通过；原HEAD `types.py` 802行超过800的结构失败与两项受控故障注入失败均保留。注入在已有hook模块上证明InstallTransaction.rollback后属性未恢复，限独立registry而非完整shim安装；独立终审 `_state/release-day23/20261029-headonly-r5-hook-cpu-overlay-terminal-audit-r1/terminal-audit.json` SHA `a335a7965fabbae23fa985874b56a4240c634a1d25b81674c32fe9071f10790b`。主树同一helper已改用事务属性和模块发布账本，新增新建/预存及多故障点回归，旧文件备份 `_state/release-day23/20261029-ddp-comm-hook-transaction-fix-r1/`；新指纹 `cbfa19af2cc2d358d9d0777b03f186c188aa8d684c704ba8af04332155a575a7`，相对 `31857f…` 仅helper及新测试变动、native源码不变。尚未运行真实回归，旧训练/性能结果保留各自身份；不要预报修复通过。

## 2026-10-29 BERT B8/S256性能归因诊断

在相同main `31857f…`、权重、B8/S256默认attention/optimizer配置及独立性能cache上，PT/JT各3步profiler自然rc0、JT fallback0；两端实际均为BertSdpaSelfAttention/sdpa。PT memory-efficient attention前/反各36次事件总3.90/8.01ms，JT mea前/反各36次总9.44/21.05ms，约6.20ms/步事件差提示attention值得查，但不足以证明无profiler正式64.12ms/步差的根因。JT三步host wall310.10ms/device busy240.71ms/device wait176.83ms、2043 kernel；PT kernel区间并集238.15ms/2994 kernel，两个profiler捕获口径不同，不得直接相加或作benchmark。JT trace未截断，PT无独立截断指标；各侧只有5个离散负载样本、无可见外来PID。独立审计 `_state/release-day23/20261029-main-bert-b8s256-attribution-r1/terminal-independent-audit-v2.json` SHA `eaccfb0460905b5ca4e9bcea0f0aa2864ce4a54515ca5bb2b3868b24f4f34529`，原正式1.735倍结果及采样限制保留。

## 2026-10-29 主树BERT B8/S256 FP32性能三轮完成

当前主树 `31857f…` 的独立性能cache、60秒readiness 13/13合格后，PT/JT交替三轮各10次同步热更新自然rc0；PT轮median 87.36/86.98/87.47 ms，JT 152.45/154.24/146.11 ms，合并median 87.30 vs 151.42 ms、JT/PT=1.735，热loss最大差2.98e-7、JT fallback0。独立3步内存测量allocator peak allocated PT/JT约3.041/3.033 GB、reserved约3.282/3.329 GB，逐热步NVML采样峰3638/3650 MiB；不构成近容量证据。独立终审 `_state/release-day23/20261029-main-bert-b8s256-performance-r1-hold/terminal-independent-audit.json` SHA `9ac32552295e89f445465fe37f10341d48f547376071054a7872028e932ad940`。cold编译期间253次负载采样一次nvidia-smi超时，其余有记录采样未见目标卡外来PID，离散采样不能排除间隙干扰；该单格稳定偏慢是需处理的性能差距，完整Day4–5及near-capacity仍未验收，历史r3诊断与readiness失败保留。

## 2026-10-29 主树BERT B8/S256配对性能实验运行中

目标GPU空闲状态相对前次60秒失败明显变化后，新用途一次60秒readiness的13/13点连续合格，单次放行当前主树 `31857f…` 的独立性能缓存实验：FP32 BERT B8/S256，PT/JT交替三轮，每轮实际首更新、4次预热、10次同步热样本及独立allocator/NVML采样，非近容量验收。controller PID/PGID 1831102，独立JT cache `_state/release-day23/20261029-main-bert-b8s256-performance-r1-hold/performance-cohort/runtime/jittor/jittor-home`，日志在同目录。首次JIT串行，阶段失败即停不重试；恢复先核PID/start/receipt及锁。当前无性能结论，历史r3共享负载诊断与前次readiness失败保留。

## 2026-10-29 四rank HF Trainer禁P2P恢复通过

当前主树 `31857f…`，tiny BERT+LoRA FP32、16样本、4rank、batch2/G2，在PT与JT均明确 `NCCL_P2P_DISABLE=1` 下，full4/stop2/fresh-resume4六阶段及PT连续性/最终分析自然rc0；24个rank执行设备/实际产品与fallback0已核，六阶段NCCL各14条channel均SHM、0 P2P。checkpoint2位于epoch2，四rank RNG齐全，各端各rank自身40项checkpoint状态恢复max_abs0、full/resume最终参数max_abs0。跨框架共有字段最大差梯度2.98e-8、参数1.0966e-7、optimizer 2.794e-9、loss 2.98e-8仅作诊断；resume轨迹PT多两项initial_lr与scheduler `_get_lr_called_within_step`/`verbose`，不声称跨框架全部状态一致。独立终审 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-4rank-no-p2p-r1/terminal-independent-audit.json` SHA `40d432deb41cacc0178f8243c5f828c6dea382b9f7ab5cc9ae4cde3df6678b88`，进程/可见锁/FD终态清楚、拒读PID未知；终审两次只读脚本因字段差异/字符串finite中止已记录，原运行未重做。此限禁P2P/SHM，默认P2P PT full和微诊断超时、17样本尾批恢复、AMP/长期质量/性能尚未通过。

## 2026-10-29 四rank禁P2P Trainer后继运行中

同四卡微型NCCL禁P2P诊断通过后，新用途对PT与JT均明确 `NCCL_P2P_DISABLE=1`，其余16样本/epoch边界checkpoint协议不变；PT full/stop/fresh-resume及连续性先过才进JT。controller PID/PGID 1823071，binding SHA `2fafe9325667d2f6a81fc2f40c0528349dac48de1ce96bd9f27d91b7cb3fea73`，目录 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-4rank-no-p2p-r1/`；单次有界、失败停止不自动重试，恢复先核PID/start/receipt/锁。该适配配置不能称默认P2P通过；原默认PT full超时及17样本尾批恢复失败均保留，目前无四rank训练结论。

## 2026-10-29 四rank NCCL默认与禁P2P环境诊断

同四卡、同PT NCCL 2.21.5库，默认配置的单次简单all_reduce 60秒超时，四rank均记录进入collective前、无数值结果，日志混合P2P/CUMEM与SHM；独立 `NCCL_P2P_DISABLE=1` 单次自然rc0，四rank均得10.0，日志通道均为SHM。独立终审 `_state/release-day23/20261029-pt-4rank-nccl-diagnostic-r1/terminal-audit.json` SHA `fa9a71c2ce3033c3134722189bb4cf9ed47ab6f8241b5cc232dc30f41e6d18f2`。此支持传输路径相关环境问题，但不确定具体根因/故障链路；不能将禁P2P诊断称默认P2P或Trainer通过。正在准备独立、明确禁P2P的四rank PT/JT Trainer后继，原默认full超时保留。

## 2026-10-29 四rank HF实验PT full超时

四rank新用途的初始权重与NCCL四rank初始化完成，PT full单次300秒超时停止；日志显示部分通道P2P/CUMEM、部分SHM/direct/direct，卡住根因未知，不能归因Jittor。PT stop/resume及JT所有阶段均未启动，无重试；controller/launcher/rank进程组退出，锁及可见FD空，拒读PID未知。独立终审 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-4rank-r1-hold/terminal-failure-audit.json` SHA `297674242dc9ae2ba6b25eaababce195d09b21205dc37098c54326f9e299143d`；控制器TWO_RANK标签瑕疵保留，实际证据四rank。四rank公开Trainer与性能/显存仍未验收，两rank16样本续训通过及17样本尾批恢复失败分别保留。

## 2026-10-29 四rank HF Trainer epoch边界实验运行中

当前主树 `31857f…` 新用途：tiny BERT+LoRA FP32、16样本、四rank/batch2/累积2，每epoch1次更新，checkpoint2要求epoch2.0；PT full/stop/fresh-resume及连续性先过才运行JT对应三阶段，每阶段cap300、整体cap1800，失败即停不自动重试。四UUID依序 `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`、`GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`、`GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d`、`GPU-b6f59824-d1a3-cd6f-8100-9149728fc8eb`；余量约81/72/73/76 GiB，外来负载与不可读PID保留。controller PID/PGID 1817682，binding SHA `de3bf82b539cf4ec70e24c312e72d39a235c558a28fd2133336dcc1376e4cf19`，路径 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-4rank-r1-hold/`。控制器成功状态字符串沿用TWO_RANK属于标签瑕疵，必须以实际四进程与设备证据为准；当前尚无四rank结果，恢复先核PID/start/receipt/锁，不重复。旧17样本尾批恢复仍未通过。

## 2026-10-29 noneditable compat wheel混合布局入口通过

旧 `2dd2b3…` 身份compat wheel安装到隔离target后，184 RECORD记录及optimizer_api/optimizer_restore实际路径/哈希核对通过，wheel入口spec明确，动态torch namespace无 `__file__` 记null；使用当前 `31857f…` 主树CPU core及显式扩展的 `jittor.__path__`，公开Optimizer.load_state_dict的零参数零状态调用实际进入wheel恢复实现，将lr 0.1恢复为0.2，fallback相关张量未涉及。独立终审 `_state/release-day23/20261029-compat-wheel-noneditable-import-check-r3/attempt-01/independent-terminal-audit.json` SHA `5ad909581b409c6adaf2a0dfa38943d4f2a05f9a14da31401b27a9a3cb1bbe23`。早前RECORD相对脚本路径和namespace `__file__` 两次观察器失败均保留，lock内容变化单列；此混合布局不证明最新wheel或完整core+compat干净安装，也不证明张量状态恢复。

## 2026-10-29 HEAD-only候选 leaf/factory/numerics 后继通过

候选r2→r3仅更正旧 `torch.ones` 非leaf断言：独立PT 2.6实测为leaf、grad_fn None，运算输出非leaf；旧失败/误选CUDA skip保留。后继实际身份为r2生产源码与CPU core + r3测试文件、显式use_mkl=0，leaf 1/1、factory17/17、numerics18/18通过、fallback0；独立审计 `_state/release-day23/20261029-headonly-r3-cpu-test-overlay-terminal-audit-r1/terminal-audit.json` SHA `ec8a7c874d59052cfb3c6bf393d02de1b36707ba90237eb94557a61db4998407`。先前候选BF16 AdamW moment dtype失败、默认oneDNN缺产品与CUDA/multirank未验证继续保留；不能仅据这些CPU结果提交或转移主树训练证据。

## 2026-10-29 主树两rank HF Trainer epoch边界续训通过

新源码 `31857f377169f420ebe6a707c67d8b841850602a6dfb3530bd4b8a0d0bb25716` 的tiny BERT+LoRA FP32、16样本、两rank、G=2：Jittor full4/stop2/fresh-resume4与复用的独立PT对照均完成，checkpoint2 epoch1的40项模型/optimizer/scheduler状态与Python/NumPy/CPU/CUDA RNG匹配，16样本ID及顺序符合full控制；两rank实际CUDA UUID/参数/梯度ordinal正确、fallback0、6可训练tensor实际更新。JT full/resume最终参数max_abs0且副本一致；对已核PT最终参数max_abs 1.04075e-7、梯度max_abs 1.49e-8、scaled microbatch loss最大差2.98e-8（诊断）。NCCL实际SHM/direct/direct，进程组/可见锁及FD空，拒读PID未知。独立终审 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-jt-hook-r1-hold/terminal-independent-audit.json` SHA `78e0bef8c0931f33a88dc58e174579fcaf12fff8006ff52063bda4206c645ed2`。这不覆盖旧17样本尾批恢复、4rank/AMP、性能或长期任务质量；原缺hook失败及17样本PT恢复/观测失败均保留。

## 2026-10-29 HF Trainer 16样本修复后JT两rank运行中

新主树 `31857f377169f420ebe6a707c67d8b841850602a6dfb3530bd4b8a0d0bb25716` 经hook/native/Torch/structure CPU四阶段89项通过；同一16样本PT full/stop/resume及41项权重/报告/checkpoint资产已重新只读核实，不重跑PT。JT-only单次full/stop/fresh-resume后继已启动：controller PID/PGID 1808060、full launcher PID/PGID 1808065、rank0 PID/PGID 1808067、rank1 PID/PGID 1808068，两rank框架/锁/NCCL身份前置通过；binding SHA `850d2b7cee25ec1fbf822c9c21994969fc5e67051d1c57b584005b1c1086fb23`，目录以 `_state/release-day23/20261029-hf-trainer-ddp-epoch16` 后继记录为准。恢复先核PID/start/receipt/锁，失败停止不自动重试；原JT缺模块与17样本失败保留，当前无JT训练结果。

## 2026-10-29 默认DDP comm-hook导入修复CPU回归通过

针对HF两rank默认NO hook导入失败，新源码 `31857f377169f420ebe6a707c67d8b841850602a6dfb3530bd4b8a0d0bb25716` 提供Accelerate无条件读取的hook模块和符号，实际自定义/压缩/PowerSGD hook或wrapper执行明确抛NotImplementedError并标注UNIMPLEMENTED，默认NO不注册；Accelerate对NO hook+非NO wrapper的既有静默忽略不宣称支持。CPU单次四阶段 hook16/16、native DDP29/29、Torch owner20/20、完整结构24/24通过，fallback0，冻结文档一致；独立审计 `_state/release-day23/20261029-main-ddp-comm-hook-cpu-focused-r1-hold/independent-terminal-audit.json` SHA `df976f0ee009ab3966615d07d0d2c31bc8d11b396b646a52ddbbcf0ee009b73e`。旧16样本Jittor导入失败保留；CPU mock不代表真实多rank。新模块和既有dirty备份 `_state/release-day23/20261029-ddp-comm-hook-surface-fix-r1/`。

## 2026-10-29 HF两rank 16样本对照通过、Jittor缺comm-hook导入

16样本epoch边界协议PT两rank full4/stop2/fresh-resume4及连续性均通过：两rank 40项checkpoint状态、四类RNG、样本顺序、最终参数匹配。PT resume rank1实际有额外GPU context，但当前ordinal/参数与目标UUID一致；不能反推原17样本失败的未记录列表。Jittor两rank在HF Trainer→Accelerate.prepare_model的默认DDP kwargs导入 `torch.distributed.algorithms.ddp_comm_hooks` 时均ModuleNotFoundError，未开始更新；fallback仅前值0，异常后未记录。独立审计 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-r1-hold/terminal-independent-audit.json` SHA `c116512c2b88c9703c2b959d64bd016eb5235d521decc33d757a51e300d49547`；旧17样本失败不覆盖。准备默认NO hook的导入兼容，实际压缩或PowerSGD hook必须显式不支持；16样本并不覆盖尾批恢复。

## 2026-10-29 HF两rank 16样本epoch边界新实验运行中

另建16样本、G=2、checkpoint2位于epoch边界的独立协议；PT/JT各full4、stop2、fresh-resume4，先核PT自身样本/状态/RNG连续性才放行JT。原17样本尾批fresh-resume失败保留，16样本不声称尾批恢复覆盖。main `a2a524d5…`、既有CUDA/NCCL产品及锁/资源已核；release SHA `cdcd1a436156ac790035e3d23bfcb52e858b1df745f3bbffc14690e502857b91`、binding SHA `0484a11c1d6891dfd8508ed345b0230af14fa390cc62e51db79d149791658793`。controller PID/PGID 1799310，PT full launcher 1799317自然rc0；当前PT stop launcher PID 1799402、rank 1799410/1799411，目录 `_state/release-day23/20261029-hf-trainer-ddp-epoch16-r1-hold/`。恢复先核PID/start/receipt/锁，不重复；当前无PT完整恢复或JT通过结论。

## 2026-10-29 两rank HF Trainer原17样本实验失败终审

独立PT full4/stop2完成；fresh resume 的rank1因设备观察器仅允许单个NVML UUID而退出，异常未记录完整UUID列表和恢复后参数ordinal，NCCL cudaDev1/nvmlDev2/busId53000与目标PCI一致但不能复原现场或断言原因。rank0 checkpoint step2模型/optimizer/scheduler 40/40及四组RNG匹配；同时full只有7 microbatch，stop4+resume5为9，样本与sync序列不一致，原因未知。原独立终审 `_state/release-day23/20261029-hf-trainer-ddp-resume-r1-hold/original-failure-terminal-audit.json` SHA `16208f0c09037f7197b5cf70cca02938622a62fa31f789443ceac561461992cd`；所有进程组和可见锁/FD空，拒读PID未知。Jittor阶段未启动，不能声称尾批恢复通过；准备单独16样本整除累积、epoch边界checkpoint的新协议，保留原失败且明确不覆盖尾批恢复。

## 2026-10-29 主树两rank HF Trainer新进程恢复实验运行中

以新主树 `a2a524d5…` 为Jittor身份，使用独立两卡及匹配PyTorch、固定tiny BERT+LoRA、G=2的单次新用途：两端各full4、stop2、fresh-resume4，逐rank报告/真实设备/恢复与同步，单阶段cap300、总cap1800；按实际失败停止，不自动重试。release review SHA `d7693adf334c45a0b1c025c4a0bfb8b0256dec03628f66ae197d0dede01efe0c`、binding SHA `c10e4eca0e0e2f3bd2603ca1db4fb69c59c1a8d3482529386b7a5dc961598796`；controller PID/PGID 1793439，PT初始化权重PID 1793445 rc0（weights SHA `4e595646d39667958e2e88efb0bf6950e9baf2d82b8ece57b31e709431391cf3`），PT full launcher PID/PGID 1793464、rank0 1793584、rank1 1793585。目录 `_state/release-day23/20261029-hf-trainer-ddp-resume-r1-hold/`；恢复先核PID/start/receipt/锁，不重复；当前任何训练、恢复或对照尚未终态。

## 2026-10-29 HEAD-only候选CPU阶段结果

独立HEAD候选首次focused的native3 3/3、native DDP 17/17通过；Torch DDP因未预装oneDNN导致11节点失败，原rc1与审计 `dd0bb57238eaf912443429f04894ac51aa75d7263ffcd52b3d4e6d954b627059` 保留。明确 `use_mkl=0` 的新后继中 Torch DDP 20/20、replay 6/6通过，optimizer 10/11；BF16参数的exp_avg实际BF16而非测试要求FP32，属于候选状态dtype差异，与主树先前FP32 moment下一步数值末位差异并非同一失败。独立终审 `_state/release-day23/20261029-headonly-r2-cpu-no-mkl-focused-terminal-audit-r1/terminal-audit.json` SHA `3f63936d2eddfa21575365b071adcd87612363eaa27e555a245f1a5d6c71a255`；alias/factory/numerics因失败即停尚未执行。目标候选未达到提交验证，不把主树训练或默认oneDNN结果转移到候选。

## 2026-10-29 HEAD-only提交候选CPU cold通过，定向回归运行中

候选独立CPU cold自然rc0，227单元、native/core/utils/stamp及全快照校验通过，独立审计 `_state/release-day23/20261029-headonly-r2-cpu-cold-independent-audit-r1/terminal-audit.json` SHA `d4fe42c9446f7bf4746ac468f5a026fabba9763b6713d2cc2f8886281b00a036`；仅编译和导入身份，无张量或训练结论。候选CPU focused已单次启动，controller PID/PGID 1788240/start_ticks 187872989，首worker PID/PGID 1788242/start_ticks 187873025；首worker identity.command启动时为空，receipt单独存实际命令，不补造。路径 `_state/release-day23/20261029-headonly-r2-cpu-cold-focused-r1-hold/focused-attempt-01/`；失败即停不自动重试，恢复先核PID/receipt/锁。候选与main的既有dirty依赖不同，main实训不能转移；用户dirty与真实index未改。

## 2026-10-29 任务独立提交候选CPU cold运行中

为不夹带既有dirty，在非Git HEAD快照上构造任务补丁 `_state/release-day23/20261029-task-only-head-candidate-ddp-r2/candidate.patch` SHA `b536de4a831202a0ce45b135c7ce7210cf8a3e45ca960660afe800bdce3cf7bb`；主进程用隔离临时index将补丁应用HEAD并逐一核对2789文件/符号链接与快照一致，真实index未改，代码适配与既有dirty依赖仍须测试。候选完整清单指纹 `d55d62807b579f420260c457ecef71552054a009c6a1640f5b994b8c20609a9b`，CPU cold单次cap600：controller PID/PGID 1781640/start_ticks 187834786、worker PID/PGID 1781642/start_ticks 187834826，目录 `_state/release-day23/20261029-headonly-r2-cpu-cold-focused-r1-hold/`。恢复先核PID/start/receipt与锁，不重跑；cold仅构建身份，候选测试/训练尚无结论，不能转移主树证据或提前提交。

## 2026-10-29 当前主树性能窗口单次检查未就绪

`a2a524d5…` 主树完成真实两rank终审后，对GPU `GPU-2be51dc7-d99a-7307-869b-30ed5888e701` 新用途单次60秒离散readiness检查：13点全部不合格，free恒59939 MiB（原准入≥60000）、利用率28–31%（原准入≤10%），外来PID 1461480全程可见，最终连续合格0；自然 `WAIT_TIMEOUT_NO_RELEASE_WINDOW`，无benchmark/JIT或自动重试。独立终审 `_state/release-day23/20261029-main-day45-readiness-a2a-r1/terminal-audit.json` SHA `210310c0c4a40e0e19198b6eff3be99f86b69e89ba8e509bcb590f04694419e2`；采样离散且GPU/PID查询非原子，不能推断间隙状态。这是无正式窗口而非benchmark失败；历史r3倍率仅诊断，near-capacity未证明，Day4–5性能/显存仍待验收，不重置旧T5/Vision预算。

## 2026-10-29 主树真实两rank Accelerate DDP尾批通过

新源码 `a2a524d5e9b416de511bb679e71944bba68b9ef381e638b25582661a12865465` 的两rank tiny FP32 Accelerate DDP后继自然rc0：两物理GPU UUID实际匹配，NCCL日志记录SHM/direct/direct，fallback0，17个样本ID全覆盖，累积sync标记false/true/true，参数实际更新且两rank最终参数max_abs=0；各rank对历史PT的254数值项、39精确项通过，最大绝对差1.4901161193847656e-8。独立终审 `_state/release-day23/20261029-ddp-owner-successor-r1-hold/rank-terminal-independent-audit.json` SHA `200495b3dbc2d6d2ffb3a001fdd995caa8dce25c88452ecf012973cd8317fcbc`，进程组和可见锁/FD已退出，拒读PID未知；旧主树两rank TypeError及CPU首轮失败保留。历史PT报告未记录fixture脚本hash及确切版本，匹配依据和限制保留；此结果不覆盖4rank、HF Trainer、DDP checkpoint新进程恢复、AMP或性能。

## 2026-10-29 DDP修复后两rank后继运行中

新源码 `a2a524d5e9b416de511bb679e71944bba68b9ef381e638b25582661a12865465` 已有native 29/29、Torch 20/20 CPU终审；经旧NCCL产品、当前源码、共同锁及两卡资源复核，单次真实两rank后继已启动：controller PID 1770885、launcher PID/PGID 1770886、rank0 PID/PGID 1770888、rank1 PID/PGID 1770889；最终binding SHA `7334c1e6d7c2bc3bf753506e82b24d6d1a2e4742ea09df089fa8ec31433fd2c5`，目录 `_state/release-day23/20261029-ddp-owner-successor-r1-hold/`。原launcher LOCAL_RANK保持，旧两rank TypeError和CPU首轮失败保留；恢复先核PID/start/receipt/锁，不重复启动。新测试结果尚未终态。

## 2026-10-29 DDP owner及dtype修复CPU定向通过

native DDP计算梯度桶大小改为从native Var读取dtype，保留现有分组；新增native/Torch多dtype与桶边界测试。新源码指纹 `a2a524d5e9b416de511bb679e71944bba68b9ef381e638b25582661a12865465`；CPU独立终审 native 29/29、Torch 20/20，覆盖先前失败的双owner用例和新增dtype节点，fallback0；审计 `_state/release-day23/20261029-main-ddp-owner-bucket-cpu-terminal-audit-r1/terminal-audit.json` SHA `7a9007d7dc95425305bde32abdb35a59d046e831ca96ea90c77960202ebdc1bb`。原先28/28与18/19失败、原两rank TypeError均保留；CPU mock collective不等于真实多卡，下一步重绑后单次测试真实两rank。

## 2026-10-29 DDP修复首轮CPU回归与打包检查

DDP源码 `5466a63e…` 下 native原测试28/28通过；Torch owner相关18/19通过，唯一失败是在构造native DDP包装Torch Linear时 `parameter.dtype` 为torch.dtype，无 `dsize()`；该失败发生在预期双owner拒绝前。两阶段fallback0，独立审计 `_state/release-day23/20261029-main-ddp-owner-cpu-focused-terminal-audit-r1/terminal-audit.json` SHA `427d53c55c13fa443c277df7a337682e3c02e80ca812c08fa2360da7e4bcc5bb`，原失败保留；准备使用native dtype描述符修复并复测。真实两rank仍未通过。

新optimizer_restore模块的隔离sdist/wheel已实际构建rc0，两归档模块字节、wheel RECORD及显式required-member审计通过，审计 `_state/release-day23/20261029-main-compat-optimizer-restore-package-audit-r1/terminal-audit.json` SHA `f9bc8a67727c5bb55fccf3b89e226793a6be99039e1ec970492b41f51903d4ae`；未安装wheel或证明非editable运行，新文件仍为Git未跟踪，主树旧egg-info尚未刷新。

## 2026-10-29 DDP owner隔离修复及CPU回归运行中

针对原两rank SimpleNamespace冲突，Torch参数使用独立 `_torch_ddp_state`，native保留自身marker，双owner包装在通信前拒绝，native loss驱动路径对Torch参数/输入列表提前拒绝并保持状态；标准Torch backward/step、no_sync和closure需回归验证。原dirty备份 `_state/release-day23/20261029-torch-native-ddp-owner-fix-r1/`，新执行指纹 `5466a63e9f2e83d585fcf6deaffa977cb191f32aa6c237ba74b847e669577a7d`，native编译输入不变。CPU native DDP全文件与Torch owner/closure两个阶段串行执行：首worker PID/PGID 1765387、start_ticks 187738264，目录以实际controller记录为准；恢复前核进程/receipt/锁，不重复。旧两rank失败保留，CPU与新真实两rank结果尚无结论。

## 2026-10-29 主树两rank Accelerate DDP真实功能失败

新源码 `2dd2b3b5…` 两rank单次测试均完成preimport guard、真实不同GPU UUID及NCCL初始化，fallback各0→0；目标测试call均在optimizer step失败，native `_assert_ddp_step_ready` 把public Torch DDP 的不可hash `SimpleNamespace` 当字典键。该state也缺native后续所需属性，不能只改hash；无更新/对照通过证据。独立终审 `_state/release-day23/20261029-ddp-preimport-guard-rebind-r1-hold/rank-terminal-failure-independent-audit.json` SHA `2e98c575f56a2b5511af592aef56f4a6ea4759019227aa450fafe67d686ed5a9`，全部进程组和可见锁/FD已释放，拒读PID未知。原失败保留，正在审查隔离Torch与native DDP状态和避免重复allreduce的最小修复；新源码需重绑，不能复用旧binding或自动重试。

## 2026-10-29 主树两rank Accelerate DDP 尾批测试运行中

新源码 `2dd2b3b5…` 的单进程NCCL custom-op预热自然rc0、真实库/NCCL初始化和fallback0，独立终审 SHA `422a2969e01914da46141219ac96d7c17b19575c49ec632c2b7bd9f36d160a6f`。受限Jittor路径共同锁审阅 SHA `b80d307b71ac33c2b52c7660aa451b7087c46485bbf47ab25ffb94bbbc4f9106`，第三方/vendor未观测行为作为边界。实际两rank已单次启动：controller PID 1757090、launcher PID/PGID 1757092、rank0 PID/PGID 1757094、rank1 PID/PGID 1757095；绑定 SHA `7a14dc0f26194d493da642b3604b629ae42911fac7f08d215b2aba2596ece712`，目录 `_state/release-day23/20261029-ddp-preimport-guard-rebind-r1-hold/`。GPU UUID为 `GPU-2be51dc7-d99a-7307-869b-30ed5888e701` 与 `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`。保持原launcher LOCAL_RANK，恢复先核PID/start/receipt及锁，不重复启动，运行结果未终态。

## 2026-10-29 主树两卡DDP的单进程 NCCL 预热运行中

新主树指纹 `2dd2b3b5278d2e431eab63118d82ecd61f541063d313c6d6354724b74bad80b3` 的BERT单步及CUDA optimizer已终审，受限调用链及可见进程/共同锁核验后，单次NCCL预热worker PID 1755063、启动epoch 1793267445.751、cap300，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`；目录 `_state/release-day23/20261029-ddp-preimport-guard-rebind-r1-hold/warm-attempt-v2-01/`。另候选第二卡 `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`，只要求足够显存，外来负载和拒读PID未知保持记录。恢复前先核PID/receipt/锁，不重复；当前预热及两rank均未通过，NCCL产品须实际绑定。

## 2026-10-29 新 optimizer拆分身份BERT单步实证

新指纹 `2dd2b3b5278d2e431eab63118d82ecd61f541063d313c6d6354724b74bad80b3` 下BERT seed821 BF16由旧checkpoint50真实恢复并单步更新至51：新restore模块实际载入201组状态并返回，603模型/moment CUDA、四组RNG一致、201梯度FP32 CUDA finite、201参数变化，样本/shape/LR与旧full及PT一致，fallback0；模型child自然rc0。原包审计器错误只检空状态call000而忽视checkpoint call001，控制器审计rc1保留；独立审计 `_state/release-day23/20261029-bert821-main-restore-split-step51-independent-audit-r1/terminal-audit.json` SHA `8be210b7258def3e6aec6d6e14c5d95009f7f1a6d09f505c2fe4a409af73a741`。旧与新step51前态201签名全同、后态200/201相同，token_type_embeddings后态只有hash不同比较，值差未知；本步loss JT 0.3322448730、PT 0.2778320313，逐步值仅观察。此单步不是完整曲线/固定eval，拆分前主树BERT/Qwen续训结果继续按旧指纹报告。

## 2026-10-29 新 optimizer结构拆分身份的BERT单步运行中

新执行指纹 `2dd2b3b5278d2e431eab63118d82ecd61f541063d313c6d6354724b74bad80b3` 基于旧BERT checkpoint50实际恢复并仅执行第51步，验证拆分后optimizer恢复调用、CUDA状态、RNG、梯度、参数变化及相同样本/LR；不声称完整训练或eval。单次cap1200，worker PID/PGID 1750026、start_ticks 187639585；binding SHA `3aacccfb90e4c116e9a9f8c9a5118a48bdba4c561020ca89a4801f012b33d0a1`，目录 `_state/release-day23/20261029-bert821-main-restore-split-step51-r1-hold/`。恢复先核PID/start/receipt及锁，不重复启动；拆分前BERT/Qwen实训继续按其旧身份记录。

## 2026-10-29 optimizer结构拆分后定向复测

主树执行指纹 `2dd2b3b5278d2e431eab63118d82ecd61f541063d313c6d6354724b74bad80b3` 相对Qwen/BERT实训身份只有optimizer_api.py和新增optimizer_restore.py两项Python变化，native编译输入未变；旧入口保留，恢复函数体AST相同。原结构失败节点现通过（705行，原800上限未改）；CPU optimizer 10/11，既有BF16 next-step moment差异依原阈值仍失败、fallback0；CUDA optimizer 11/11、fallback0。独立终审 `_state/release-day23/20261029-main-optimizer-structure-split-focused-terminal-audit-r1/terminal-audit.json` SHA `ce9c37344587d8022c0551baeb832b873ba24dd9c3b7e193711acb9c93c30e63`；原十文件总体失败、这次三阶段总体rc1均保留，不能称十文件全套复跑通过。旧身份两网络训练与新身份定向回归分别记录。

## 2026-10-29 主树 Torch CPU 十文件核心集结构门槛失败

230项实际执行，226通过、3原规则skip、1失败：`optimizer_api.py` 为852行，超过原800行结构上限。独立终审 `_state/release-day23/20261029-main-torch-cpu-core-ten-terminal-audit-r1/terminal-audit.json` SHA `35fb18dbf425dcaa4e138c6a66a75c400919ad1ee48e7609eb169db5d4fc2778`；四份文档测试窗口冻结一致，终态fallback未记录，进程组和可见锁/FD空，拒读PID未知。原结果保留，不调整阈值；准备拆出optimizer恢复实现并绑定新源码身份、按受影响范围复测。之前Qwen/BERT训练均属于拆分前源码 `6c3decf3…`，不会自动转移为新身份。

## 2026-10-29 主树 oneDNN 显式 provider 及三个CPU节点通过

默认启用、显式指定已核库的oneDNN sentinel真实CPU matmul及数值通过；新后继复用该custom-op和主树CPU core，原先受阻三个节点的setup/call/teardown全部通过，父进程fallback 0→0。两个frontend子进程均rc0且日志分别记录1、6条真实oneDNN matmul，子进程fallback未独立测量。锁及可见FD无holder，拒读PID未知。独立审计 `_state/release-day23/20261029-main-r12-onednn-cpu-successor-terminal-audit-r1/terminal-audit.json` SHA `96fa0652da767dc61c4ccf7cbc712b200cc56e0d18cc86f733053e7fc7c09d22`。原recorder参数错误rc1、原no-build oneDNN缺失失败、CPU BF16 optimizer差异及CPU cold rc1均保留；此结果不证明默认库自动发现或完整CPU套件。

## 2026-10-29 oneDNN 实际 sentinel 通过，补测脚本失败保留

主树默认启用并显式指定现有oneDNN库的单次sentinel自然rc0，实际加载所指定libdnnl、选中 `_mkl_matmul` 且数值检查通过。后续pytest recorder把hook参数写作 `r` 而非hookspec要求的 `report`，三个节点收集之前失败，控制器rc1；原结果完整保留在 `_state/release-day23/20261029-main-r12-onednn-cpu-supplement-r1-hold/attempt-01/`，不得写为三项通过。正在独立修正测试记录器并准备只对未执行的三个节点作新后继，复用已经构建的custom-op，不重复冷编；自动发现及frontend子进程fallback仍未证明。

## 2026-10-29 主树 oneDNN 显式库 CPU补测运行中

Qwen/BERT续训独立终审后，实际源码、CPU产品、provider库及头文件和可见进程/锁已核对；仅可见范围无FD holder，拒读PID未知。oneDNN单次补测controller PID 1723686、sentinel PID/PGID 1723687、start_ticks 187515046，sentinel cap420；目录 `_state/release-day23/20261029-main-r12-onednn-cpu-supplement-r1-hold/`。先核实际载库和matmul，成功后再跑三个旧受阻节点；恢复先核PID/receipt/锁，不重复启动。此为默认启用但显式指定库的配置，不预报默认自动发现或frontend子进程fallback通过。

## 2026-10-29 主树 BERT 50→100 续训独立终审通过

主树seed821 BF16从旧候选 checkpoint50 续训50次、0 skip，50×201梯度实际CUDA0/FP32且有限，首步201参数均变化；603模型及moment真实CUDA、fallback0。99个microbatch的样本/shape及50个学习率与旧Jittor full及PT一致；固定33样本主树eval loss 0.4661458433、accuracy 26/33，历史独立PT full loss 0.4836055934、accuracy 26/33。实际checkpoint100模型、moments、step与签名相符，scheduler epoch100；原checkpoint50严格恢复及Python/NumPy/CPU/CUDA RNG检查通过。进程自然rc0、未超时、可见终态锁和FD无持有者，拒读PID未知；独立审计 `_state/release-day23/20261029-main-bert-resume-terminal-independent-r1/terminal-audit.json` SHA `7dae1c2561ec5b7eeeb4fb3055e9d0744e2cbb5d828b1696c844c23cb90403d8`。前50步属于旧候选，PT为独立full任务参照，不声称主树从头训练或抹去旧双seed strict失败。

## 2026-10-29 主树 BERT 50→100 自然完成，待独立终审

主树seed821 BF16 BERT续训单次自然rc0、未超时；包内终审14/14报告50次更新、0 skip，逐次201个FP32 CUDA梯度有限，99 microbatch样本游标及50学习率与旧full一致；step51 201参数变化，step100 201模型/402 moment真实CUDA、fallback0，checkpoint100核心文件存在。固定33样本eval loss 0.4661458433、accuracy 26/33；同协议历史独立PT full为0.4836055934、26/33，属于任务级参照而非同一续训来源。receipt SHA `36c649d749d139ac4f0a2974d4a7d3d18bfb5e89e6584c6339395f17eb83e0c3`、包审计 SHA `3601fe446a3d570e82ca79ebb6049b201b7198a46b1bba91d257bdaa2120e4cc`，独立终审待完成。前50步为旧候选，不称主树从头训练；旧BERT两seed strict FAIL保留。

## 2026-10-29 主树 BERT 首次续训更新通过

主树BERT seed821 BF16第51步非零学习率实际更新，201/201模型参数发生变化，更新后201模型与402 Adam moment真实CUDA0、fallback0；已记录step51–59每步201梯度设备，进程 PID 1710979/start_ticks 187428056 继续至100。checkpoint50的逐名状态及四组RNG此前已实际恢复，前50步来自旧候选；训练全程有限值、固定验证和终态仍待报告，不预报结果。

## 2026-10-29 主树 BERT checkpoint50 加载通过，续训进行中

实际主树seed821 BF16 BERT checkpoint50 的model/optimizer/scheduler/scaler具名签名strict与functional均通过；201模型参数及402 Adam moment实际public/native cuda:0，201 optimizer step、trainer及scheduler为50，Trainer真实加载边界Python/NumPy/CPU/CUDA四组RNG一致。worker PID/PGID 1710979、start_ticks 187428056仍运行，进入更新；报告路径 `_state/release-day23/20261029-bert821-main-resume-executable-r1-hold/runs/bert-821-resume/`。当前只是加载通过，不预报训练与固定验证终态，前50步来源旧候选。

## 2026-10-29 主树 BERT checkpoint50 续训运行中

seed821 BF16单次主树BERT 50→100续训已启动，worker PID/PGID 1710979、start_ticks 187428056，cap1800；release review SHA `7a1a74e902ea377a611a092d9a051b792e30d4a66e269d57932a7769703bf8ae`，binding SHA `b428f7e81d13c865b83342952c1951c7fa543208e1859e44e22dd0a21f989da7`，目录 `_state/release-day23/20261029-bert821-main-resume-executable-r1-hold/`。旧checkpoint50已离线逐名核对201模型tensor、201 optimizer state及scheduler，RNG文件记录各组但旧raw hash不可得；实际主树设备、恢复、更新、数据及验证仍需实测。前50步来源旧候选而非主树从头训练。恢复前核PID/start、receipt、锁，不重复启动。oneDNN与其他JIT保持等待。

## 2026-10-29 主树 Qwen 实用续训及 lifecycle 终态通过

主树seed821 BF16从旧r10 checkpoint51续训至102，实际51次更新、0跳步；每步112梯度均FP32、真实CUDA且有限，101个样本ID与旧full及PT后半程一致，112 LoRA参数均变化并与旧full最终参数一致。固定验证loss主树2.4555516243、PT 2.4566440582；新checkpoint原始portable内容和sidecar严格一致。predict返回loss 2.4555516243，adapter重载/生成及merge完成，merge logits最大绝对差0.000289917在原0.001预算内，fallback0；进程自然rc0、未超时。独立终审 `_state/release-day23/20261029-main-qwen-resume-terminal-independent-r1/terminal-audit.json` SHA `2960455d2342bb592ed7a1e83756fb430984c4d3eac738131429d103e94308d8`。此为主树后51步续训，前半程与完整full来自旧r10；eager evaluation为限定workaround，当前checkpoint直接由原生PyTorch torch.load optimizer仍因portable格式失败，旧strict/CPU BF16和oneDNN失败保留。GC实际重计算及正式性能/显存不在通过范围。

## 2026-10-29 主树 Qwen 续训已达102步，lifecycle运行中

主树seed821 BF16从旧r10 checkpoint51续训报告写出：51→102共51次optimizer更新、0跳步；51组梯度无缺失/非有限，dtype FP32；101个续训microbatch的step与sample ID同旧full后半程逐项一致。checkpoint 337实际CUDA张量、112 step、scheduler和四组RNG恢复已核验，冻结基座不变、fallback0；两次eager evaluate返回，固定验证loss 2.4555516243。报告和记录位于 `_state/release-day23/20261029-main-qwen-checkpoint51-resume-eager-r2-hold/runs/qwen-821-resume/output/`。前51步与对照full为旧r10源码，主树不是从头训练。进程PID 1674876/start_ticks 187236743仍运行，strict lifecycle/predict尚未终态，不预报最终通过；恢复前核PID、receipt与锁。

## 2026-10-29 主树 Qwen checkpoint 加载通过，续训进行中

主树Qwen训练进程 PID 1674876/start_ticks 187236743 仍运行；checkpoint51 同进程状态比较strict hash及functional通过：337个张量（112模型参数、224 optimizer张量、1 buffer）实际public及native cuda:0，112个scalar step均为51，trainer step及scheduler epoch均51，Python/NumPy/CPU/CUDA RNG恢复检查通过。报告在 `_state/release-day23/20261029-main-qwen-checkpoint51-resume-eager-r2-hold/runs/qwen-821-resume/output/`。仅证明加载，不预报后续更新、样本、验证或终态；恢复先核实际PID/receipt及锁，不重复启动。

## 2026-10-29 16:39 +0800 主树 Qwen 实际续训运行中

主树checkpoint51续训已单次启动，controller PID 1674865，训练进程PID/PGID 1674876、start_ticks 187236743，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`；同进程身份文件确认导入主树及CUDA core SHA `bbd4a209b3f0e780bc19cac53dde4d1b7f7ea98d305fae55d70c012800084839`。路径 `_state/release-day23/20261029-main-qwen-checkpoint51-resume-eager-r2-hold/`；恢复先核PID/start、receipt、锁与实际记录，不重复启动。尚无训练/续训终态结果。oneDNN补测及多卡补测仍HOLD，不与首次JIT并行。

## 2026-10-29 主树 CUDA 定向回归通过；Qwen 续训放行

主树CUDA 8阶段91/91节点、各节点setup/call/teardown通过，replay 6、optimizer 11、alias 22、writes 1、cat 14、RMSNorm 2、pow 1、factory 34，全部自然rc0、fallback0；writes与pow含CPU子分支，代表性alias/pow探针确认CUDA，不能推断每节点独立dispatch遥测。独立终审 `_state/release-day23/20261029-main-r12-cuda-targeted-plan-r1-hold/attempt-01/terminal-audit.json` SHA `2203487021fbfe15d226a72a39bf9cea36d4afb3f20669bac9437df2689f0c98`；目标锁和进程组空、终态cache 1576项。factory 使用与旧PyTorch独立对照完全相同的测试文件，本次不是重新运行PyTorch。

主树Qwen从旧r10 checkpoint51继续至102，固定seed821、BF16、eager evaluation、无Python overlay；审核实际产品、CUDA阶段、数据/checkpoint和GPU后放行，release review SHA `c79c7396686ca97befea50aba544dbb1378dc5a79e149ccae03550f1260067b2`。前51步及完整两epoch训练曲线来源仍是旧r10，主树只验证后半续训，不预报结果；CPU BF16数值失败、旧oneDNN失败及旧Qwen失败均保留。性能独立未验收。

## 2026-10-29 主树 CUDA 定向回归进展

实际主树源码及CUDA产品下，replay 6/6、optimizer 11/11、storage alias 22/22、显式写入1/1（含CPU/CUDA子分支）、autocast cat 14/14、mixed RMSNorm 2/2、pow 1/1（含CPU/CUDA子分支）依次自然rc0、fallback0；代表性alias/pow设备探针为真实CUDA，不推断每个节点均有独立dispatch遥测。factory 第八阶段正在运行，PID/PGID 1671783、start_ticks 187221014；恢复先检查进程与 `_state/release-day23/20261029-main-r12-cuda-targeted-plan-r1-hold/attempt-01/serial-receipt.json`，不要重复。结果未终审，不能推断训练通过。主树Qwen续训待所有阶段终态和审查。

## 2026-10-29 主树 CUDA 定向回归已启动

串行8阶段计划基于已核验主树CUDA cold产品与指纹 `6c3decf371a2cb7d14b7c224f2f3effaa77817a3cbe5b93d0193891cc93818d2`，首先运行 replay；worker PID 1656851、PGID 1656851、start_ticks 187179714，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`，入场可用显存58054 MiB。记录位于 `_state/release-day23/20261029-main-r12-cuda-targeted-plan-r1-hold/attempt-01/serial-receipt.json`；各阶段单次cap900，失败即停，不自动重试。恢复时先核进程和receipt，不预报测试结果；完成后才释放主树Qwen checkpoint51实际续训。

## 2026-10-29 主树 CUDA 冷构建身份核验通过

主树执行指纹 `6c3decf371a2cb7d14b7c224f2f3effaa77817a3cbe5b93d0193891cc93818d2` 的 CUDA cold 自然 rc0，耗时393.37秒、231个编译单元；实际设备 GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`，Jittor及Torch导入均为 has_cuda/use_cuda=1。receipt SHA `0ae76fd99a6f28033d7a76a3fdae80df8e1ade9ef454353fa3c62ae096cb2aec`；独立终审 SHA `ec4847e591ce955d0e89f68bae0d56b7dd3112610a9428704b8a081386051332` 核对源码、38个补充pyc、core/utils/stamp、编译输入及缓存终态，可见锁与holder为空，拒读PID仍未知。仅证明冷构建和shim身份，不证明tensor数值、训练或性能。原CPU cold rc1及限定native审计继续保留。主树CUDA定向回归准备串行执行，主树实际Qwen checkpoint续训尚待；历史mixed身份结果不直接转写为主树通过。

## 2026-10-29 16:19 +0800 主树 CPU 补测通过，CUDA cold 运行中

主树CPU补测：修正后的 FollowRuntime 内部分支在默认配置1/1通过，3个原oneDNN缺失节点在明确 `use_mkl=0` 的CPU配置3/3通过，两个测试worker sentinel native CPU、fallback0；独立终审 SHA `67f92e338d0c2fa81a143cb713b70000877032b958e4754b89a22094654ed18b`。原四项失败保留，`use_mkl=0` 不证明默认oneDNN路径；frontend自建子进程的fallback未单独记录。原主树optimizer BF16严格moment差异10/11保留；别名、factory、replay及library已分别按前节记录。

测试断言修正后主树执行指纹 `6c3decf371a2cb7d14b7c224f2f3effaa77817a3cbe5b93d0193891cc93818d2`；与已审CPU native core对应清单只有该测试文件变化，生产和编译源码未变。主树独立CUDA cold 单次cap900已启动：目录 `_state/release-day23/20261029-main-r12-cuda-cold-resources-testfix-r2-hold/`，worker PID 1648007/start_ticks 187120314，GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701`，根审阅 SHA `364fe9f057b84dea7666d8396214ea2970374bd3415c191f74dea8da99c03800`。38个gitignored pyc及对应源码单独绑定，不删除或重写；恢复时先核 PID/start/receipt/lock，禁止重复。CUDA构建与后续main设备、optimizer、replay测试尚无结论，性能验收独立。


## 2026-10-29 16:16 +0800 主树 CPU 定向结果与补测

独立审阅采用实际主树 native core 后，native JVP/pow三项全部通过、fallback0，终审 SHA `43ba3c2e9e2e1fdf4a1feb11f12a92b0939c6dc1400137467495d7590736ec6a`；最初 pytest hook 参数错误的0执行失败与 cold rc1均保留。主树 Torch CPU：replay 6/6，optimizer 10/11（已知BF16下一步moment差异），storage alias 22/22、factory 17/17、library/owner 17/17通过，fallback0。autograd semantics 23 passed/2 failed/1 CUDA-only skip，independent frontend 0/2 failed：其中 CPU fill/zero 和两个frontend均被默认启用oneDNN但 `JITTOR_NO_BUILD=1` 且无本缓存匹配oneDNN产物拦住，不是测试通过；将单独在明确 `use_mkl=0` 的配置补测，并保留默认oneDNN路径未验证。

另一失败为内部 FollowRuntime 测试错误地要求 native `detach()` alias 的 requires_grad=False；历史实际诊断显示该 native alias 创建时为True且写入前后不变，frontend target为False。主树仅修该测试：记录并检查 native alias原值不变，保留 frontend标志、共享地址、数值和dispatch/fallback断言；修改前备份 `_state/release-day23/20261029-main-followruntime-test-fix-r1/`，原失败保留。生产源码/已产main CPU core未改；新测试身份需重绑，不能将后继补测回写为原通过。


## 2026-10-29 16:03 +0800 主树 CPU cold 产物已生成，原终审因 ignored pyc 失败

独立限定审计 SHA `a244fb856a7f467ac2d2b2fa05c17b8e5481c108cde5aafe3d39f9a30d5050de` 确认实际主树CPU core、utils、stamp、227单元及8个生成器可作为 native 后继验证产品；38项pyc都不属于实际编译单元或生成器，header与对应冻结源码及递归编译语义一致，静态对比不能确定哪些曾被加载。原 child rc1、Torch shim 未导入的边界保留，后继须实际 warm import 和运行，不重跑冷编。cache-after SHA `504f145f17e7c15503c8cc709498a2a52d7d07ec6d479898e61e44c883565c97`，可见锁/holder空、拒读未知。

主树独立 CPU cold 实际编译227个单元（190仓库、37生成），core/utils/stamp已生成；child 自然 rc1、非超时，原 receipt SHA `1a832710782011869fc03a488c6796bb4788cd052ed70426a41c9c1c152dadc5` 保持失败。fresh-build 检查38项报错均为 `backends/cuda/**/__pycache__/*.pyc` 不在 tracked+非ignored source清单；尚不能将产物直接写成已通过。已核实38个pyc均是既有被忽略文件、对应冻结 `.py` SHA一致、Python magic/时间戳/大小匹配，按同解释器递归比较编译语义码38/38一致；是否可在限定身份内采用产品仍在独立终审，不重复冷编、不删除字节码。终态组/可见锁空；实际 native/Torch测试仍待运行。


## 2026-10-29 15:57 +0800 Qwen 实用续训通过；主树 CPU 冷构建运行中

r10 native/base + r12 三个 Python overlay 的 seed821 BF16 实用续训已独立终审通过：checkpoint51 实际加载337 tensor均真实 CUDA0、状态值严格一致，112 scalar step与scheduler一致、Python/NumPy/CPU/CUDA RNG恢复；后续51次更新及101个样本 ID/shape与full后半程一致，51×112 gradients finite/FP32/真实CUDA、fallback0，112个LoRA tensor实际变化，checkpoint102全部928数组与连续full相同；验证与predict loss均2.4555516243。此前崩溃的 predict 此次返回，adapter保存/重载、生成及merge在原协议内完成；进程自然rc0、未超时。独立终审 SHA `19d01e4bf02a2595a4522587727c2356d1224d62d32ba352a50800b2da5e9f70`，旧 r10 full/predict 和 resume rc1均保留。该结论只覆盖本配置、mixed源码身份、eager evaluation；CPU BF16 optimizer 10/11差异和通用图内写回未解决，性能/显存未验收。

主树代码已整合相应补丁，自己的执行源码指纹 `647717bf832a74e555e6d3b12f8a9b77f1c530a49ff04322391af99e440ab0d6`；独立 CPU cold 单次cap600正在运行，目录 `_state/release-day23/20261029-main-r12-cpu-cold-docmutable-r1-prepared/`，controller PID 1630556、worker PID 1630567/start_ticks 186981816，根源码审阅 SHA `e5e21fa657fb1861ba004f9df08cba88e5dcf6614827d463594aa6a15c675501`。恢复先查PID/start与receipt，严禁重复；该构建不是测试通过，主树 native/Torch/真实CUDA回归与模型验证待后续单独执行。状态文档例外只用于此构建/指定native，不用于读取看板的结构门禁。


## 2026-10-29 15:41 +0800 r12 混合 CUDA optimizer 回归通过，Qwen 续训已启动

独立实用续训审计 SHA `4d4d6529efe3be362d2a92f452a1ed97812008b31292ddf0a0b149309b9da7e9`：continuous full 与 resume 的 checkpoint102 sidecar各928键、无缺失或数值差，112个 LoRA tensor 自51→102均变化。程序已越过原 `trainer.predict` 崩溃点并创建后续 lifecycle 保存目录；预测数值及 lifecycle/进程终态仍待实际报告。该审计范围仅本次 r10 native + r12 Python mixed、seed821 BF16、真实 CUDA/eager评估，不能推及其他任务。

真实 mixed Qwen 新进程续训报告已保存：从51到102共51次 optimizer更新、0跳步，后半程101个 sample ID/shape与原 full 逐项一致，记录 loss 均值均1.153616764167748；每步112个postclip梯度均存在、finite、FP32、native cuda:0，fallback0，冻结基座未变，固定验证loss与full同为2.4555516242980957。checkpoint模型/optimizer/scheduler数值、337 tensor CUDA设备、112 scalar step和四组RNG此前已实测恢复；practical-report SHA `fbdd09311192df7dd56795d5afd801c83771fc5b1d2c258e825b11ac3216332a`。该直接证据支持此次 mixed 身份的实用续训，预测/lifecycle 与进程仍未终态，不把整个流程预报为成功。

新进程 checkpoint51 实际加载通过：strict hash与functional比较均true，928项值无失败，337个模型/optimizer tensor 均真实 cuda:0/native CUDA0，112个 live scalar step 均51且与 checkpoint/trainer/scheduler 一致；Python、NumPy、CPU、CUDA 四项 RNG 恢复检查通过。后续样本游标、51次更新及终态验证仍在运行，不能提前认定续训通过。

主树已按完整前置字节核对带入 r12 optimizer 设备恢复修复与测试，移植前后哈希及原文件备份位于 `_state/release-day23/20261029-main-optimizer-restore-integration-r1/`。r11 replay 缓解主树备份在 `20261029-main-replay-integration-r1/`。main-specific 构建和运行验证均待执行；CPU BF16 10/11、ACL未验等范围仍保留，不能直接转为主树通过。

r10 native/base 加冻结 r12 三个 Python overlay 的真实 CUDA 定向回归11项、33阶段通过，覆盖 CPU+CUDA 两组 AdamW FP32、BF16参数/FP32 moment、SGD momentum、CPU map_location后迁回每个参数的真实 device、下一步更新与异常原子性；fallback0，独立终审 SHA `f14b9e3616f5f8350c2f8b76cca887b4950aa1558b0a43fc30ef2346e78ec677`。范围为 mixed 产品，不是 r12 native冷编或其他NPU/多GPU的证明。CPU 同源11项为10 passed/1 BF16下一步 moment比较FAIL，原 receipt `3c90b5b760fd42e7f729e82117a0e9f81c95cf3c165c34cda8789091e5ff9767` 保留。独立诊断四场景记录到两侧第二步前参数、梯度、FP32 moment一致，第二步参数一致，moment最大差6.676465e-5/约0.093548%；与一次BF16中间舍入相符为数值推断，尚非已验证根因或长期无影响证明。

基于同源已审核的 r10 full checkpoint51，r10 native + r12 Python mixed 新进程 resume821 已启动，单次 cap1800s，worker PID 1619262/start_ticks 186889301，根审阅 SHA `b8ab39769dc547649b454edd513f8313da0560d2c1b05595207e7107b7fb7fb2`，新状态目录 `/data6/cgq/jittor-lab/_state/release-day23/20261029-r12-mixed-qwen-resume821-eager-r1-hold/`。其目标是实测模型和 optimizer state 的设备/值、step/scheduler、RNG、样本游标及后51次更新；启动不等于恢复通过。旧 r10 resume224个moment落CPU失败以及full后predict失败保持。需要恢复时先检查 PID/start、receipt、锁，勿重复运行。

主树已按单独备份移入 r11 已观测 alias写入时预先拒绝捕获的缓解及测试；main源码身份尚未独立构建/测试，不能转移 mixed 结果。optimizer r12 已按本节增量带入 main，主树验证待执行。


## 2026-10-29 15:36 +0800 r11 真实 CUDA 六项通过，r12 CPU 选择器待修

r11 混合验证使用已审 r10 native 产物及 r11 两个实际加载的 Python 模块，真实指定 GPU、cuda:0 native dispatch、fallback0，六个目标测试与18个 setup/call/teardown 阶段全通过，pytest/worker rc0，实际 item.path 均为冻结 r11 测试；独立终审 SHA `af8f53df14a0feb0c4c496db1f4584141efca0b5deff68a2bc1e92a6a52936ab`。包括已观察 alias 写入时拒绝捕获、别名与旧值、多次不同输入、异常恢复、无写模型仍回放和仅捕获阶段写入的 guard；不是 r11 原路径冷编，也不能宣称通用图内原位写回。原前置 use_cuda 断言失败及文件级 accounting 警告保留。

r12 optimizer CPU 首次混合尝试在收集结束前失败、0测试实际调用：`-k cpu` 同时命中了函数名并选入 CUDA 参数，recorder 因22项非预期中断；原 receipt SHA `9e4388c7c775e701c742285b91e974aed757690dfef436dbd7f8cb9d8de9a0cd` 保留。尚无 r12 optimizer CPU/CUDA 结果，正在按真实参数 device 精确筛选后继续。真实 Qwen checkpoint resume 与 predict 问题仍待解决，性能验收仍独立未完成。


## 2026-10-29 15:32 +0800 恢复：r11 CUDA 启动断言失败，r12 待功能验证

已重新检查现场并 fetch `origin/cgq_transformers`：SHA `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，仍为 main/r10/r11/r12 HEAD 的祖先，不需整合；所有树的既有 dirty 状态保留。上次 r11 混合 CUDA 尝试在收集前 rc1、0测试：worker 在 `import torch` 后错误断言 `jt.flags.use_cuda==0`，而同源真实 preflight 已记录导入后 use_cuda=1；该测试基础设施失败原 receipt SHA `b76b9e01426199dcecf5bd01cf91720284e0bc4abbf0e09c6a94ab7c400872d4` 保留，未证明 CUDA 功能失败，也未证明通过。当前可见范围无本任务活跃训练或构建；不可见进程/锁持有者仍按权限范围核查。

r12 已冻结为 `torch-compat-replay-optimizer-restore-20261028-r12`，fingerprint `3b98d6922e55309fc10342220d1cf5cf7fd7964fc6621fa2132ca8fc306b0206`：r11 基础上按参数 native dispatch 迁移 optimizer moments并保留 dtype，新增 optimizer 恢复测试；尚未实际运行。当前推进修正 CUDA 启动断言的 r11 定向后继及 r12 CPU/CUDA 状态恢复测试，保持首次 JIT 串行。原 Qwen 新进程 resume 的224个CPU moments是真缺陷、112个 scalar step设备检查为检查器误判；完整训练/eager eval 已有证据但恢复仍未通过。


## 2026-10-28 17:07 +0800 r11 replay 写入缓解：混合 CPU 六项实际通过

r10 native/core + r11 两个 byte-verified Python 模块及 r11 测试的明确混合 CPU 运行，6节点、18阶段实际通过、pytest/worker rc0、fallback0，覆盖已观察写入时 alias、旧reader、自动/显式replay拒绝、异常恢复、无写入正常replay，以及捕获时才发生写入的 guard 反例。controller 原 receipt `3812b428bf9983900064d81a5e95c563b841b7bf76a44cedfbab9df64c458743` 保持 `FINAL_GUARD_FAILURE_RETAINED`：14个现有 shim 文件只有mtime改变、SHA/size与canonical source均相同，另新增24个 CPU JIT 文件、无删除；完整保存证据独立终审 SHA `4714dbf888fd3317b1119b2b76623025d2b9073ec145c83a4a5f5932f6ad8a3d`。pytest六个call确有记录但 nodeid文件部分为空，JITTOR 文件级统计报0 executed，保留此统计界限。此结果不是 r11 native冷构建、CUDA验证或任意图内写入重放支持。

optimizer moment CUDA恢复缺陷正在 r12 新候选修复；新进程恢复审阅器单独修正 Python step 标量误判，15项无GPU假对象检查通过，仍未绑定实际新源码或证明恢复。


## 2026-10-28 17:04 +0800 r10 checkpoint51 恢复被真实 optimizer 设备缺陷阻断

seed821 新进程 resume 自然 rc1、非超时，严格模型/optimizer/scheduler 状态 hash 全一致，928项数值检查通过；模型参数/缓冲 113 项仍为 cuda:0，但 112组 Adam 的 `exp_avg`/`exp_avg_sq` 共224个保存时为 CUDA、加载后实测为 CPU。单卡 Trainer 用 `map_location=cpu` 加载 checkpoint，`compat/torch/optimizer_api.py::_load_state_dict_torch` 直接引用这些 tensor 而未迁到对应参数 device，属于实际设备恢复缺陷，首个续训更新、RNG及样本游标尚未执行，不能判恢复通过。另有112个 optimizer step 是 live Python标量，检查器错误地要求它们具备设备元数据，须与真实 moment 缺陷分开修正。

原失败及对照证据保留；独立终审 SHA `07d935cac14d5b1bd29b6ad9de65925f81711b0378a541049c6d4fda4bd1f4a2`。下一步在新候选修复参数对应 device 的状态加载及检查器标量处理，先做CPU与真实CUDA聚焦测试，再从原完整 checkpoint 新进程验证；不抹去已完成的 full/eager eval，也不重置旧T5/Vision预算。r11 graph-replay 缓解测试与 optimizer 恢复为独立功能项，首次 JIT 仍串行。


## 2026-10-28 17:02 +0800 r10 Qwen checkpoint51 新进程续训进行中

基于新 full 实际 checkpoint51，不采用旧缺失文件或 best模型/末优化器拼装状态，seed821 resume 新进程已启动，单次 cap1800s，续训51→102，评估仍显式 eager；worker PID 1214963/start_ticks 178729759、argv 已核对，目录 `/data6/cgq/jittor-lab/_state/release-day23/20261028-r10-qwen-practical-resume821-eager-final-r1-hold/`。根启动审阅 SHA `875a368721158b8b33d794b6871e100b40bf0b6cdf156506ec64a64b81502ba8`，preflight→full 缓存新增1220、无改删的转换审阅 SHA `35c076a5fefc9fdeb3d2e7383547f96ba39c2c1eadcea3cd0eed11fdc58d47b8`；启动前 GPU 空闲39214 MiB，可读锁/holder空，1658个拒读PID未知。恢复时先核 PID/start、原 cache/receipt，禁止重复。

checkpoint portable CPU 文件与112组状态已验证，但真实 GPU 加载、RNG、数据推进与续训训练行为尚无结果。原 full 的102更新/验证证据有效，随后 predict rc1 仍保留。r11 针对自动捕获写入的缓解与主树回归仍待在所有首次 JIT 串行条件下实测。


## 2026-10-28 16:59 +0800 r10 Qwen eager full 已完成训练，predict 仍失败

独立终审 SHA `7a74ecef613a8a2056198b65017e6467b58e137e4ad8578be16afdd9013473f5`：202条 microbatch 的样本 ID/shape 与匹配PT一致，epoch1 JT/PT 1.20546931/1.20568009，epoch2 1.15361676/1.15406398；每步112个 postclip 梯度均 finite/FP32/真实 CUDA0，checkpoint51→102的112个可训练 tensor 均变化，固定验证 loss JT/PT 2.45555162/2.45664406。新进程续训尚未运行；predict 的 rc1 仍是未修功能缺陷。

新 seed821 BF16 run 真实 CUDA 完成 102/102 更新、202 microbatch、0跳步，fallback0、冻结基座未变化；按每 epoch 101 条 microbatch 同口径计算训练 loss 为 1.205469→1.153617，匹配 PyTorch seed821 约1.20568→1.15406。固定17样本验证 loss JT 2.4555516243、PT 2.4566440582，单seed观察接近；三个 Trainer eager evaluation 均返回，step51/102 checkpoint 已保存。`practical-report.json` 在后续阶段之前保存，当前训练+eager验证已有直接证据，仍待独立终审和新进程续训；模型/optimizer的各文件必须审阅，best模型与训练末optimizer不是同一原子快照。

完整进程自然 rc1、非超时：训练和验证后 `trainer.predict` 未处于 eager 作用域，再次触发 r10 graph replay / alias-preserving write 捕获错误；不能写成整个工作流或预测通过。原严格结果、失败 traceback、两次 checkpoint 和报告均保留。r11 只针对 warmup观察到的写入，尚未运行；checkpoint51 portable CPU 对照通过不等于新进程恢复。正式性能/显存验收仍未完成，不能从训练耗时推断性能。


## 2026-10-28 16:50 +0800 Qwen eager 评估新 full 已启动；r11 定向缓解待测

checkpoint51 的9个文件已确认稳定并用独立 PyTorch CPU 与可信 portable pickle 显式映射核验：112组 optimizer state 的 step 均51、scheduler step51、四项 RNG 均存在，926条 sidecar 数值严格一致；保存 LR 与 scheduler/6步warmup线性计划吻合。CPU audit SHA `7480abd38074fd7ac74ab05a3ae6c815c84ffbefa64022b98f10d78cfa6fe232`。原 `torch.load` 直接读取 optimizer.pt 因该兼容层使用 portable pickle 格式失败，属于跨框架读取限制，保留原失败，不称文件损坏；一次日志/保存LR时序误判也保留并以实际计划复核。此为CPU文件审计，不证明Jittor新进程设备、数据游标或续训行为。

新 full 已走过原崩溃点：首次 Trainer eager evaluation 自然返回，inside `auto_graph_replay=0`、`keep_graph=0`，423个参数/缓冲 native cuda:0、fallback0，退出后 replay 恢复1；epoch1 eval loss 2.468442678451538，terminal SHA `7b781e179925ec8f91dcf96be5b0a48f8be897e4b47c576dec1f2ad15a367724`。checkpoint51 的 adapter、optimizer、scheduler、RNG、trainer_state和 sidecar 已出现，保存和完整训练仍进行；文件存在不是已核验恢复，不声称训练/恢复或 graph replay 通过。首两步实际CUDA梯度/更新门禁也通过。

针对 r10 自动 graph capture 验证崩溃，新的 seed821 BF16 full 以相同权重、数据、102更新预算单次3600s启动，仅在整个 `Trainer.evaluate` 作用域设 `auto_graph_replay=0` 并记录真实设备、keep_graph、fallback与标志恢复；这是实用训练的 eager 验证路径，不是 graph replay 修复。状态目录 `/data6/cgq/jittor-lab/_state/release-day23/20261028-r10-qwen-practical-full821-eager-eval-final-r1-hold/`；worker PID 1200873/start_ticks 178667264、argv 已核对，恢复时先查原进程身份和 receipt，禁止重复。根启动审阅 SHA `3c5efd89b500c1c8e74038b6fe2c6eb5762d6c321b4db09badb0b97163cf43d1`；原失败前后 cache 无旧文件改删，新增1200项，与当前 baseline 逐项核对，转换审阅 SHA `82e1b4e13accfb63c17c2a2c3142907dfca368c39612823cef83eb5d8ce57390`。启动前 GPU 空闲33290 MiB，1624个拒读PID未知；实验/正确性 cache 串行，不用于性能。尚无新训练、评估、checkpoint或续训结论。

新候选 `torch-compat-graph-replay-inplace-20261028-r11` 另行实现只针对 eager warmup 实际观察到的 alias-preserving 写入：完成该次结果后拒绝 capture 并返回结果，后续走 eager；保留 native capture guard，不能认定任意状态变化、非连续路径或图内原位写均得到支持。CPU/CUDA定向回归尚未执行；r10原崩溃和日志原样保留。


## 2026-10-28 16:20 +0800 r10 Qwen 首 epoch 后验证崩溃，修复中

seed821 full 自然 rc1、非超时；前51次更新日志 loss/grad_norm/LR 均有限，同口径51日志平均 loss JT 2.387310、PT 2.387724，首 epoch 局部变化受样本顺序影响，不能据此断言两 epoch 下降。验证阶段 graph_replay 捕获执行 Qwen causal mask 的 `kv_arange += kv_offset`，进入 `_copy_into_preserving_aliases` 时因 `state_capture || keep_graph` 被显式拒绝而崩溃；这是当前统一候选真实功能缺陷。Trainer 在保存前先验证，故无 checkpoint51、无固定验证结果和续训证据，不可从该运行续训或声称实用训练通过。首两步真实 CUDA、finite 梯度和实际更新证据仍有效。

原 receipt SHA `c53039bfdc0845d0bbabad842c8c2ec39491d60d4e20e6a8615143f3908867d6`，worker.log SHA `6346ee30ef7cc212e7bd5ebbcf2db1f251b238ea56ffc90413955c8289a6e5fe`，终态 cache SHA `76cee5443414f3a41cb9604c223c0637bdbedfcc6da5a5638de2e5ce8f42a3d3`；可见所属进程/锁清理，1601个拒读PID仍为未知。已开始检查保留别名语义的 capture 拒绝/安全降级或图内写入机制；未经具体测试不能简单改回 assign 或宣称修复。原失败不覆盖，旧T5/Vision预算不动。主树已整合的 r10 代码受同一缺陷影响，CPU cold HOLD 只适合验证该旧身份，修复后须形成新身份。


## 2026-10-28 15:53 +0800 r10 Qwen seed821 full 实际运行中

更新：第2步即首个非零 optimizer LR（约5e-6）完成，112个参数的梯度及 Adam moments 真实 CUDA、FP32且 finite，fallback0，56个 LoRA B 全部实际更新，全部112参数数值变化；56个 A 此刻梯度为零，其变化可能来自 weight decay，不能把变化当成 A 的非零梯度证据。`first-nonzero-update-evidence.json` SHA `efb7194c2b27a48138d6870e57e8806994b86b0a9fd7a5972891a1e326066033`。证明实际前后向与一次非零更新；完整训练、固定验证和恢复仍无结论。

更新：真实 Qwen 第1个 optimizer step 已通过有限值和设备门禁，两个 batch1 CUDA 输入、112个梯度及两组 Adam moment 逐参数均 finite/FP32/真实 CUDA，optimizer 覆盖无遗漏重复，fallback0。warmup 实际 optimizer 前 LR 为0，112参数保持不变符合预期；两条累积口径 loss 0.913073/1.469827。首步证据 `first-update-evidence.json` SHA `d1a106e68094e3fcb27800ce2a1722b07a44d74b2eb2124e09604c63026ffa96`，仅证明首步前反向和零LR状态；首个非零更新、学习曲线与恢复仍待实际结果。

新增 `test_library_saved_alias_version.py` 检查注册的 library backward 保存输入后经 detached alias 修改必须拒绝反向；同字节测试副本在独立 PyTorch CPU 实际1 passed，Jittor 主树尚未执行。FollowRuntime 新测试补充写入前后 stopped-gradient 断言；这些均未替代真实主树构建与回归。

主树进一步按依赖整组带入 shared storage version、反向实际读取校验、no-grad 保别名 copy、fill/zero 目标 placement 及存储回归；既有 dirty 内容前后副本与哈希保存在 `_state/release-day23/20261028-main-storage-integration-r1/`。FollowRuntime 内部回归改用已独立诊断的 native holder swap 构造，其旧错误前提的 r10 FAIL 原件仍保留。15个受影响 Python 文件已做 AST 语法检查，主树冷编译、真实运行、CPU/CUDA 和新回归均待执行，实用训练尚无主树结论。

主树另外逐项带入 SiLU 低精度中间值、autocast cat、梯度 dtype、混合 dtype RMSNorm 和 pow 舍入修复及对应回归，修改前副本、前后 SHA 保存在 `_state/release-day23/20261028-main-numerics-integration-r1/`；autograd 只移植梯度 dtype，存储版本 hunks 尚未混入。主树重新编译和相关 CPU/CUDA 验证均待执行，不能把候选测试当作主树通过。

主树已单独带入常量工厂叶节点修复和17项回归测试：只在 `compat/torch/installers/factories.py` 插入六个常量工厂的 stop_grad，保留原有 dirty 修改，插入前文件与哈希保存在 `_state/release-day23/20261028-main-factory-leaf-integration-r1/`；测试为 `compat/tests/torch/test_factory_leaf_training.py`。此为新的主树源码身份，尚未在主树运行回归，不把 r10 通过结论转移给主树；正在运行的 r10 训练源码未改。其余存储/alias、SiLU、cat、gradient dtype、RMSNorm、pow、fill/zero 改动仍需依赖审阅后整合。

新用途为同源 r10 的 Qwen LoRA BF16 实用训练，单次上限3600秒、102步、2 epoch，checkpoint51 留待新进程恢复；不动旧 T5/Vision 诊断预算。实际 final 在 `/data6/cgq/jittor-lab/_state/release-day23/20261028-r10-qwen-practical-full821-final-r1-prepared/`，source fingerprint `2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22`，launch review SHA `70b4b906596dd40e6583d4a70db7cca2c04c59e25e38853e04b07fd899a650e8`。controller PID 1144682，worker PID 1144693 / start_ticks 178318133，argv 指向真实模型、BF16和 CUDA；进程现正在执行，恢复时先核查 PID/start、receipt 和 cache，禁止盲目重复。启动前GPU UUID `GPU-2be51dc7-d99a-7307-869b-30ed5888e701` 空闲33108 MiB，正确性不以利用率为条件；可见cache holder和锁为空，1625个拒读PID仍是未知范围。

首步 warmup 学习率为0，记录逐参数梯度和 Adam 状态，下一非零学习率步骤检查实际更新；个别非零梯度参数不变保留数组待审，不自动中止。尚无完整学习曲线、固定验证指标或真实恢复，不能提前判定通过。完整 logits 原严格诊断与 CPU focused 原23/24失败仍保留，性能和显存验收独立。


## 2026-10-28 15:48 +0800 Qwen r10 no-training CUDA preflight 终审通过

preflight-only final 20261028-r10-qwen-practical-preflight-final-r2-prepared 单次自然 rc0；同一 PID/start 在物理 GPU UUID GPU-2be51dc7-d99a-7307-869b-30ed5888e701 上实际导入 r10 source与 CUDA core，完整231 compile order和ingredients与已审计产品一致。小张量输入/输出 cuda:0、native dispatch cuda/device0、值[3,5]、fallback初值/增量均0。receipt 24fca00c9d88fdcaefa0fa892d5a7d51ec975d7e82b99fb6714eece711efe97e，result c2b9b9a1e5ad04e2faf532ead4f15424b08fc58a219fa883190d55d43e7a29fd，root terminal audit 3440e3b2b965e387f9317ace71ff430ef0a8846bb4f9cb1efc783dc883852ad7。实际cache常规文件1137→1141仅新增4项JIT、无删改，after SHA 0b3ea7f0062e117bb0a3917d24928234d9d6617ec5c118a70759ebe3dce6bf9d；锁及可见holder空，1624个拒读PID的FD仍为未知，不扩展为全机无holder。此仅小张量身份前置，未加载Qwen或训练。

训练final仍待冻结审阅：首step实际学习率为0，必须记录optimizer之前的LR，零LR时参数应保持且检查梯度/Adam状态，首个非零LR步骤再检查实际变化；batch积累、逐梯度设备/dtype、optimizer组和moment逐项验证，失败先留证。新模型full/resume与任务指标、CUDA dtype补充和Day4–5性能均未执行，CPU focused原23/24 FAIL保持。


## 2026-10-28 15:42 +0800 Vision 历史实训逐格复核

限定的末位差异例子：未版本化 20261026-vision-model-recovery-r2/ 中 ordered-grad 候选的 ViT FP32 seed821（真实CUDA、fallback0）连续61→67与新进程64→67恢复，最终参数hash、40个momentum、9个scheduler字段、eval loss 2.2559596135066107、accuracy 16/65及步骤65–67的30个已保存历史字段逐字节相同；原 same-runtime-exact.json SHA 069194fcae3588ca4456e9a10a5dc8653180504c2b1e46f236846dfc1e85d178 仍为1/769 FAIL，仅 final/eval/logits，max_abs 5.960464477539063e-8，527/650 logits位不同。独立 bitwise-audit.json SHA 1041421904b6503d1204c73737a580303e90bb6c2d3056c25c8336be0efb73a9；未保存每步参数或原始梯度，故只将此单例降为低优先级，不外推其他格。

另有可复现的严格恢复相关归约非确定性：未版本化 20261026-vision-1305/HANDOFF.md 与 20261026-vision-para-ab-1316/HANDOFF.md 记录固定 broadcast反向在默认level3的多次重复产生不同hash，level4与PT各为一种；后者与PT max差不超过2.384e-7。它是确定性覆盖缺口，不能单凭差值大小断定训练崩溃。Vision旧诊断家族余额仍310.095694191s，没有重置或重跑。

r25 是 HEAD 9e0e187b 加冻结 repair 源，source fingerprint 1be46e54d5399286c35ab11a986e2e088931fa983d6eaf0ba38517eb1928bcd5，非 r10；两 epoch 各3137张，100步数据顺序/input SHA/lr及初始权重在 PT/JT 一致，真实 CUDA、fallback0，各100更新；训练 loss 是按每步样本数加权均值，验证固定65样本。原报告/缓存以未版本化 20261020-r3/vision-deterministic-cuda-r25/ 与 20261020-r1/vision-cuda-matrix-r1/ 为准，821 ResNet FP32 PT 为 20261020-r1/cnn-cuda-pytorch-seed821-det-full.json。

| 网络/精度/seed | 两epoch训练loss PT；JT | 最终验证loss PT/JT | 准确率 PT/JT |
| --- | --- | --- | --- |
| ResNet18 FP32 821 | 2.312802→2.217021；2.312916→2.217066 | 2.190382 / 2.190925 | 11/65 / 9/65 |
| ResNet18 FP32 823 | 2.322336→2.220860；2.322480→2.221158 | 2.183609 / 2.184424 | 10/65 / 7/65 |
| ResNet18 BF16 821 | 2.313232→2.217282；2.313635→2.218671 | 2.191783 / 2.189611 | 10/65 / 7/65 |
| compact ViT FP32 821 | 2.299955→2.261349；2.299955→2.261349 | 2.2118864 / 2.2118864 | 21/65 / 21/65 |
| compact ViT BF16 821 | 2.299963→2.261352；2.300069→2.261586 | 2.211983 / 2.211897 | 21/65 / 21/65 |

五格 full/resume实际运行，最终 CNN 62/62 或 ViT 40/40参数相对初始化变化；不能推断每步每参数均变化。ViT 两种配置短程趋势与固定验证指标接近；CNN loss接近而准确率差2–3张，PT FP32双seed自身差1张，65样本与seed不足以判定稳定质量差或一概认定无影响。BF16 CNN resume为8/65、continuous为7/65，原strict 245失败保留，功能恢复执行成功不等于轨迹严格复现；其余四格恢复accuracy与continuous相同，ViT FP32/BF16原strict恢复FAIL100/143，CNN FP32两格同runtime恢复exact、跨PT原strict分别FAIL160/177；其他原跨runtime比较未执行，不写成通过。短期训练证据不能外推长周期或当前r10。


## 2026-10-28 15:37 +0800 r10 真实 CUDA 35 项定向通过

r10 source 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22、真实 A800 UUID GPU-2be51dc7-d99a-7307-869b-30ed5888e701 的 Jittor focused 35/35 pass、0 skip/xfail、fallback0，105 个 setup/call/teardown 全通过，322.43s / 600s 自然 rc0。相同 34 项 PyTorch 同 GPU oracle 先通过；Jittor 多测原 CUDA fill/zero 对显式 CPU/CUDA目标的 alias 和 storage 语义。覆盖 FP32/BF16 六工厂叶节点、前后构造梯度、_like图/存储独立、输入和参数梯度、SGD及二阶，sentinel 实测 cuda:0/backend1/device0/dispatch(cuda,0)。receipt d4d483fcf7e67655472c255f90b9d669dd35ef9dbfc9a02e1cca3812dfab03d3、result 2a66f1fc21ae4e3eb44ff675ff970489b1e0e8d88c04fcce98747ed04751c31f、root terminal audit 6a9a4b56d034ec8e2b3f5bbcc04d7cb21575cf33740617e6d99915259f454530；源码/实际产品/完整编译顺序与环境一致，cache 942→1138仅新增196、无删改，终态 SHA 16257e96394bb034a016392f28028644c841e973f44d655367b604328fd3feaa，组和锁空。CPU focused 23/24 原FAIL仍保留，内部 FollowRuntime补充独立结果见下节。

这只证明有界算子与更新功能，不是 Qwen/BERT/T5 的 r10 实用训练通过，也不证明长期训练或性能。Qwen 新 r7 按此真实产品与缓存准备无训练 preflight，训练与日志另置 state、同功能 cache 串行、性能 cache 独立；preflight、模型加载、LoRA full/resume 及任务指标均尚未执行。真实checkpoint映射、scheduler/RNG/数据游标仍须新进程验证，旧严格结果不改。


## 2026-10-28 15:30 +0800 r10 CUDA 35项定向运行中

独立 PyTorch CUDA 34项对照与终审见下节。r10 Jittor 同源 CUDA focused final 20261028-r10-cuda-factory-focused-final-r3-prepared manifest 3c4101d313d6840cd5c3880bedbdd19acb1f0d67efae54a59c5e6033da84297c 已绑定实际 CUDA cold/core/942缓存和 oracle，root launch review ea2c43f47bf7c6d5d86581da33c47d0c359bcf29e707d6c4fefc0f0cf32e4d5d。单次 cap600、controller PID1120719、worker1120721、同UUID已启动，实际进程身份已记录，正在编译算子；恢复时先查 receipt/PID/start/锁，禁止重复。34项工厂/输入梯度/优化器/二阶/断图和原 CUDA fill-zero alias共35节点，尚无终态，不预填计数，不声称训练/性能通过。资源部署只允许14个精确路径和内容哈希，外部测试包不自动加载仓库 conftest，实际环境漂移仍记录失败。


## 2026-10-28 15:25 +0800 独立 PyTorch CUDA 对照 34 项通过

原 oracle 首次因 pytest recorder hook 参数名错误在收集前失败，0项执行；receipt 0e38282b48d942669bdb86812e1ab90a55990839fb4595b86f70586b57c31b71 和独立失败审计 4e8ffeaca525de3e2fe02b92d7cb2266036a01a1f9eebf68ada38cf319a2a626 保留。新 hookfix successor 使用完全相同测试字节 d2b99208885b39070a20ac986f6e1f09f40fb91e45919b241172d5719e9c3175，在独立 PyTorch 2.6.0+cu124 和指定物理 A800 UUID 上 34 pass/0 skip、102阶段完整，前后 CUDA 同步，28.22s自然 rc0；receipt 38b6ee43cd60dd34b0f1ae115261f899038e8b8c68a5fd2579b02600237284fb、result 0d7ed1c362c8187e66ef5c24df8cb8fdb16aec0a13d346b614ad5d1dcf0d4406、root terminal audit 75a03b0b1f3e17413f3531b6c4c5d8e244459061b03d07870a70262552e07d59。此仅是独立 oracle 通过，不是 Jittor CUDA、实训或性能通过。

下一步把该 oracle 终审和真实 r10 CUDA cold 产品/942缓存绑定到新 Jittor 35项定向包，核对真实 CUDA dispatch、梯度、SGD、二阶、alias及 fallback0，再逐项审计；尚未启动。Qwen r5 仍 HOLD，SIGTERM、恢复有界功能对比和身份均只有无 GPU dummy 验证，真实 checkpoint、新进程数据游标、训练产品与模型仍待实测。


## 2026-10-28 15:19 +0800 r10 CUDA cold 终审通过，数值待验

r10 source 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22 在物理 GPU UUID GPU-2be51dc7-d99a-7307-869b-30ed5888e701 上的新 CUDA cold 自然 rc0、524.12s / 900s，231编译单元（194仓内、37生成），Jittor 与 Torch 导入后均 has_cuda=true/use_cuda=1；未创建 tensor，不能当成真实 CUDA 数值或训练通过。receipt 49bc5fa678b9e5716fc8cd50a89ea3927990d721a28c933aba8af34197559f97，child a1767d8d3a08e83aff3aabfa3a4b3e04af0495dbaad4eefd23cbfe6a5bd77c17，实际 942 cache c58b6866cb55ce273d487f84306fcb656a18c756a7a80699e46f1b2409f10d21；root terminal audit a3d5c1c1c9c862bf232495a4af014d78a53363e320b8a384c9863e20ba8b7340。初始 child 命令采样为空，存活期独立身份证据 2d8ae797e4a894ffe645b93ded82fe4c139fb7e044b6d7488076ff614cfb742b 核对相同 PID/start/组/实际命令，原收据不回填。

下一步在同物理 GPU 对 CUDA 常量工厂/输入梯度/SGD/二阶/断图的34项执行独立 PyTorch oracle，再绑定实际 CUDA core与终态执行 Jittor 34项+原 CUDA fill/zero alias节点。实训先决条件和性能验收均未完成。CPU core 的历史 dtype skip 不直接表示当前失败：同源 Torch promotion 测试实际覆盖旧16个 dtype 对中15对和两顺序，int8+int32仍需直接补测；.long()小值/负数在 CPU 正向测试通过，大数和真实 CUDA仍待验。


## 2026-10-28 15:11 +0800 r10 CUDA 冷构建进行中

实际 CPU core 终审已通过（见下节）；r10 同源 GPU UUID GPU-2be51dc7-d99a-7307-869b-30ed5888e701 的新独立 CUDA cold final manifest 9a8fe2efc96f9787fae831c77dcb0fc4a762ecc1792e700a7c6ba1cc4c7dfd36 已单次启动，cap900；root launch review SHA 60a23df1d39eb6bbf3b34a7ba880b6f100813a628e9ac45b850ec76d15fbaca5，入场可用显存54024MiB，未以利用率设门槛。controller PID1096115、child1096119，原 receipt 初采 child command为空，存活期独立 identity 20261028-r10-factory-leaf-cuda-bootstrap-final-r1-prepared/root-cuda-cold-live-identity.json SHA 2d8ae797e4a894ffe645b93ded82fe4c139fb7e044b6d7488076ff614cfb742b 已核对 PID/start/命令/私有组。当前正在编译；不得重启，恢复先查 receipt、PID/start、子进程及锁。冷构建成功只证明产品身份，真实 CUDA 34项工厂/梯度/二阶及原 alias 定向、实训均未执行。

Qwen runner r4 的无 GPU dummy 覆盖 normal/timeout/异常/SIGINT/另 session后代和训练证据先保存，但独立审查仍发现 SIGTERM 清理和 exact signature 误阻断实用续训问题，正在新 r5 准备，不能运行训练或以 dummy 代替实训。


## 2026-10-28 15:08 +0800 r10 十文件 CPU core 终审通过，准备 CUDA

r10 source 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22 的原维护 Torch core 已一次自然完成：230 collected/called、227 pass、3 skip、fallback0，450.22s / 600s；十文件实际全部执行，新工厂和 autograd_semantics 定向方法不在此计数。三 skip 分别为 CPU 无 CUDA、历史 .long() dtype 和混合类型提升标记，后两项是跳过声明，不自动当作 r10 已复现。receipt e5b751e11bdee16dfaae0c446ceaebafa249f61eb8571311d8cf892195ab0163，result dd98aa4a4875c4a92e09816d59895180ee94428361bdcfc684473c7f9063dd4f，root 独立终审 a5e2b3f1b71906e9e79c3c02b404cac9ae0a6f5d813de1556ae3107c7d6e0941；源码2807文件、core/utils/stamp、环境、进程身份与实际7314 cache核对，6590→7314新增724、无删除或内容改变，14个shim部署文件仅mtime变化。原 focused 23/24 FAIL和独立 FollowRuntime补充诊断分别保留。

恢复后再次 fetch origin/cgq_transformers 为 e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，main、验收/修复树、r9、r10均包含该提交，无需整合，已有 dirty 保留。共享 GPU 有足够显存但负载繁忙，正确性可在显存足够时进行；不能作为正式性能窗口。下一项审阅绑定同源 CPU 证据的新 CUDA cold final，单次900s并串行启动；CUDA前反向、factory/别名/梯度和实训尚未通过。Qwen 新 runner 和控制器仍 HOLD，dummy 验证中的元数据问题要保留并先修，严格生命周期结果单列，实训数据先保存。


## 2026-10-28 14:10 +0800 FollowRuntime 内部 CPU 补充诊断通过

r10 的原 23/24 Torch focused 收据仍为 FAIL；唯一失败在写入前的对象构造断言，不回填。后续单次 120s 新诊断以内部 swap 构造真实 Torch frontend、placement=-1 的停梯度 holder，且与 native alias 同址；_api_fill(2)/_api_zero() 均保持返回自身、placement、共享地址与 alias 值，真实 CPU dispatch、fallback0。常规 frontend array/empty/ones 均实际发布 CPU placement0。该补充仅证明内部 swap 路径的 CPU 停梯度写入，不证明公开构造、CUDA 或模型训练。receipt 19ebd9cdc92d0f4500311f551fe0993558a1a6370d8b8aeebb08749f53d79b4b、result 5b4a4008a509ba762ae01e616bd9c830643b373f156076a20d3d3b4dfce9d6f3，独立审计 dd32272e6257e81f313d63da2f14ef02f675369c51ff547cdb7a2af0eb38117c；cache 6590 无增删改、终态 SHA 17752785201408dd9b8706831cd87c1eda77b9f7ef25142c498fd5df8424034a。下一步依据 23 项实测与此补充审计绑定原十文件 Torch core；原失败与不在 CORE_FILES 的新增测试单列。CPU core、真实 CUDA、实训和性能仍未验收。


## 2026-10-28 14:07 +0800 r10 Torch CPU 23/24，内部 FollowRuntime 前提失败

修正 selector 的单次 Torch focused 自然完成 331.76s：23 pass、1 fail、0 skip、fallback0。17 项独立常量工厂测试（构造时/后启用梯度、_like 图与存储独立、SGD、二阶梯度）及原 CPU fill/zero 的别名/值/输入和参数梯度回归通过，旧 leaf 回归、独立前端和相关构造器通过。唯一失败是内部 FollowRuntime 测试在调用写入之前，构造的前端张量 placement_backend 为 0 而测试要求 -1；该结果不证明写入失败，也不证明该内部路径通过。原 receipt b0284aa30c8f83e6692cbc360e2152fb2258bc2d27bc9db827e15cd9aa77fab3 保持 CPU_FOCUSED_FAILURE_RETAINED，result 5c290baaff40c21243aaec3c91ac493cd096393e2dafc06cba86759c9a3b30f3。独立保存证据审计 90498970762652b007f87766bcabcb1d2d1a015bb9d0c43ba78f46bd37770bb4，仅采用原真实 23 项与失败边界；actual cache 6590、SHA cf16c05b253a3c632371d8a453863b61b4210e127ff3c3c40fb83679459f4fe9、锁及身份有效。

下项只准备有界真实 holder 构造诊断，区分不可构造与内部写入缺陷；不改原 FAIL 或反复整组。r10 未采用 main，CPU 原十文件 core、真实 CUDA、实训和正式性能仍未完成。Qwen 后继保持 HOLD，需先持久化实际训练/验证证据，再将严格生命周期检查单列，并补异常退出进程清理与 checkpoint51 状态审计。


## 2026-10-28 14:01 +0800 r10 Torch focused：修正 selector 后运行中

首次 Torch final 因旧内部测试名称在收集阶段失败：pytest rc4、0 collected/0 executed，原 receipt dfc5812f94416a45ea05b60c2f56665df07ea310738ddde15ec681769eee667d 保留，不能解释为源码通过或失败。独立审计 2afb9dae163acd7a58305338ab5644b688891c89927462af4542a0eebf7889a1 确认实际 778 cache 及唯一失配名称；源码/产品未变。新 selector successor manifest ffb6e2e8d995949b29851687eeec6a211efbf9a47eb8b55dc89a8b7850e1e272 只修正测试名并绑定该真实终态，单次 cap600 已启动。当前 17 项新工厂测试全部通过，后续 fill/zero 回归在编译 oneDNN，整组仍无终态；不要重启或提前声称通过。进程 controller 1023968、worker 1023974，恢复时仍需核对身份/receipt 和实际锁。CUDA、模型实训与正式性能未验收。


## 2026-10-28 13:58 +0800 r10 原生 JVP 通过，Torch CPU 定向进行中

r10 source 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22 的 CPU cold 385.86s 自然 rc0，227TU、732 cache；receipt 656c802ff699fb0b3224962d417964544a04735e6d11ef492ba212cdc11bb32d，独立 terminal audit 990f9c3f59c856dfcf0367cfc04797fc5cde1aba206631262a8fd15e6f3a3b3e。仅构建身份。继而原 native 多输入 JVP 旧反例 1 pass、0 skip、fallback0，13.71s；receipt 2462b27c692b1714c736c92a8ad27a58cde23906a177d8f3a7e10de2338b87f5，独立审计 6e0c4f7f6b535444a58169cccc0ea6455a90807ef34f61f33ea61503c57e3962，终态 cache 764。此结果仅 native CPU，不外推 Torch 或 CUDA。

已绑定真实 native 终态启动一次 Torch CPU focused（cap600，final manifest b4ad82cc3b617ebc3b2d35212d34d1cb8c4fbc76a5fb7dfaa0d8a728f5395ceb），涵盖 17 新工厂测试和 7 项别名/leaf/前端/构造回归，仍在执行；先核验真实进程与 receipt，勿重启。Qwen 训练后继 20261028-r10-qwen-practical-training-r2-hold 有控制器静态实现但仍 HOLD，尚无 CUDA 产品/功能绑定及动态审计。实用训练和性能显存仍未验收。


## 2026-10-28 13:54 +0800 r10 CPU 冷构建进行中

r10 仅常量工厂叶节点修复、相关回归和内部 FollowRuntime 检查，独立源码指纹 2b8fa30420b413fc2736f8c29b509bc37ce09387ddca1f37db71f064636f4c22；source manifest 20261028-r10-factory-leaf-source-r4/candidate-source-manifest.json SHA 9e059ac7691e39121d432c7fb68e51851d9243c59d32ca53226c84fb1b02514a，root 静态审计 SHA 7c05f2f9607bd442e2fe696d04e10b9005d299dd5d344270557c3e04cbc3aeab。新增 17 项测试在独立 PyTorch 2.6 全通过（oracle-r2.log SHA 41c3aee667fd6fd4d97f0094c8388df337b2aa54bc6c58df49c00c10289a7a65）；Jittor 尚未验证。CPU 单次冷构建包 20261028-r10-factory-leaf-cpu-bootstrap-r4-prepared manifest d45f0c3c4c19c78264aa3622bc7370f11958858d8adf299ec309cce117cd3d10，cap600，controller PID 1009630、child 1009635，仍在实际编译；勿重启，先核验进程身份和 receipt 终态。完成后先独立审计再绑定 native JVP 与 Torch focused，CUDA 和训练未启动。后继 Qwen 草案 20261028-r10-qwen-practical-training-r1-draft 为 HOLD，不具可运行训练控制器；原 strict 失败及 r9 不改。


## 2026-10-28 13:42 +0800 r9 工厂叶节点梯度缺失已复现

新 21 个独立 CPU 进程诊断自然完成（113.36s / 600s，fallback0），receipt SHA 0eb5633396f1683e392dbad25321718997c076152a59826ad593c892d8db44ec，未版本化目录 20261028-r9-fill-zero-grad-diagnostic-r1-prepared/attempt-01/。18 个 torch.ones(..., requires_grad=True) 案例包含无写入/fill/zero 与六种 alias/物化状态，全部 grad=None；无写入时创建即 is_backward_leaf=false、未登记。fill/zero 后可变为 leaf，但仍未登记。3 个 Linear Parameter/SGD 对照的参数梯度全 1、lr0.1 参数逐项减 0.1；相同案例中的 ones 输入梯度均缺失。这是 KI-COMPAT-004 的实际功能缺陷，不能继续描述为仅信息查询；也不能据此推断所有参数漏更新。已有结果仅 CPU，不能外推 CUDA 或模型训练。

源、产品、进程身份、零 fallback、6594 文件终态及锁已独立只读核查；cache-after SHA f1364c252582bdc80f5737717a373ca1427547a17480936cb90b597405856bc7，6586→6594 仅新增 8。新 r10 隔离工作树正在从 r9 构造，拟仅修 Torch 常量工厂输出叶节点并验证真假 leaf、grad、SGD、原 JVP 和真实 CUDA；尚无修复结果，不修改 r9 或覆盖原失败。


## 2026-10-28 13:35 +0800 实用训练与历史证据复核

历史 BF16 训练绑定各自冻结候选，不能转移到 r9。BERT 两 seed 各真实 CUDA 100 更新：epoch microbatch loss PT 0.346→0.197 / 0.335→0.190，JT 0.349→0.232 / 0.324→0.201；验证 loss 均值 PT 0.53521545、JT 0.45265336，accuracy 0.7878788 / 0.8030303。旧 15.42595% 差为 JT loss 更低，样本小不足以宣称更好。T5 各 100 更新：PT 1.686→1.476 / 1.661→1.493，JT 1.690→1.484 / 1.660→1.497；验证 loss PT 3.05906 / 3.04997，JT 3.06212 / 3.05675，双方 exact match 0。step50 恢复后各续训 50 更新，strict 各 771/2088 FAIL 保留。Qwen LoRA 旧 source 441ddd71f35d22da9826fdfcdb699110e3087eae8b621562411a1049e3a5c342 两 seed 各真实 CUDA 102 更新，112 adapter tensor 全变化；epoch loss PT 1.20568→1.15406 / 1.19540→1.18008，JT 1.20514→1.15379 / 1.19489→1.18070；验证均值 PT 2.45661926、JT 2.45599473，checkpoint51 新进程续训 strict 各 0/930 FAIL。真实 activation recompute/显存节省尚未证明。三任务 fallback0，原 microbatch 梯度 missing/nonfinite 均零；训练均值是累积口径，不混同验证 loss。BERT/T5 为 197/33 样本、batch2、累积2；Qwen 101/17、batch1、累积2，均两 epoch。原件分别为未版本化 20261020-r3/hf-resume-acceptance-r10/ 各 report 和 20261026-qwen-causality-1300/lora-campaign/qwen-adjudication.json。旧 22 个 model/adapter 与 optimizer 文件仍缺。

用户恢复授权，优先真实设备训练、有限梯度和更新、学习曲线、验证指标及功能续训；原 strict FAIL 和预算保留，性能显存独立验收。fresh fetch e5cffe1ae60c418f53c5bb1cdca9182577fd2b81 为各相关 HEAD 祖先，dirty 保留。r9 source 8afd3ed550f5cf113c67c0b4f7c159334781047bfda7e11232c9ab3184e7089d 保存 native 两节点独立终审 2 pass、0 skip、fallback0；2806 源文件、实际产品、740 cache 和精确锁核对。审计未版本化路径 20261028-root-r9-resume-audit-r1/root-native-terminal-audit.json，SHA fb26aee6e3cb6498d57962c6b564f06d6315939ce4318ab15bc1814783a90950。Torch focused final manifest 440072a5b064cbd3b3f5b55fb4114be8b19708a8d369e728068f5f639ecfce21 单次运行约 313.7s 终态：2 fail、7 pass、1 CPU capability skip、fallback0。普通可训练 tensor 写入后 target.grad 为 None，需无写入/写入及别名对照；FollowRuntime 测试在写入前因 torch.Tensor(native) 被构造为 CPU placement 而失败，其测试前提须核对。独立 frontend、RMSNorm4 和 public pow2 通过。原 receipt 5908db0e91b8d8e44c5f01dfbe0b7e0461d3069ab999d9fba0f421206aa60d8a 保持失败；11 条 MKL .o.d 目标为实际观察到的独立 frontend 子进程 PID 982482，合同预期 worker 982201，原 metadata 错误保留。源码/产品及 740→6586 cache 和锁的独立证据审计为 20261028-root-r9-torch-focused-audit-r1/root-torch-focused-evidence-audit.json SHA ad786152422b2c5465380fbccbeeee4dfd9979667953a36d30ad8c401fb4e5a4；仅允许将实际终态作为后续诊断基线，不将测试或原 metadata 改为通过。r9 实用训练和 Day4–5 性能尚未验收，发布 Not complete。


## 2026-10-28 13:10 +0800 用户暂停：已完成项与未关闭门槛

用户要求“暂时先暂停任务，汇报已经完成的以及尚未完成的目标”。线程goal现为 **paused**；整个Day2–5发布验收仍 **Not complete**，没有缩减原目标或放宽门槛。本次只核验保存结果/现场并更新三份既有文档，未继续源码修复、启动测试、模型、性能测量或等待队列。未发现可见同UID范围内匹配本任务路径的活跃进程或JIT锁FD，r9精确锁inode33030298无内核持有记录；4个权限拒读PID保留，不能声称全机进程绝对空。agent列表仅root，无活跃子agent。

暂停快照 `20261028-user-pause-1310/PAUSE_SNAPSHOT.json` SHA `b564c8439e623e300f5e3e9d37def655593bae1b5eb7b37d8a00b0905ae0312b` 保存六树HEAD/porcelain-uall状态hash、进程与锁观察、关键原件hash及下一阶段HOLD；三文档更新前副本也在同目录。main仍HEAD3dedae001372b6be106ae6700bcf10ebc7faebac，r9仍HEAD765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe/source8afd3ed550f5cf113c67c0b4f7c159334781047bfda7e11232c9ab3184e7089d。此次只读取本地remote ref仍e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，未另做fetch；恢复时重新fetch并按规则整合，不自动合并2.0-refactor。

最新执行证据是 `20261028-qwen-fill-zero-r9-focused-cpu-native-r1-prepared/attempt-01-native-focused/`：单次120s cap内自然rc0，正式receipt状态PASS_NATIVE_CPU_FOCUSED，SHA `da8ece3048fad10c74056b3bd79e0cc6db558b9d9936a02ea710fcb94f1d1b55`；result SHA `b3612b3e1f627c52ffa2240d7db55c0354ada6099ca26e0a6312a1f95d41407d`。实际仅运行TestBinaryOp与TestBinaryOpCpuFp16的test_pow_rounds_before_widening两个节点，**2 passed、0 skipped、fallback0**，原生模式未激活Torch，source/products/env guard通过，worker与controller已退出。保存缓存732→740，仅新增8，changed/removed0，cache-after SHA `13ee1d9a7afe2eff3ff0899948f1337df82064a44625558294b795933955da2d`。这些是实际执行收据；独立root保存终态审计仍待恢复后完成，本次暂停不补做或绕过此前置。07:19所写“下一项native两节点”已被本条终态替代，不能重复运行；fill_/zero_的Torch数值/别名/梯度尚无r9执行证据。

已完成的局部成果包括：保全原始失败和多树dirty工作；隔离候选的storage/lifecycle、SiLU、cat及梯度、mixed-dtype RMSNorm等已取得各自范围的CPU/真实CUDA定向证据；CPU pow融合精度泄漏取得old-fail/new-pass与consumer-first测试盲点控制；r8 Torch core唯一fill_失败完成四项最小复现，r9显式目标placement继承修复已冻结、审阅并完成CPU冷构建。r8 native core103pass/45skip/1xfail经保存审计采用，原metadata失败不改写；r8 Torch core226pass/3skip/1fail仍FAIL。结构门禁历史1396pass/4skip和布局通过是维护证据，各自源码绑定及skip边界保留。

| 原目标 | 暂停时结论与缺口 |
| --- | --- |
| Qwen BF16 / LoRA | 最新有效原完整no-hook仍8,690,528/29,171,712超原atol0.005/rtol0.03；局部修复未关闭模型门槛。r9 Torch focused/core与真实CUDA未执行，LoRA full/resume未启动。 |
| T5 BF16恢复 | 双seed strict各FAIL771/2088。ordered-sum primitive/public norm局部稳定性和初始化修复有证据，模型singlemicro仍超时，无完整梯度/比较；原1800s家族余51.29068389162421s，不重置、不自动重试full100或从r66/r70恢复。 |
| Vision | 原五格strict仍FAIL；Lt进程选算法差异有局部控制及候选准备/执行记录，但无合法最终continuous与恢复验收闭环。最新文档余额310.095694191s，旧1289.681100597838s不得当作现余额。 |
| BERT BF16 | 原双seed任务gap15.425953905%超过15%；Linear输入梯度诊断不代替任务修复或重验。 |
| checkpoint审计 | 11个原checkpoint目录存在（BERT5、Qwen LoRA3、T5 3），22个model或adapter与optimizer原文件缺失；不可把目录存在写成审计完成。 |
| Day4–5性能/显存 | 真实AdamW更新、独立cache、同步热态10样本及allocator/NVML采集已有诊断；r3共享负载，所谓near_capacity仅两轮且未证明真实容量边界。旧等待及后继readiness均已超时，无正式稳定三轮配对、容量和峰值结论，也无已验收的退化修复。 |

恢复入口只做下一未完成项：先重核现场/source/cache/remote，再审计上述已保存native结果，不重跑r9冷构建或native两节点；在新Torch final绑定真实740终态与root审计后，按原cap单次运行Torch focused，再维护原10文件Torch core。新增三方法不在CORE_FILES，focused和core分别报告，不预填计数。CPU前置通过才准备同源r9 CUDA、真实定向回归和原完整no-hook；LoRA仍须等待完整门槛。未决定的MKL依赖/key元数据变动审阅策略仍未批准，不能把完整记录或controller rc0直接写成cache合同通过。所有原失败、源码/工具身份、缓存、checkpoint与预算保留。

## 2026-10-28 07:19 +0800 r9候选已隔离修复，CPU冷构建身份通过

新候选 `worktrees/torch-compat-storage-silu-cat-grad-rmsnorm-pow-fill-zero-20261028-r9` 保持HEAD765f1791，完整source fingerprint `8afd3ed550f5cf113c67c0b4f7c159334781047bfda7e11232c9ab3184e7089d`；manifest `20261028-qwen-fill-zero-r9-source-r1/candidate-source-manifest.json` SHA `67e71aa4e9f669010e1e60b27fc15ee461c5184fc19009c4822d1cf67d970140`。r8全量2807项复制并独立重hash后只变四文件：method_api仅对显式target进入现有like-scope构造fill/zero RHS，legacy保持原路径；autograd_semantics新增CPU aliases/地址/值/训练标记/Linear前反向、FollowRuntime alias及独立CUDA回归；native/public两条pow先读取consumer。C++原实现与原独立前端失败用例未改。root源码审阅13项全部通过，SHA `35532b02fbab3e5ffc3bf4e31dbd854e1f70b54aee693581affbc8d32cc7977f`；这不是执行证据，r8与main未采用新改动。

新包 `20261028-qwen-fill-zero-r9-cpu-bootstrap-r1-prepared/` manifest `39ab2d8fa480aa92255326da86396414685223ae8876dca6a186ec67db7ae64d` 已在原600s单次cap内自然rc0，耗时337.69353580474854s；正式receipt SHA `cfd4752d2e315d19c15d5ed8db336e169c2bec11159744e7ad608d5f8fd108f0`，exec43630已收回。实际227TU（190仓内、37生成），core SHA `1562721b0867b0218f564ce23a2faaabe3faf7d98b7b874efdbea54b437df4a6`、utils `083f5111924a69c5d8b81019851ea14b8462d2531f57af015d57ca544accad2b`、stamp `e6d41524e27df9978820bbf2d512459fe2bec4b62d0d0af2af4cd2b6c93594dc`。root独立13项终审全真，SHA `a1d1645e5e75e243b44f0562965c0f5864f14d2d0e64fbeb2ba5c35b143b87e9`；完整环境/实际CPU/source/产品与hash链有效、732缓存逐项匹配、自有组与精确锁空。存活期真实PID/start/命令另存，不以旧PID推断状态。本cold未创建tensor、运行pytest或模型，不能记数值修复通过。

后续仅定向native consumer-first两节点120s、Torch focused10节点600s（含专门CUDA方法的CPU capability skip），然后维护原10文件Torch core600s；首次MKL扩展编译计入正式回归，不另行预热。r9新增三方法不在维护core10文件中，因此focused结果单列，不能凭静态计数预填core233。实际绑定产品和终态后才运行，CUDA/full-nohook/LoRA仍等待前置；其他FAIL、原预算及Day4–5稳定三轮/真实容量/峰值门槛保持，发布 **Not complete**。

## 2026-10-28 07:02 +0800 r8 Torch core终态失败与fill_/zero_最小复现

原Torch core已结束：**226 passed、3 skipped、1 failed in226.19s**，唯一失败为独立前端 `test_independent_tensor_installation_preserves_native_type` 中普通Linear子类weight的 `fill_(2.0)`。正式receipt SHA `641967f6cac11156ee78466baccfa5d213f8b8deb594cfef8457ad0dc593fff6`，缓存终态7919、SHA `6d4d686f426b53928baea887095da53bea9a7b5415797e1a5dceec642ab040b0`；root失败审计确认原source/core身份、fallback0、选集与原600s cap。一个既有MKL JIT组四文件重写，cc/so字节未变、mtime变，依赖/key内容变；原执行与cache合同失败状态保留，不能记CPU前置通过，也不启动旧CUDA HOLD包。

新 `20261028-r8-fill-zero-placement-cpu-r1/` 复用真实r8 CPU core与上述7919基线，在新120s单次上限内自然rc0；没有numpy/sync/data_ptr取值请求，没有修改公开方法。Linear weight和显式CPU `torch.ones` 各做fill_/zero_，四项全部复现相同storage-preserving RuntimeError。目标raw backend=0/device_id=-1，独立严格复刻当前方法RHS为backend=-1/device_id=-1；shape/dtype/contiguous和公开cpu及实际dispatch完全相同。两种构造的目标metadata失败前后相同，fallback0，缓存零增改删、source/core/env guard通过、私有组与精确锁空。receipt SHA `bb10a3031bd7a520f79a0c01ecb77eb45e4d4ef5be52cc455e3a83729b7ad32f`；`saved-placement-audit.json` SHA `b6c84dff2dcf986b5ebe38cda43b5cf9d2fec92116662c830d95b5f8c605a064`。mirror不是对方法内部对象的截获，该观察边界保留。

静态调用链与实测相符：独立Module构造scope退出后，`_api_fill/_api_zero` 裸用native ones/zeros创建FollowRuntime临时值；`_ip`的public-device检查通过，但C++严格raw-placement合同拒绝。最小修复正由新r9隔离候选准备：显式target才用既有frontend like-scope构造临时值，legacy FollowRuntime保持原路径；不削弱C++合同、不改成assign回退。同时只在新身份应用两条已验证的consumer-first pow测试顺序修订。修复、同源CPU回归与真实CUDA尚未执行，发布 **Not complete**。

fresh fetch确认origin/cgq_transformers仍e5cffe1a，为main/原验收/修复/Qwen诊断/r8五树祖先，无需整合；main HEAD3dedae001372、99项dirty，原树51、修复85、Qwen87、r8为116项。快照 `resume-worktree-sync.json` 另存。根盘82GiB、data6约6TiB；没有本任务残留测试或JIT锁。单次GPU采样全部共享忙，不能构成性能窗口，未创建waiter。原完整Qwen FAIL、LoRA阻断、其他FAIL/预算和Day4–5门槛不变。

## 2026-10-28 06:47 +0800 CPU pow消费者优先old-fail/new-pass已成立

最小三链控制 `20261028-qwen-pow-r7-r8-consumer-first-cpu-r1-prepared/` manifest `5631a101c184db8492392d02421047c0b21bab99e4c47a70cfa0d92461b1fc22` 已由root分别单次执行r7→r8，每侧新120s、均自然rc0。r7 receipt `b37804a70aa81ae10b68c862137f1074d4455966e361761846f50f3786e655c0`，r8 `1f1c8d458ea0508dde6ee8133312929b9ccb55b71104a65d69ac33683d9b3906`；两个exec25218/61884已收回。先保存消费者、再读取power时，旧r7的native/public float64 widening各62/64项不同（max_abs0.010855744068976492），public倒数25/64项不同（max_abs5.960464477539063e-8）；新r8三项全部0。三个独立float32 power数组在old/new间逐字exact，故该干预改变的是声明输出边界被融合consumer观察到的精度，而不是更换保存power数值。

原native物化优先顺序在两侧都0差，直接证实新增native测试盲点。root15×2独立保存终审通过：`20261028-root-pow-r8-continuation-r1/root-r7-consumer-first-terminal-audit.json` SHA `1575475bc7b77ebdc5b30daab22426e45588ec85c3828cb6f3125108dd97d6f0`，r8对应文件 SHA `fde3a7c801dc9f4e1623ffac37ba09e882dcff51497a92e01af34a4b0c7dec5f`；paired4 `root-r7-r8-consumer-first-saved-pair-audit.json` SHA `2dda56a5d0a62599beef586d794910eabc556bad3154178ec7e9d7f92828c26a` 确认old-fail/new-pass与power exact。实际CPU dispatch/fallback0、完整环境、source/products/真实命令与原caps有效，r7缓存816→840只增24、r8 7143→7155只增12、无旧文件改删，组/精确锁空。r8新基线SHA `3ca6a79ff2c2a46847d8fb8bd778647274507fd4f61e075645052230229fd1cb`。首次保存数组审计因系统Python无NumPy退出1，记录在 `saved-array-audit-interpreter-failure.json`；只换已安装NumPy的Python复算保存数组，未重跑任何runtime。

两测试方法的consumer-first修订已离线形成 `20261028-qwen-pow-regression-order-proposal-r1/corrected-regression-order.patch` SHA `65091e6e2491ca4a1ef976d26a16d22b143a07ebda2e00bfdf23ea37bb08b55c`，AST及独立副本apply-check通过；未应用到已执行r8，后续新源码身份采用时再合入。该控制验证CPU pow修复与测试顺序问题，不证明完整Qwen因果或发布通过。下一项在真实7155基线上独立Torch core600s，CUDA900仍HOLD待其终态；候选未采用main、LoRA不启动，原完整no-hook FAIL与其他FAIL/预算/Day4–5门槛不变，发布 **Not complete**。

## 2026-10-28 06:40 +0800 r8 native core实际完成并获独立采用，原工具rc1保留

root审阅的native final manifest `d66e2e0e11fcd6072e14787e4660a60ddd72dff95b8ab4c6a23482024d362efd` 在新600s单次cap内走完维护的10个core文件、149节点，pytest实际 **103 passed、45 skipped、1 xfailed in388.46s**，零普通失败、fallback0，两个native pow节点确实执行；skip类别和既有xfail均保留。首次oneDNN构建也在本cap内完成。原worker/controller均rc1，receipt `d2417388abf66f67100779bea5a3e870628bed6970f7b8a2f73122d9e2054b43` 仍为 `FINAL_SOURCE_CACHE_LOCK_GROUP_FAILURE_RETAINED`：工具遗漏了 `tests/core/test_fused_op.py:10` 在收集期设置的唯一环境键OPENBLAS_NUM_THREADS=1，以及11个MKL编译 `.o.d` 和原save/load测试生成的sub.pkl，不能改写成原执行自然rc0。

root21独立保存终审 `20261028-root-pow-r8-continuation-r1/root-r8-native-core-saved-adoption-audit.json` SHA `20bee571e0d35debc6f29a7972ef856e18699d351da2aaded7acf4672bbce575` 全真，核对全部环境键、声明的CPU/native模式、实际core/utils/stamp、前后source、选集/全部阶段报告及原预算/身份；每个未列编译依赖文件的target均匹配真实worker685527，source指向本r8的11个实际MKL源文件，sub.pkl来源原test_save_load_sub_module且该例PASS。缓存814→7143只增6329、无旧文件改删，所有新增均为既有窄合同或上述12个明确源码产物。实际终态 `20261028-qwen-pow-r8-cpu-core-r1-prepared/attempt-01-native-core/cache-after.json` SHA `8f357097f4ef50ae45f0dda5f70d75f7c9ba2f2d8db3d19160df6211755a22c5`；当前7143逐项一致、原JIT锁与新增oneDNN锁均未持有、自有组空，exec35759已收回，没有重跑native。

本次另发现新增native pow回归先power.numpy()、后读取widened，弱同步可能先物化float32、掩盖旧融合错误；它的PASS只能作为实际测试结果，不证明防护能在旧源失败。公共测试的detach会建更晚clone，不能直接断言同样盲点，仍准备统一的consumer-first验证。下一项仅在旧r7/新r8实际CPU产品上分别新120s单次最小控制，先保存widened/倒数消费者，再读power，同时保留原native物化优先对照；不改冻结源码或重跑core。Torch core将在此控制的真实新基线上单独600s执行，CUDA900继续HOLD。原完整no-hook、LoRA阻断、其他FAIL/原预算/Day4–5门槛保持，发布 **Not complete**。

## 2026-10-28 06:28 +0800 r8六项CPU数值证据已独立采用，原worker元数据失败保留

r8 focused CPU单次cap180在17.32s内实际完成6 passed、0 skipped、fallback0：原mixed四项（数值2、mock2）、新公共pow宽类型/倒数边界和既有reflected scalar pow。原包 `20261028-qwen-pow-r8-focused-cpu-r1/` 的worker最终在pytest之后的环境assert处rc1，正式receipt仍为 `FOCUSED_CPU_FAILURE_RETAINED`（SHA `84d65718bcf9d77f8dec9112b378b126d90b3059092288a31a79f4c1fb687470`），result SHA `84bc725544bb713b0ffb7a52b0ecdd83828af7db8b83b0183c5cf4523e43bf57`；没有重跑、改写或补填原件。根因是工具把post-pytest环境错误比较为post-import环境：canonical `compat/tests/conftest.py` 只给PYTHONPATH追加pytest-site-packages一次，其余完整环境键exact；没有数值断言失败。

root纯保存文件19项独立终审 `20261028-root-pow-r8-continuation-r1/root-r8-focused6-saved-adoption-audit.json` SHA `d73481745303577d2b7ba6e9ed8d576d63dc36ad2e736dc7bf699628e6bd8fba` 全通过，按源码filesystem-only `_plan`与`check_details`确认14项部署文件逐字一致，18条setup/call/teardown报告全部passed、6个nodeid精确匹配、fallback0，原worker/父controller身份与source、实际r8 core/utils/stamp保持。parent完成的source-after和当前源码guard有效；worker自己未到source-after的观察边界保留。缓存732→814，仅68个JIT和14个canonical部署文件新增，旧文件无改删，组/精确锁空；保存终态 `20261028-qwen-pow-r8-focused-cpu-r1/cache-after.json` SHA `d017b0938eb955358e34023f58e43f312a0dea11cdb8f6d8af40fbc96e8e43ec`。这允许采用六项CPU测试证据，不等于原执行包自然rc0或完整模型通过。

本次再次fetch `origin/cgq_transformers@e5cffe1a` 为六树祖先，无需整合、未碰2.0-refactor；主树原99文件字节/大小/模式均保持（同步快照 `20261028-root-pow-r8-continuation-r1/root-resume-sync-0625.json`）。没有本任务框架/JIT进程。下一项绑定真实814基线，root串行单次native core、随后独立Torch core，各新600s；不再额外重跑focused6。CUDA900仍HOLD待CPU core真实终态，原完整no-hook与LoRA门槛、其他FAIL/原预算和Day4–5性能门槛不变，发布 **Not complete**。

## 2026-10-28 06:12 +0800 r8 CPU冷构建终态通过，数值回归尚未开始

`20261028-qwen-pow-r8-cpu-bootstrap-r1-prepared/` 已在原600s单次上限内自然rc0，正式receipt SHA `6f9e4639b8859d1a5a03733ef1a1990222731cd98c10001ff60d81c52f52d786`，exec15130已收回。root独立17项终审 `20261028-root-pow-r8-continuation-r1/root-r8-cpu-cold-terminal-audit.json` SHA `fefa46c12d45bf81d4b5428b2f56a7fa95a5e5b220d8f789e58a3f0d06c6fa1c` 全真；实际227TU与新的core `4944108e3ee106d14eea1327c88af428d0bcf4f0dd101b2efcd11a0756e6cd99`、utils `a6c35acd96e61633b5e1d5b006436548ecd03b47d9247a7028dcea6f2b30828b`、stamp `a0cca8088b44cefc00c0814b2279ea1a197a35193f22885a1afbd8ed90ebe1b8` 一致。完整环境、实际CPU flags、source/compile order/ingredients、两次fresh审计与真实命令有效，当前组/精确锁空；无tensor、测试或模型，不能记成数值通过。

下一项在此真实CPU core/cache上分native和Torch子进程串行执行core白名单；Torch另加原mixed四项、新pow边界回归与既有reflected-pow控制，native core已含新方法。两侧新单次600s上限，准备包正绑定实际产品，尚未执行；native必须关闭shim并移除会激活它的PROJECT_ROOT/RUNTIME_ROOT，不能只写SHIM=0，也不能硬套Torch导入环境。CUDA900仅离线HOLD，等待真实同源CPU原mixed回归等前置，不借旧core、不自动后继。原r7完整logits FAIL与LoRA阻断保持，其他FAIL/预算和Day4–5门槛不变，发布 **Not complete**。

## 2026-10-28 05:58 +0800 CPU pow融合精度泄漏已复现，r8仅隔离源码修复

第二个最小包 `20261028-qwen-r7-public-pow-barrier-micro-r1-prepared/` manifest `b72a77caf2988665000d26639bfe8e33dbbd16ff26ea8c0c35ccb2a37ac7056e` 已单次完成CPU→真实CUDA，四侧自然rc0（CPU PT/JT2.441/4.855s，CUDA4.554/14.928s）；CPU receipt `503fcf557520f7f3f6bedb2f50ab679ef79c76e2ab3a2d6faca21044dfba97f4`，CUDA `bc7d1c79a20e39a75f14d6e61941b7e2f83952b877cf2cf14386ef5b5cc47de2`。root两侧22项独立终审通过，CPU SHA `0a78a6fdc8020b4ac22fa7bcdd268395dac6aa72f791bd23d6802c1fe31a8cf8`、CUDA `4d4455fcbebf61da942d0430e707926880d0ef45d98a882aac3c1914f279b3d5`，新CPUcache816/CUDA1358均仅增8、无旧文件改删，自有组和精确锁为空；旧预算不重置。

独立保存数组对照 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-pow-barrier-saved-comparison.json` SHA `dbde72a52749183374f869afec6fcacef5c9f9ce1c94342e7505ef87d8e3ee36` 确认：JT CPU原整数theta连续链与保存power重建float32 tensor后的倒数相差25项；float(theta)或float32 tensor(theta)链均逐位exact该物化分支。PT CPU及两侧CUDA原链与物化分支均差0。对同一保存PT power，两runtime两device的倒数与reciprocal各64项全部exact。因此CPU问题在pow→融合宽类型consumer的声明dtype边界；保存的double pow再取倒数也能独立重现旧JT64项，但该重算本身不冒充device干预。修复这一边界后仍可能有原pow库的一项CPU数值差，不能承诺完整模型通过。

root仅在新隔离树 `torch-compat-storage-silu-cat-grad-rmsnorm-pow-20261028-r8` 对CPU pow返回表达式显式cast到声明输出dtype，保持参数计算和CUDA `_signed_pow`；另在native BinaryOpCases/public TestBinaryOps各加一条宽类型consumer/reciprocal回归。r8 source `f6751970c2da991ee09e23b7b08f50d8d777b9e916c21a76b2b895daea6879ef`，`20261028-qwen-pow-r8-source-r1/candidate-source-manifest.json` SHA `7a08b93941b4466eea82267bce1dc1d045fdcc37f23a51ec9da6ab995edd2dd1`；source review `d6a2c0bab063612b7fd9309e55cb808736f7d08d34a85db9759a3552cbc38848`。相对r7仅3文件改变、无增删，2806 regular/2807 inventory，两个回归AST有效；r8 CPU600 cold final已由root审阅单次放行（manifest `979785d33186c7c7f5f733e2ce0b4f553de264f62f64d266685700a9335062b3`、launch review `fcba36097328b8440ec76d63d1e769f9f53f3a63672b04a335056bade181d0ef`，exec15130）；实际child663790/start174781968正在串行编译，当前没有构建终态或数值结果，禁止重复启动。真实命令/私有组补采保存在 `20261028-root-pow-r8-continuation-r1/root-r8-cpu-cold-live-identity.json`。CUDA900必须等实际同源CPU证据，不能借旧core/cache。主源/r7/旧失败全保留，尚未采用main或启动LoRA；原完整no-hook仍FAIL8,690,528/29,171,712，其他FAIL/原余额/Day4–5门槛保持，发布 **Not complete**。

## 2026-10-28 05:46 +0800 r7公共pow四侧控制完成，CPU初始化链可复现保存频率

新最小包 `20261028-qwen-r7-public-pow-micro-r1-prepared/` manifest `388c709b978aea0ff81610f2c86832b6099c4bfdc897c5c68e9b7a4fcc9d7d41` 已由root依次单次执行CPU→真实CUDA；两device PT/JT均自然rc0，CPU各2.391/8.577s、CUDA各3.395/36.933s，均在各自新120/180s微预算内，未重置旧模型/T5/Vision预算。CPU receipt `d978831aa55b80ef7b6643d74bb87f35c4b44d8bfd29400145cdf9ab878e824d`，CUDA `2865c203c8ebd2eaa6d0e187c3f05202d3c10bd1e27e920dd2ef437eb11bb220`；双方真实cmdline/PID/start/私有组和worker自身身份均有交叉核对，实际device/fallback0、source/core/env/组锁有效。最新CPUcache808=780+28，CUDA1350=1326+24，旧文件无改删。

root两侧独立22项保存数组审计全部通过：`20261028-root-rmsnorm-r7-continuation-r1/root-r7-pow-micro-cpu-terminal-audit.json` SHA `c032331d84ad17ae72ab3e206e5f14cffe7719ed5526e525d2caf5e070a443a6`，CUDA `root-r7-pow-micro-cuda0-terminal-audit.json` SHA `2d6ec391b19d858559b13dc498c5a8afe8fa32223e997ff800102133d0b00aee`。64项中独立PT CPU inv_freq逐位exact保存PT，JT CPU逐位exact保存JT；两CPU的power仅1项不同，但inv_freq及reciprocal均24项不同。新PT CUDA相对保存PT不同2项、JT CUDA相对保存PT不同1项；同device CUDA两侧power不同4项、inv_freq不同3项。故不能把保存PT频率当作CUDA初始化结果，或直接修改CUDA double-only `_signed_pow`来解释本次保存频率。四值既有reflected-pow数值控制及六项正负/整数/非整数NaN语义均通过；NaN payload位不同不等于有限数值差。

新CPU生成源码可见整数theta与float32 exponent经 `std::pow(int,float)` 产生double的auto局部，后续融合double除法再castfloat32，虽然pow张量声明float32；这是待最小干预验证的中间精度假设。下一项仅离线准备int/float/f32 tensor基底和保存power重建tensor的断融合控制，尚未执行，未修改r7或创建r8。独立审计首次把CPU launch环境直接等同进程环境，因已知CPython增加LC_CTYPE而rc1；失败记录保留，改用冻结expected_process_start精确合同后只重跑NumPy审计，没有重跑四侧数值。fresh fetch仍e5cffe1a为五树祖先，原dirty字节保持；原完整no-hook仍FAIL8,690,528/29,171,712，LoRA禁跑、候选未采用main，其他FAIL/原预算与Day4–5门槛保持，发布 **Not complete**。

## 2026-10-28 05:38 +0800 r7首层attention同输入控制有效

新微回放包 `20261028-qwen-r7-first-attention-replay-r1-prepared/` manifest `e739f1bade43ee3a7d6d814b3d98dbcb1476077fcb81dd1a2244352bb9fe849d` 已由root逐阶段单次执行。独立PT/JT均native原概率逐字exact、manual/public概率及pre-o exact、显式BF16 matmul与AMP exact；各在新300s微预算内自然rc0（10.672/33.255s），不是性能结果。正式PT receipt `33d2fd50b529812d14823a056f8954ae86fec32ad9f980a234d77aae3d6caa2a`、JT `ba9f058573a2cd2845fad5533318fcad3acaa2c0a6d1e14830b6a040f7c4f001`；独立saved比较receipt `1fb8dbcc82c8e2ec8b06074c4dba2e9d7ca1dd3580fbfc16954bcff38804c068`、result `d4c86356ba678e5adc619a493766a082cf5be13e39b354b867054af74794de6c` 已在原120s内自然完成。三个exec30264/59089/26655均收回。

root独立26项saved终审 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-first-attention-replay-terminal-audit.json` SHA `85290495d6cc4c3c0009062462af2d4c00aa7e27daf5ea19053fbeeee3df6b3c` 全真：固定同一cos/sin时，PT/JT全部32数组exact，包括mask、BF16 QK、scale、masked logits、概率和pre-o；只替换PT/JT cos/sin，两侧均产生40个Q/58个repeatedK BF16变化、61个masked-logit变化，并恰重现原1302个概率差（max0.004319220781326294）及466个pre-o差。它证明本首层固定QKV控制的cos/sin影响，不证明完整模型根因或发布通过，不能继续以旧cohort的RoPE无影响结论代替本次结果。

实际CUDA/fallback0、完整JT环境/core231/order/ingredients/source和组锁通过；CUDAcache1326=1306+20、旧文件无改删，终态 SHA `1d7ea7b9d2100ff836cba12de4ca82bf4ff90b97510e05c2b3e7c074a9318c31`。两原launch identity的cmdline因Popen瞬时采样为空，parent Popen argv/PID/start_ticks/private组保留；JT由root存活期另采实际命令（`root-r7-attention-jt-live-identity.json`），PT退出前未补采，该身份观察界限已明列，未回填原件。下一项仅离线准备公共pow→reciprocal的CPU/真实CUDA四侧最小控制，对双方保存inv_freq逐值核对，不假定PT初始化device；静态CUDA double-only `_signed_pow`仅是待证假设，未修改源码或创建r8。原完整no-hook仍FAIL8,690,528/29,171,712；LoRA不启动、候选未采用main，其他FAIL/预算和Day4–5门槛保持，发布 **Not complete**。

## 2026-10-28 05:21 +0800 r7独立root38通过，原dtype差清零

root单次独立复算117数组/113有序边界及五probe全部有效，`20261028-root-rmsnorm-r7-continuation-r1/root-r7-layer-capture-terminal-audit.json` SHA `9416389296587a6c103ba8669c22e6967495be834e4716188d9ad4978e73b8de`，38项全真、exec84321自然rc0。实际CUDA/fallback0、完整环境/source/core231/order/ingredients、1306终态cache与组锁有效；这仅允许采用诊断边界，full/nohook/causal/release仍全false。

本次113边界原tensor dtype差异为0（旧r6为六处q/k norm），首层q_norm/k_norm输出、输入、weight及q/k/v投影均对PTexact；不能继续把旧mixed-dtype差写成r7剩余失败。首层attention概率仍不同1302/294912、max_abs0.004319220781326294，o_proj不同1515/196608、max0.001953125；更早RoPE order1仍有1284项微差/max7.539987564086914e-6，首层input_layernorm有FP32微差。新首差不是根因证明。下一项仅离线准备saved r6/r7逐边界对照和首层attention同输入算子回放，区分PT/JT cos-sin单因子及后续计算；两侧本地回放须先exact各自原概率，尚未执行，不先修改新源。旧r56/r58对其原cohort的无影响结论保留，不能未经验证推广至改变dtype后的r7。原完整logits仍FAIL8,690,528/29,171,712，LoRA不启动、候选未采用main，Day2–5仍 **Not complete**；其他FAIL、预算、Day4–5门槛保持。

saved r6/r7/PT逐边界对照也已单次完成，未运行任何框架/模型；包 `20261028-qwen-r6-r7-pt-capture-diff-r2-prepared/` manifest `cb348b2ddf4aca9e7d4526856ce49acb41162778ba29403369bf07a529e45190`，正式receipt `b99d7ebcfabc577c7c730e4e1691462e4643ff538a3cd8bc9673d96f5a4588c2`、comparison `1edd164b6d955b33891f9b5e7b6c0d6c757eef99627bcd44aa4fb394e2ffc5bc`。113边界中六处原dtype改为PT匹配、零新增dtype差；两数组由不exact变全exact（首层q/k norm），零数组级exact退化，但元素级仍有2,594,086项旧exact→新不同，不能称单调改善。第二/三层q/k norm输入已与PT不同，不能以这些不同输入的输出误差孤立归因RMSNorm。sum113含重复底层激活，仅为保存边界exact诊断，不是完整logits容差统计。

## 2026-10-28 05:16 +0800 r7 capture已终态，独立边界复算待验

新r7原113边界capture的worker/comparator均自然rc0（64.618/7.088s），exec89591已收回；正式receipt `cdb04b21cf898066c72c9c81187a5b01e93fb4a049a813f749bbe69e7f980362` 为 `DIAGNOSTIC_CAPTURE_COMPLETE_GATE_UNCHANGED_FAIL`，comparison `8b25c1c290f6c44d140b97a0eb224c857b30dda46f7d12de7c4ab54a980a146b`。原capture/comparator确认五probe逐字复现本次r7 no-hook；尚待独立root38，暂不采纳新首差异解释。

最新CUDAcache1306=1274+32、无改删或非法新增，cache-after SHA `e22439219f48fa12839a9624b764adce0d5d02c17d22cffd805ad8d97cc2f652`；自有组与精确锁均空，未重跑PT或修改候选。下一项只离线绑定真实终态的独立审计final，再由root单次复算117数组/113顺序及五probe、原dtype与float32存储区分。完整no-hook仍FAIL8,690,528/29,171,712，所有发布/LoRA前置门槛与其他FAIL/预算保持，Day2–5仍 **Not complete**。

## 2026-10-28 05:15 +0800 r7原113边界capture单次运行中

root19项启动审查全部通过，`20261028-root-rmsnorm-r7-continuation-r1/root-r7-capture-launch-review.json` SHA `38dd87687ca96c31e094f902f2a6977ebc1b833a57c9426f06a272cd944dcb52`。新包 `20261028-qwen-r7-layer-capture-r1-prepared/` manifest `c9689d9022ee8cdaa4fbb25aa1a3209b666b239a5e0f2b02aeb14020480578d2` 已单次启动（exec89591），原worker1800/comparator120、113边界/117数组/五probe exact合同保持；capture、comparator及controller逐字复用原root38模板，环境/source/products/support绑定实际r7，最新1274缓存exact、精确锁unheld。入场free51296MiB，真实CUDA导入及原worker正常，当前尚无终态，不重复启动。

fresh fetch远端仍 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为五树HEAD祖先，无需整合；main519d0c的99项逐文件dirty字节全部保持。完整no-hook仍FAIL8,690,528/29,171,712，LoRA不启动、候选未采用main。完成后绑定真实终态执行独立root38，只能依据有效剩余边界继续定位；Day2–5仍 **Not complete**，其他FAIL、原预算与Day4–5门槛保持。

## 2026-10-28 05:07 +0800 r7原完整no-hook数值仍FAIL及旧新对照

exec32616已收回自然exit1；原worker和原comparator均自然rc0、分别257.699/3.173s，正式receipt SHA `e32c32336a20a48b4e9b77346d0aea5c4eae3962fbcf838233a941dbe13cc842` 为 `FULL_NOHOOK_NUMERICAL_FAILURE_RETAINED`，不是运行时或工具失败。原29,171,712含padding logits仍超限8,690,528项，max_abs3.9296875；active3,714,487/22,030,720超限、max_abs.4375，padding4,976,041/7,140,992。三输入exact，loss误差.0046465396881103516满足原门槛，全部有限；.005/.03不变。

root复用既有generic28审计器单次独立复算，`20261028-root-rmsnorm-r7-continuation-r1/root-r7-nohook-terminal-audit.json` SHA `f8ae8cb66c4f9e5c7bbbf245d8ee2d742c0c527f0d7608869278b810e6b1e8e2` 的28检查全真，独立全量计数及max_abs一致，精确PT logits4,254,663项；numerical_gate_passed=false。真实CUDA/fallback0、原模型设置/无hook、完整环境/source15cafdfa/core231/order/ingredients、原caps/单次执行与清理均有效，新CUDAcache1018→1274仅增256，无改删、精确锁及私有组均空。没有重跑独立PT、完整模型、CPU4/CUDA2或unit1。

随后仅离线读取原r6、本r7和同一PT oracle的保存NPZ，`r6-r7-nohook-saved-difference.json` SHA `23517f625ea9d8ce288e2a84b8487af4ed06ab45f325eeae7521c73f47bf0e0a` 独立确认两份原root28有效、source delta仅module_methods守卫+原测试、三输入exact。25,811,571 logits改变，旧超限→新通过3,643,481、旧通过→新超限2,214,035、共同超限6,476,493、共同通过16,837,703；净少1,429,446，不能写成全部改善或根因闭环。active的修复/新增超限分别2,140,760/1,479,684，padding分别1,502,721/734,351。首变化sample0 position1旧34,144超限→新0，全151,936项exact PT；该控制只说明本cohort有可观察前向作用，完整修复仍false。

下一项只离线准备新r7原113边界capture及原root38独立终审后继，绑定本次no-hook/root28/最新1274 cache与实际产品/完整环境；五probe必须逐字节复现本次r7 no-hook，PT113复用，原tensor dtype与保存float32分列，不据最早RoPE微小差异直接推定模型根因。目前capture尚未执行；不修改r7/main、不重跑已通过阶段、不启动LoRA。Day2–5 **Not complete**，T5/Vision原余额与禁止重试、BERT gap/原checkpoint缺口、Day4–5正式稳定窗口/三轮/容量/峰值门槛保持。

## 2026-10-28 05:01 +0800 r7原完整no-hook单次执行中

新final `20261028-qwen-rmsnorm-r7-nohook-final-r1-prepared/` manifest SHA `e46ac379ec010b0f83182539d213d476e67c478c973edfd86d25556d93199a17` 的28工具已冻结，offline24项全真。root实际18项launch审阅 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-nohook-launch-review.json` SHA `a26fa337643e1d26415ed3dbb0635b1a2d605786ff68c6ddf3c4fce6384efda9` 全真：双cold/root17、CPU4/CUDA2 root31及mock unit1/root29实际终态、source15cafdfa、真实CPU/CUDA products、最新1018 cache exact/精确锁空、原main99 dirty字节均有效。七数值文件及原controller/identity/helpers逐字不变，validator suffix仍0a0eb93d；完整模型环境仅在实际CUDA2 launch的PYTHONPATH前面加入该final support目录，原环境其余字典完全相同。

05:01 +0800 root已单次启动exec32616，receipt `attempt-01/receipt.json` 当前RUNNING，worker身份由实际receipt保存；入场free51296MiB、不以util为条件。原全部29,171,712含padding/.005/.03、worker1800/comparator120、同权重/输入/初始化/依赖pin与独立PT oracle均不变。已见实际CUDA导入和原worker正常开始，尚无新完整数值或完成收据，不重复启动，不采用main或先启动LoRA。终态后复用既有 `20261028-root-cat-r6-continuation-r3/audit_nohook_terminal.py` 的generic28独立复算与终审；Day2–5仍 **Not complete**，其余FAIL、原预算及正式性能/容量门槛保持。

## 2026-10-28 04:53 +0800 r7 CUDA原mixed两例及邻近unit1终审通过

root13项launch审阅 `root-r7-cuda-cold-launch-review.json` SHA `b672ecd6156c8d9da67491198cc2c811798634bebda4552eb01fe4daea415e27` 通过后，单次运行新 `20261028-qwen-rmsnorm-r7-cuda-bootstrap-final-r1-prepared/` manifest `b7a106515c169c369a09ecdaed1bd9bf4b291d5e11db7d363cfce7da7b5cff7f`。实际入场GPU UUID沿最终绑定，free51296MiB，仅按足够显存，不以util为条件；cap仍cuda:0=900，CPU禁跑。正式receipt SHA `1d2aa20b7f4607d2c0d6836dec22d3210b5419414a046d7ee2ae0629caa42df4` 为 `PASS_COLD_BUILD_IDENTITY_ONLY`，自然rc0；未创建tensor、未启动测试/模型，也无自动后继。

实际231TU=194 repository+37 generated，core SHA `9b4d1a11f5d6c8066c79f2d6c3e2dd6b4d71514786357beaec654fd78fcaccfc`，utils `b2de88a150f8f255f47a31010870b6c64dd4509c5401723a5386f19dd080cc5a`，stamp `ccecf2373ee29e9bc27856af96f3969644aa154b26eb06c0c98cbbbb79d58ba5`；全新私有cache、两次fresh审计、完整环境及实际两phase has_cuda/use_cuda=1有效，与launch环境use_cuda=0分别记录。root复用既有stdlib审计器单次独立终审，`20261028-root-rmsnorm-r7-continuation-r1/root-r7-cuda-cold-terminal-audit.json` SHA `866ecddc21d5fc7140ea118a26d5601c1f4e4dfa269cb777652385979bbd50da`，17检查全真。自有组与精确cache锁均空，exec14980已收回自然exit0。

root13项数值launch审阅通过后，单次运行 `20261028-qwen-rmsnorm-r7-original-cuda2-r1-prepared/` manifest `75ef462c85f2bd40291723e21259a3f25668312d2ab9563ee8f89518e998af5a`，cap360沿旧mixed控制。唯一一次原pytest为2 passed/2 deselected in58.59s，未重跑CPU4、mock或独立PT；exec29364已收回自然exit0。正式receipt SHA `974dba80935337740d04dc1c4b9ab553d19f999bfe76d4f809d6f79c18e82de2`，result `d86bc1fd3c8b714af3ea4bc6c139766a959ee780149d3bb8045ad46360990ab1`，状态 `PASS_R7_ORIGINAL_CUDA2_CAPTURE`。FP16/BF16原eval与train输出FP32、input_grad低精度、weight_grad FP32，以及module/direct-forward四张量exact断言全部通过；旧r6 eval-only错误已在此fixture消除，不据此外推完整模型。

root独立stdlib31项终审 `root-r7-cuda2-terminal-audit.json` SHA `4f56a3795bfc91d3b1967a16db30924f31353a4a384bbd8ffa7d78992137eb7a` 全真，独立复算16张量payload/hash/dtype/shape/native CUDA dispatch及全部route一致性；已保存独立PT CUDA decoded值对比全部exact，仅诊断，保留PT原LAZY元数据FAIL，无例外扩到JT。实际CUDA/fallback0、完整环境/source/core231/order/ingredients有效，新cache942→998仅增56、旧文件无改删，私有组与精确锁空。

随后root单独放行 `20261028-qwen-rmsnorm-r7-adjacent-unit1-r1-prepared/` manifest `353ec2ad817ae29feb32b8b315114612b4b28cc12d1e4b2a2e4a15f37f1937a2`，只运行原 `TestRMSNormDispatch::test_standard_contract_routes_and_offset_variant_falls_back`，原norm整文件字节不变。1 passed in21.96s，cap360内自然rc0，exec83908已收回；receipt SHA `b1d46c1eadb4c48c02aac31df57199835f625c05fe4ca52e0a4ac83755b6b1c5`、result `efded387baa8659f8f0077b76c847cad69426382b1baea5b7e8415f938a70971`。root独立29项终审 `root-r7-unit1-terminal-audit.json` SHA `aecee948ab7af7c3eb752ead4c82fbf54dc823fa6c73bb91baa8f3972f86a659` 全真，独立复算原3个tensor的3/1/5 payload/hash、FP32 dtype/[2,4,8]/native CUDA以及原三相pytest、fetch_sync仅原调用一次/对象原样返回/原函数恢复。该测试明确仅mock routing，不作为独立RMSNorm数值。完整环境/source/products/fallback0有效，最新cache998→1018仅增20，组锁空；不重跑任何已通过阶段。

下一项仅原完整no-hook新final，正离线绑定实际双cold/CPU4/CUDA2/unit1/root审计、完整环境和最新1018 terminal baseline，尚未执行。原七数值文件、support、validator后缀和29,171,712含padding/.005/.03/1800+120门槛保持。Day2–5 **Not complete**，不采用main、不启动LoRA，其他FAIL/预算/Day4–5门槛保持。CPU记录已精确提交为 `765e0368fdd985592dddfa2fabbf2681129f5daa`，无push，原dirty字节未变。

## 2026-10-28 04:34 +0800 r7 CPU冷构建及原mixed四例终态

新r7 CPU600 cold已自然rc0，原包 `20261028-qwen-rmsnorm-r7-cpu-bootstrap-r1-prepared/` 正式receipt SHA `df8a4750b7399b55e24ff99a14d112e37520985322ec33b37bd95d328e476526`，状态 `PASS_COLD_BUILD_IDENTITY_ONLY`。实际227TU=190 repository+37 generated，新core `1f6945f6e816b1a45f26579caacd45891bcbc5a4726ba9103d04b549fabf170f`、utils `b2de88a150f8f255f47a31010870b6c64dd4509c5401723a5386f19dd080cc5a`、stamp `fb13fbc04a90e9c22205a4e7d0fc37b7b48f55877fbc386f33c6f72382e143cd`。两次fresh审计、完整环境/source/生成物/组锁有效，root独立17项终审 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-cpu-cold-terminal-audit.json` SHA `159c707c9c7dfa25b5c3254bedcd0b7db8138d72c27ee84b7119c954c186bc97` 已实际通过；没有重跑cold。

root审阅实际cold产品绑定后，单次运行 `20261028-qwen-rmsnorm-r7-original-cpu4-r1-prepared/` manifest `e1584946bf1c842736effd3fbd04cb627c9aee2226ae8e294e88b99c88ea5ae3`，CPU cap360沿旧mixed控制，不占用T5/Vision余额。唯一一次原pytest得到4 passed in 13.17s（数值2+mock2），正式receipt SHA `31e5f5efd9abf0279c482ccbb1b3f527edeb5a3500bec4b2904633a14c74b6a6`，result SHA `d5666a8f9227ad2332be6c951897a39d49259ce9fde8371a0633c590c59415fe`，状态 `PASS_R7_ORIGINAL_CPU4_CAPTURE`，自然rc0。原测试和全部断言字节未变，array helper每次只调用一次且返回原数组，未另起shared数值child。FP16/BF16原两route均满足输出/训练输出FP32、input_grad低精度和weight_grad FP32；module/direct-forward四张量exact。与已归档独立PT CPU的16张量decoded对比均exact，仅诊断，不增加跨runtime门槛。

root独立stdlib终审 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-cpu4-terminal-audit.json` SHA `215b435e0f740426382176eab95ebcadf7cd8f333d6eb4e3a38adb747be3758a` 的31项检查全真，独立重算payload/hash、dtype/shape/device/native dispatch、route一致性和PT诊断。完整环境、同一source fingerprint15cafdfa、实际core/order/ingredients、fallback0及组锁有效；新cache732→780只增48个合法文件，旧文件无改删。进程已自然退出；CUDA产品仍null，CUDA-only900 final仅离线准备，尚无新CUDA数值或完整模型结果。fresh fetch `origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 仍为五个本次复核树HEAD祖先，无需整合。主树HEAD `ec12e67ab16b5f4da91821f3b4cf224141a67e63`，逐文件99项dirty（聚合目录视图96项）均保留，另存 `main-dirty-bytes-before-cpu4-continuation.json`。下一项新CUDA-only900 cold/root17→原CUDA numeric2→最小同dtype邻近unit→原完整no-hook，逐项单次root放行，不自动重试；Day2–5 **Not complete**，不采用main、不先启动LoRA，其余门槛/预算不变。

## 2026-10-28 mixed-dtype 旧CUDA契约复现与隔离r7源修复

metadata-only后继 `20261028-qwen-r6-rmsnorm-old-control-r2-prepared/` manifest `2706331e2d8bae94191205123fec2c19981803c5cc6505ccd97524dd38ffc248` 只首次执行JT CUDA和单列mock unit，没有重跑已完成的PT CPU/JT CPU/PT CUDA。两阶段均自然rc0，原数值/断言失败有效：JT CUDA receipt `64c6d177889b73ac2805dc22d4c9ab642036aeed8f8650d8c6ca79547f2a7cd1`，unit receipt `f9fa12a4057dfdbb61d770ef9cd02c3b0cbef06afbb45151cbd50ed889a396d2`；summary SHA `5bdba40fea75d1f5d8657af80f039c8180eb142faf49499c984e86e4036f4388`。

准确失败范围是eval module调用：FP16输出仍FP16而direct-forward FP32，11/16项不同、max_abs0.0006103515625；BF16输出仍BF16而direct-forward FP32，10/16项不同、max_abs0.0087890625。训练输出两路线均FP32，input_grad为输入低精度、weight_grad FP32，训练和全部梯度的module/direct-forward比较通过。mock两例原断言均因mixed路径进入ACL检查而FAIL，单列为compat unit。root独立24项终审r2 `20261028-root-cat-r6-continuation-r3/root-r6-rmsnorm-old-terminal-audit-r2.json` SHA `f74d35c18b902d1d9f93d8ad5320dae06be318d2eb177ba01da8f26c14500112` 实际全真：复算全部原断言和payload，CUDA231TU/CPU227TU、core/order/ingredients、完整JT环境、fallback0、缓存链、组锁均有效。root终审r1曾误要求training dtype也失败，原FAIL文件SHA `d181700b…`及脚本保留；r2仅接受实际eval-only失败列表，未重跑任何runtime。最新CPU984/CUDA1430，旧文件无改删。

root随后从r6 manifest完整复制到新detached worktree `torch-compat-storage-silu-cat-grad-rmsnorm-20261028-r7`（HEAD765f1791f），相对r6唯一changed为 `compat/torch/installers/nn/module_methods.py`，唯一added为原76行 `compat/tests/torch/test_rmsnorm_mixed_dtype.py`，无removed。仅在Var类型检查后/epsilon前沿用历史5行混合dtype guard，返回None以执行公开forward，不更改native RMSNorm、reduction或工具链。source fingerprint `15cafdfafa0bafa854aa23b1e0cd4f6fdd477874f43a9eaf9de95fe0e6006e06`；源manifest `20261028-qwen-rmsnorm-r7-source-r1/candidate-source-manifest.json` SHA `5e2ce491c5b3742dbf5efff970ea19356803d69258e7e302e2f1b5184fd79346`，root source review SHA `1e21f04f7410ffddf61b1f9856e8df19291cd953a39b5c1d0a279f2c54ce6861`，2806 source/2807 candidate文件。main/r6、原dirty、失败和cache均保留。

04:22 +0800 root已完成新CPU600 cold包的14项launch审阅，文件 `20261028-root-rmsnorm-r7-continuation-r1/root-r7-cpu-cold-launch-review.json` SHA `8573a351a3ade20c7cd1c40f6d7b9574d3a878717a3871b6cb2fffc1e37b2f01`；包 `20261028-qwen-rmsnorm-r7-cpu-bootstrap-r1-prepared/` manifest `8b38981b065c558c3de7df7406e95f5c281da00fd1c0233c4ed242f925527625` 已由root单次启动，当前仍编译。新私有cache、NO_BUILD移除、CPU无GPU UUID、r7完整环境纯字典重派生与4helpers原字节均已核对，尚无构建终态或数值PASS，不重复启动。原mixed数值/维护、CUDA-only900与完整no-hook后继均仅离线HOLD。先核验本次CPU终态/root17及原CPU4，再单独释放真实CUDA；继而仍须原完整no-hook门槛，不先启动LoRA或采用main。完整模型因果归因尚未关闭，Day2–5仍 **Not complete**，其他FAIL/预算/性能HOLD不变。

## 2026-10-28 04:11 +0800 mixed-dtype 旧控制前三侧与PT环境失败

root审阅原控制包 `20261028-qwen-r6-rmsnorm-old-control-r1-prepared/` manifest `88d1b625c91bda6b0e641334ded0c02e5cd600d5921fb8d1e135bc3d716032af`，独立逆转记录包装后确认原类、array、8维fixture/数值主体和unit AST一致；launch11审阅SHA `c5674447910f68869ae6146afb07126f26a11757734dce84ae242344550b44d1`。主四阶段单次命令已自然exit1：PT CPU receipt `5feb46c6…` 与JT CPU `dc6dc678…` 原FP16/BF16各2例及两route全部通过，输出/训练输出FP32、input_grad reduced、weight_grad FP32、同runtime module/direct-forward四张量exact。JT CPU仅加40个合法JIT文件至964，CUDA1354无变化，未重冷编。

PT CUDA原两例/两route均完成且36/36公共断言通过，但最终完整环境检查发现唯一新增 `CUDA_MODULE_LOADING=LAZY`，触发test_child.py line135断言，原worker/receipt均保留FAIL与自然rc1，receipt SHA `c4327f07182ca455221516f4a146d293e50e2b7f8a827fbb770c1ba4266cd145`；不能改写为原rc0。root独立stdlib审计 `20261028-root-cat-r6-continuation-r3/root-r6-rmsnorm-old-prior-terminal-audit.json` SHA `ff30a5636cf1c62679247982d9dd9100467b15cad54cffb69f2bb19ca35962e2` 实际31/31通过：48数组的float32 payload hash、dtype/shape/device、全部fixture与原断言独立复算一致，独立PT身份/固定版本、JT227TU/order/ingredients/core、完整source、缓存链及自有组/精确锁有效。仅对独立PT CUDA这一个退出环境键接受已完成的原数值，不回填旧收据、不重跑前三侧、不给JT环境引入例外。此时JT CUDA从未启动，8维旧CUDA失败尚未证实。

下一项为小型metadata-only后继，绑定该root31和CPU964/CUDA1354终态，只首次执行JT CUDA，再分列原mock2 unit；旧包/r6/main源码保持，不提前创建r7、不启动模型或LoRA。发布仍 **Not complete**，其他FAIL/预算/性能准入HOLD不变。

## 2026-10-28 03:57 +0800 r6 capture 独立终审

root 在确认 manifest `c8e98118bceba2d591fbd2278e07b1f5dd9bf056bf0f1a978a034b9abf5ef421` 及全部工具字节后，单次运行 `20261028-qwen-r6-layer-capture-terminal-audit-r1-prepared/audit_capture_terminal.py`，未重跑框架或模型。结果 `20261028-root-cat-r6-continuation-r3/root-r6-layer-capture-terminal-audit.json` SHA `9ef576c67018306a1123b6b1b3918ce4e0497b206cd6ac3421fcd6674de8f013`，状态 `PASS_ROOT_CAPTURE_TERMINAL_DIAGNOSTIC_AUDIT_GATE_UNCHANGED_FAIL`，38/38 检查通过。独立 NumPy 复算 117 数组与 113 有序记录一致，五 probe 对本次 r3、PT 五 probe 对原独立 oracle 的 dtype/shape/raw bytes 均 exact；source、工具、完整环境、真实 CUDA/fallback0、core231/order/ingredients/stamp、只增16至1354的终态 cache、自有组与精确锁均有效。

最早数值差异为 RoPE cos order1：1284 项，max_abs `7.539987564086914e-6`，原 dtype 两侧均 float32。最早原 dtype 差异为 layer0 q_norm order13：PT float32/JT BF16，374342 项、max_abs0.1484375；layer0 k_norm 为191884项、max_abs2.25。前三层共六处 q/k norm 输出 dtype 不同。capture 将 BF16 转成 float32 后存盘，故原 dtype 与 NPZ dtype 分列，不能混用，也不能把 dtype-first 误称 numeric-first 或完整模型根因。终审的 `causal_attribution_allowed/full_logits_passed/nohook_gate_passed/numerical_gate_passed/release_acceptance` 仍全 false，原 no-hook 失败不变。

开工 fresh fetch 仍为 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为本次五个复核树 HEAD 祖先，无需整合。main `48dcbd3dd`、99 项 dirty 与空 index 保持；原树51、修复树85、历史 Qwen 树87、冻结 r6树2813项状态只作身份记录，未清理或覆盖。开工未发现本任务框架进程，根盘约84GiB、数据盘约6TiB；10张卡当次最大 free66466MiB，不满足71680MiB容量入口，不代表稳定性能窗口。

下一项只准备并审阅原 `test_rmsnorm_mixed_dtype.py` 的8维 MixedRMSNorm 旧路径控制：独立PT/旧r6 JT、CPU后真实CUDA，逐项记录eval/train输出FP32、input_grad reduced、weight_grad FP32与module/direct-forward exact；两条mock dispatch单列为兼容层unit。当前包尚未执行，不创建r7、不改r6/main runtime、不重复原capture或完整no-hook。Day2–5仍 **Not complete**；其余FAIL、预算与性能准入HOLD保持。

## 2026-10-28 03:33 +0800 r6 原完整 no-hook 数值仍 FAIL

Day2–5仍 **Not complete**。r3 元数据修正后，原完整 Qwen BF16 worker/comparator 均自然 rc0（225.725/3.116s），正式receipt `1644a95cb3da74682d98712b61c874857a19a6891fec9e9b1d47f3c58abfb89d` 为 `FULL_NOHOOK_NUMERICAL_FAILURE_RETAINED`。原全部29,171,712含padding logits仍超限10,119,974项，max_abs4.587890625；有效位置4,375,563/22,030,720超限，max_abs0.4375。输入exact、loss满足原门槛，不改变0.005/0.03。root独立28项终审 `356cfd07ce83a178857aef66fc1c7f4c6d6e17fc2f713f7107869b80e9256415`确认实际CUDA/fallback0、source/products/完整环境/cache/清理有效并独立复算一致，但 **numerical_gate_passed=false**；证据有效不等于模型通过。

运行包 `20261028-qwen-cat-grad-r6-nohook-r3-prepared/` manifest `db03ec32dd5ccc312cd0989f3842242b340bb8744cdc94096d4ba8cb5437d311`，source fingerprint `5f4107c5…`。comparison SHA `6fc40028ffc504e05d069c38220c8b1f0ff7126c63f2dff1bff0a301455dbcc1`，新完整NPZ SHA `cd6b4fd21e3e8026822c3863744b1e34b1961be7cdbe5debcf73067791630eaf`。三个输入exact，loss JT2.6934099197387695/PT2.6900217533111572，abs0.0033881664276123047；padding7,140,992项中5,744,411超限，不删除padding来通过。全部数值与r4原报告相同尚不代替数组逐字核验。

root28终审文件为仓外 `20261028-root-cat-r6-continuation-r3/root-r6-nohook-terminal-audit.json`。完整环境只比实际CUDA14增加本包support的PYTHONPATH前缀，七数值文件及原validator不变；保留r2模型前导入FAIL及其未记录的child退出环境，不回填。r3审阅r1误要求r6前置adapter整体等于旧r4 adapter而FAIL，原件保留；r2核验实际r6继承字节通过（SHA `9b34a607…`）。运行未重冷编/原14，cache仅新增248允许JIT文件、无改删/非法新增，最新1338文件终态SHA `912aa41af329f367bbb9161d9554f1c56b7e94dcc9e3111ead0a6c4e67d5f92e`，自有组与精确锁空。无训练/LoRA/自动重试；未改变main源码。

本轮fresh fetch仍为 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为六树HEAD祖先，无需整合；main起点`f474b69250ac891336a57e1d2cd7e3b41139bd68`，原dirty工作及空index保留。根盘约84GiB、数据盘约6TiB；未清理证据或干扰共享GPU进程。

候选未采用main，LoRA不启动，不重复本次cohort。只读分块比较已确认r4/r6完整NPZ及全部五数组bitwise exact，29,171,712个logits全部相同、修复项与新失败项均0；analysis SHA `014567ebec376f2c01b021e587584c9eb3d25d343f22d7854b549faf77f2e76a`，仓外 `20261028-qwen-r4-r6-nohook-forward-diff-r1-analysis/analysis.json`。因此cat/grad局部修复在本次cohort无可观测前向作用，不能据此关闭模型门槛。r6逐层capture包 `20261028-qwen-r6-layer-capture-r1-prepared/`（manifest `ccbbd038…`）已单次完成，worker/comparator自然rc0、50.757/6.943s；正式receipt `aa3e373e23af600dbfb50173013bcd2d3a8211b7e879817377d36d3b9dd91968` 为 `DIAGNOSTIC_CAPTURE_COMPLETE_GATE_UNCHANGED_FAIL`，仍非模型通过。原r54数值主体不变，PT113边界复用、五probe数组对本次r3逐字exact，cache只加16至1354；独立终审尚待执行，禁止重复启动。观察到embedding及首层q/k/v投影、q/k norm输入和weight exact，但q/k norm输出原dtype为PTfloat32/JT BF16，q_norm374342项不同/max0.1484375、k_norm191884/max2.25；RoPE更早也有差异，不能据边界直接归因模型根因。下一项只准备原 `test_rmsnorm_mixed_dtype.py` 数值主体的独立PT/旧JT CPU→真实CUDA最小控制，先复现混合dtype标准RMSNorm dispatch；没有修改r6或提前采用历史mixed-dtype guard。T5/Vision原余额与禁止重试边界、BERT原gap、跨模型原checkpoint缺口及Day4–5正式稳定三轮/容量/峰值门槛均保持。


原件和容量准入窄复核 `20261028-original11-stat-and-capacity-admission-review-r1/`（manifest `1722585fbabfc01fa021aa585b0da2519f4fcc0dd023265940703b8dc369870d`）按旧audit精确逐项stat：11个checkpoint目录存在，分布BERT5/Qwen LoRA3/T5 3，但22个model或adapter safetensors与optimizer.pt原文件仍全缺，零新增/零stat错误；这不是BERT独有的11份原件。一次GPU只读采样显示10张A800均未达71680MiB free，最大66466MiB、差5214MiB；单次采样不证明稳定性能窗口。未启动waiter/JIT/模型，source/adoption/Qwen-success HOLD独立保留。

## 2026-10-28 r6原cat CPU与真实CUDA数值闭环

原完整no-hook启动增量：root完成新final `20261028-qwen-cat-grad-r6-nohook-r2-prepared/` manifest `962930153d5c5359fa25fbcef37b77aa12890926197239bae2975479ed28c8fb`的25工具与13项launch核验，审计SHA `5136148f8e3be6854e930b9a42c963f9f43f650d2a16155424a880bbc7b566e4`；七数值文件逐字与原r4一致，`validate_full_comparison`原后缀不变、完整环境直接等于实际CUDA14 NO_BUILD环境、最新1090文件cache exact。root随后单次串行释放worker1800s/comparator120s，当前尚无终态；全部29,171,712含padding/.005/.03原门槛保持，无训练/LoRA/重试或自动后继。不要从旧HOLD或当前RUNNING启动重复模型。

r2终态纠正：正式receipt SHA `a47f1674b90768493d2a617bf572c59400c636f13576717f47f7c1268a970ee9`为`RUNTIME_OR_CACHE_FAILURE_RETAINED`，原entry SHA `39848a23dd1556645ee9fa9600b74107bab6a79efc32d0591fed87a4ff3a0bc9`在`unchanged_worker` phase记录`ModuleNotFoundError: No module named '_hf_trainer_e2e_runner'`，worker.py line31，先于原probe runpy/模型加载。只有一个自然rc1 worker，无原worker identity、模型输出/NPZ或comparator，不是logits数值FAIL/PASS。root独立15项保留失败审计`root-r6-nohook-support-path-failure-audit.json` SHA `ce429068b888605cc208b075d3b6f9ca5b90db4f55de5e8c5b2b5136ef1b7781`确认source/tools/新core及已记录导入环境有效、cache1090全delta0、组与精确锁空；失败后child最终环境未记录，不回填。原r4正式no-hook有唯一PYTHONPATH变更以加入本包support，新r2直接复用CUDA14环境漏此前缀。stdlib PathFinder对原PYTHONPATH无结果，只增加support即解析原字节模块；下一项仅新r3 metadata后继恢复此前缀和完整before/post合同，其他环境键不变，绑定原FAIL/entry/本次terminal-cache，不重冷编、不重跑原14、不改七数值或原validator/阈值。LoRA继续阻断。

新CUDA14 final `20261028-qwen-cat-grad-r6-cuda14-r2-prepared/` manifest `cab691249dad72d39b8499d46d758bce7f3a2fd873d061c8620be7cb626958c0`显式继承原CPU14/cat1/grad5终态及各root审计，并绑定本次r6真实CUDA cold产品、完整NO_BUILD环境和942文件cache。root核验25工具、11项launch检查，独立逐文件重建缓存转换：仅移除941个等于root/relative-key的冗余path，把唯一operational_locks映为单项operational_lock，inode32909956及所有其他字段不变，重建对象与原cache严格相等；未泛放松cache/source/环境守卫。launch-review SHA `230c26cf770a439b048f64099f55d7ad80e9eebd08d9cd8d4e539dceb5aa6a1b`。

root单次串行执行原CUDA14，360s上限内约149.042s自然rc0、原14/14 PASS、零skip，原cat/rotate值与dtype、输入梯度和二阶导数全部断言完整执行，实际CUDA0/native dispatch、fallback0。receipt SHA `c2dcd7388e395a11dcbc817efaaaeb87921be9b8a281ffb4b96681d567541352`；原结果SHA `3c8921f4e0d22cddc95484669b061710d4fcf87b26249b251bbf7bd0431f594f`。root18项终审`20261028-root-cat-r6-continuation-r3/root-r6-cuda14-terminal-audit.json` SHA `cbd678960c02318b96732239190e2d18296abf3dfc45371d577daab1e63f7935`全真：原独立PT nodeids/参数相同，source/core/utils/stamp/完整环境不变、两份cache守卫通过；CUDA只新增148允许文件、无改删（终态1090），CPU924完全不变，自有controller/child组和精确锁空。原旧端JT10fail/4pass、r5 CPU10pass/4fail及PT原结果保持，没有重跑CPU/PT或旧cohort。

这关闭r6的原cat CPU/CUDA14与CPU维护1/5局部链，不是完整模型或发布通过。候选未采用main；下一项只在新final绑定全部实际前置及最新CUDA14 terminal-cache/完整环境后，单独释放原完整无hook Qwen：七数值文件/独立PT映射/原comparator不变，全部29,171,712含padding logits、atol0.005/rtol0.03、worker1800s/comparator120s。当前尚未启动模型或LoRA；Day2–5仍 **Not complete**，其他FAIL及原预算/性能门槛保持。

## 2026-10-28 02:57 +0800 r6原CPU14及维护通过，后端验收仍开放

CUDA cold终态补充：上述单次900s CUDA-only构建已约423.131s自然rc0，正式receipt `fd36f0bdccdd2a24bb4df97cc408b03ea1eae8224a87637debd735d536024dce`为`PASS_COLD_BUILD_IDENTITY_ONLY`。root17项终审`20261028-root-cat-r6-continuation-r3/root-r6-cuda-cold-terminal-audit.json` SHA `4ee9af9493880dcea6d0c48c01fc8b086dc4b58ce7232d82d17dd5a941e850a4`全通过；231TU=194仓内+37生成，新core `036023be7e02c88c9bc88a1e9b73f09cd08c6820d1d18b3a6c436cc21f497b5b`、utils2925281c、stamp `e7d42d42…`、942文件terminal cache、source与完整环境exact，两次import实际has_cuda=true/use_cuda=1。没有张量、测试或模型执行；自有组与精确锁均空。这关闭构建身份，不是CUDA数值证据；CUDA14仍待新产品绑定后继独立释放，原完整no-hook及LoRA未启动。

03:05 +0800现场增量：root已单次启动 `20261028-qwen-cat-grad-r6-cuda-bootstrap-r1-prepared/` manifest `9518cb60a7f7a96eb354b194336195ceb66060f0987c127139cfd3b3da644f9d`，仅CUDA fresh cold900秒、首次JIT串行、无自动后继；仍在编译，没有终态/数值结果，不得重复启动。root首份launch审阅因错误要求CUDA-only config同时包含CPU cap而FAIL，保留于`root-cuda-cold-launch-review.json`；按实际CUDA-only900与disabled CPU范围修正的`root-cuda-cold-launch-review-r2.json` SHA `34e36129519931597b35cf292843faf1925da5c483d069089be0238a42f0f4d0`通过，14工具及四可信helper原字节/完整CPU前置/231TU schema/source精确。CUDA14与原no-hook只完成缺真实产品/终态的HOLD模板，不读取或绑定RUNNING收据，不作为数值通过证据。

r6 CPU cold包 `20261028-qwen-cat-grad-r6-cpu-bootstrap-r1-prepared/`在原600s上限内约342.730s自然rc0，正式receipt SHA `451435eeba1516c54c3ca245b804ba61253fca5977ca735ef3f5d4236cda4833`为`PASS_COLD_BUILD_IDENTITY_ONLY`。227TU、core `55ee206a…`、utils `2925281c…`、stamp `6805d809…`、完整环境与两次fresh审计有效；root独立17项终审 `20261028-root-cat-r6-continuation-r3/root-r6-cpu-cold-terminal-audit.json` SHA `f774a8b95c72c4f42a0e99661f907abc6e6dd41f9fa603de0b2fea8ecc641f95`通过，自有组和精确cache锁均空。

随后root核验19个工具及真实新产品/NO_BUILD环境/732文件terminal cache，单次释放 `20261028-qwen-cat-grad-r6-cpu14-r2-prepared/` manifest `c5bd04b04d94ed407f1dbaea35dbe612a717d1215cbbfed702aa29f1250d2693`的CPU14（360s上限）。正式receipt `8355e7730a31d2944cd7a2980830e217a6937dd892f9c8d1a7041be67c101ab8`为`PASS_NEW_CAT_SIDE`，原14/14通过、零skip，含原cat/rotate值、dtype、输入梯度和二阶导数完整断言；同原独立PT nodeids/参数、真实CPU/native dispatch、fallback0。root16项终审 `20261028-root-cat-r6-continuation-r3/root-r6-cpu14-terminal-audit.json` SHA `2b795145fdab3228ce364ad4497a606887ff6f791d95a24eb1d6eda07d7c3b72`确认source/core/utils/stamp/完整环境不变、cache仅允许新增、自有组与精确锁空。独立PT旧通过原件及r5原10pass/4fail均保持，未重跑或改写。

维护后继 `20261028-qwen-cat-grad-r6-cpu-maint-r3-prepared/` manifest `33ec687c56d80dda19e98727fb9aaebdb34da604119d468c18b140650fb2e965`显式绑定原CPU14 receipt/result/cache/root审计，没有复制attempt或重跑原14。root分次释放原CPU cat_stack1及既有graph/retain/detach/unused语义5（各180s），分别约7.098/9.691s自然rc0、零skip。receipt SHA分别`e6a8799d2fe891769d29d1a3828b6710ad781defd14cbc751baf2461e68b51c2`、`2664d86a20b544729ccdae62b74f9ed688c374ca6b3db434412c6f68df8de5ce`；root15项cat1终审`60c21c3d1ac0c1bb21511407a3aad80feda3943b1fe23159e3a8d04b470c14ff`、16项grad5终审`f28c771a74a74776a9e62df361905eadf2e6cff98c8e89d165e253c2411cd030`均通过，文件在上述root目录。CPU/native dispatch/fallback0，source/core/utils/stamp/完整环境exact，cache分别只新增16/32允许文件、无改删，最新924文件terminal exact，自有组及精确锁空。计数分别是原数值14与维护1/5，不合并成新的原数值矩阵。

下一项仅离线准备并审查新r6 CUDA900s cold，再绑定真实CUDA产品和终态执行原14；尚未执行CUDA数值或原完整no-hook。候选未采用main，LoRA不启动，Day2–5仍 **Not complete**。fresh fetch仍为`origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`且为六树HEAD祖先，无需整合；main起点`98fab0098…`的96项原dirty与空index保持，快照存于未版本化`20261028-root-cat-r6-continuation-r3/`。T5余51.29068389162421s且禁止重试、Vision余310.095694191s且无合法最终continuous、BERT原15.425953905%>15%及缺11份PT checkpoint不变；Day4–5容量入口仍仅离线实现，没有正式模型/速度/显存证据。

## 2026-10-28 02:34 +0800 r5原14项CPU未通过：混合dtype梯度仍错误

r5 CPU产品绑定包`20261028-qwen-cat-new-r5-cpu-r3-prepared/` manifest `1d05d9e8…`在root核验18工具、真实新产品/完整环境/732 cold-cache后单次执行原14项，37.131s自然rc1、未超时。正式receipt `2420d5c92b1a1abdd3e8165b9de2b0a91f573f4acb5a206f9ecaef9a0f4a58a5`保持`NEW_CAT_FAILURE_RETAINED`；原测试10pass/4fail/0skip，失败分别为mixed+training FP16/BF16的line70梯度dtype与二阶FP16/BF16的line112导数dtype。四case均已通过先前cat值/dtype断言，但其中后续rotate或导数数值断言可能被短路，不能据此宣称全14或所有后续操作通过。Recorder实际CPU/native dispatch/fallback0，reduced leaf为FP16/BF16而gradient（及二阶case curvature）均FP32。

root14项终审`20261028-root-cat-continuation-r2/root-r5-cpu14-failure-audit.json` SHA `630fa8560b9393a05c8a576cc0e6a45908e0f495bee1cb9dfb4fcb50e21375f4`确认真实语义失败；source/core/utils/stamp/env不变，cache仅新增128允许文件、无改删，组空/精确锁空。维护CPU1没有执行；r5 CUDA cold虽已离线准备（manifest cb9283d7…）但无attempt，不启动CUDA/no-hook/LoRA。旧FAIL及r5冻结源不改写，r5未采用main。

已静态定位公开`compat/torch/autograd.py::grad`缺少返回梯度按对应input dtype转换；历史Qwen树有独立9行块，但同文件保存版本逻辑较旧，禁止整文件回灌。下一项仅在新隔离r6中移植该通用块，放在unused/materialize后且detach前，保留create_graph和当前storage/version实现；root已按独立审阅补丁在新隔离`torch-compat-storage-silu-cat-grad-20261028-r6`应用该9行块；相对r5仅autograd.py改变、无增删，saved-version前缀逐字保持。source fingerprint `5f4107c50f50c2922f7f9c23c9b8243f8269b95f06f8ee094800a94fb57c9f37`，`20261028-qwen-cat-grad-r6-source-r1/candidate-source-manifest.json` SHA `131c2edcac1e7a8ca0e60429dffbc3bb7e7914e37718f51076d6c41f31f2937f`，root source review `0a676aa3…`。仅源码身份；新CPU600 cold包manifest `bd67c774…`已由root唯一串行启动，尚无终态/数值证据，CUDA900继续HOLD。原14→CPU cat_stack1→既有graph/retain/detach/unused语义5→CUDA14的后继模板manifest `977d1354…`保持真实products/env/cache缺失HOLD；各阶段上限360/180/180/360秒，新增5项只属定向维护，不改模型/T5/Vision预算。性能后继仍HOLD：`20261028-day45-r5-source-hold-template-r1/` manifest575e1749…只提供源提案/三轮六阶段及B224S512单步容量规格，不是正式测量；单步实际入口已在新仓外`20261028-day45-capacity-singleupdate-implementation-r1/`实现，tool manifest `b6db2065bf2513dd3855af14c33d3d94f2bb606099b9d831d613459578b88a39`，40项纯stdlib synthetic/AST调度/计数/边界契约通过且21冻结文件hash核验。共用原BERT加载/固定masters与inputs/真实update/memory helpers，每侧恰好一update，0warmup/hot/attribution，低于80% live或6144MiB headroom即停；并非模型/显存或速度证据。最终source/adoption/原Qwen成功收据/环境/设备仍未绑定，runtime目录不存在，禁止执行；旧包和失败原件均保持。Day2–5仍未完成，其他FAIL/预算原样保持。

## 2026-10-28 02:19 +0800 cat旧路径CPU/CUDA复现与隔离r5

发布仍 **Not complete**。旧cat控制现已取得独立PT CPU/CUDA各14项通过，以及旧JT CPU/真实CUDA各10fail/4pass的完整终态；JT失败集中在原cat值/混合dtype与二阶dtype断言，不能声称失败case已执行后续rotate/梯度。CUDA JT单次112.839s自然rc0，实际device与native dispatch一致、fallback0；source/core/stamp/env exact，cache只新增96个允许JIT文件、无改删，组空/精确锁空。PT CUDA原wrapper因公开CUDA_MODULE_LOADING=LAZY元数据守卫失败的原件保持，独立metadata审计f2a1098e仅接受该已完成14项结果，不回写PASS。root新终审r2 `b5331c68f80a68318d5e2a05143d335b17bc49fac168d1c5cddeaf47a3d9854b`的59检查全真；r1将该PT子进程rc1误当自然rc0的审计失败另保留。

root按verify-then-fix新建隔离`torch-compat-storage-silu-cat-20261028-r5`，完整逐文件继承r4，仅dense compat cat调用原生concat前在公开autocast开启时暂挂amp_reg，另加入原14项测试原字节；未改rsqrt/mean/RMSNorm或采用main。source fingerprint `25708e7d45537a4090daa77e985bb93f61d01d6f003c445ec92e54cc2ba03bd6`，未版本化`20261028-qwen-cat-r5-source-r1/candidate-source-manifest.json` SHA `c6e4d2fa83d2fdf7acd2d8306365e7699d593eb8938a806a7834cc6ffbc79b12`；root source review `089fe00f…`确认只改一源/新增一测试、无删除。CPU cold现已自然rc0（约337.041s），receipt `5f8e5301…`；227TU、新core `c91614bf…`/utils `736e3418…`/stamp `e5e2c0c9…`及完整环境/组/精确锁均通过，root17项终审 `a8bb58fa…`。这仅为构建身份，尚无r5修复数值证据。下一项单独审核新CPU600s fresh cold包，再CPU原14及最小邻近；CUDA900s cold与原14分别后继，均禁止自动retry/模型。只有原14通过后才审原完整no-hook，LoRA继续HOLD。

fresh fetch目标远端`e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`仍为六相关HEAD祖先；main起点`94bc5e15208c584d0c176513d29e1a5cf82f2519`、原96dirty和空index保留，fetch前后六树逐dirty文件hash相同，未整合。现场原cat阶段已退出，未启动重复实验；根盘余约84GiB、数据盘余约6TiB。续接快照在未版本化`20261028-root-cat-continuation-r2/remote-sync.json`与`worktrees-before.json`。T5原余额51.29068389162421s/禁止重试、Vision余310.095694191s/无合法最终continuous、BERT原gap15.425953905%>15%/缺11份PT checkpoint，以及Day4–5正式稳定三轮/容量/峰值门槛保持；性能模板仅离线准备。

## 2026-10-28 01:57 +0800 SiLU后继原完整Qwen数值仍FAIL

新包`20261028-qwen-silu-r4-nohook-successor-r2-prepared/` manifest `d93a36a0fdde79a46997aae0b3dce720bb730d56ca64724880b3f9fdfbc46490`经root核验22文件与CPU3/CUDA4/root前置、1150终态cache exact后单次执行。worker/comparator自然rc0、分别250.166/3.117s，未达1800/120s cap；正式receipt SHA `544f7f9d7d42549653e61da1b99aa05ccc73b5dd5c312e8126c3c64b14827949`终态`FULL_NOHOOK_NUMERICAL_FAILURE_RETAINED`，comparison SHA `2fc9e50c7399be0da4ba9d0983f80ec87a666e620bb6f6fb427d13a1aa404677`。原29,171,712全logits含padding有10,119,974项超0.005+0.03|PT|、max_abs4.587890625；有效22,030,720项有4,375,563超限、max_abs0.4375，padding7,140,992项有5,744,411超限。三输入exact，loss JT2.6934099197387695/PT2.6900217533111572、abs0.0033881664276123047满足原门槛；仅logits失败，不能删padding/改阈值。七数值文件保持原字节、独立PT oracle43a5c1e9保持，原失败不改。

root独立终审`20261028-root-silu-r4-continuation-r1/root-nohook-terminal-audit.json` SHA `a0da43740e2e3692a8446badddbb530cb0d26ef128d83b860b8c04c79420f883`的28项检查全真，分块独立复算全部输入/logits/loss及3,393,527 exact logits，不依赖comparator PASS标志。实际CUDA0/fallback0、source b3c6c32a/core b1e8991a、完整环境、actual compile-order/ingredients及所有工具exact；NO_BUILD无core重编、仅新增248允许jit文件，无改删/非法新增，自有controller与两stage组空、精确锁无内核holder。原主树dirty文件除本三文档均未变，候选未采用main，没有LoRA/训练/自动retry。

SiLU-only已修复其真实CPU/CUDA局部缺陷，但不足以关闭当前完整模型门禁；不可由FAIL计数小幅变化声称已证明模型因果修复。离线建议`20261028-qwen-r4-nohook-fail-next-control-offline-r1/review.md` SHA abb33230e931fd5547a39523f014440219de3e53400ac316484171a6eb5af0ea仅核验旧差异中的8个forward文件，下一假说为compat cat在autocast下的原生amp_reg泄漏。先原字节复用`test_cat_autocast_precision.py`的14case（12组合各cat/rotate，另2二阶）完成独立PT/旧JT CPU/真实CUDA旧端最小证据，未复现即停止；未开始cat修改或新模型。rsqrt/mean/RMSNorm/历史工具链仍未排除，不叠加。Day2–5、BERT/T5/Vision/GC及Day4–5门槛与原预算保持。

## 2026-10-28 01:46 +0800 SiLU CPU与真实CUDA原精确门槛通过

隔离r4 source b3c6c32a、core b1e8991a的新CUDA numeric、维护、FP32 activation、GroupNorm融合/残差回归均单次自然rc0。四个原23点unit低精度/native-AMP行的forward/input-gradient对独立PT及原量化数学exact，nonunit补充也全部exact；FP32保持原路径与差异，没有诊断退化。numeric receipt5c0200ff8754d6012bec314df0da1a3d3c114d7710a8a00d62f4851bb5e83e25；原selected维护2/activation1/groupnorm2为5pass/0skip/0xfail，receipts3cf44bfd/f4a8787d/b8497192。source/core/utils/stamp及完整环境/actual compile-order/ingredients exact，fallback0，四阶段仅新增允许jit文件60/8/4/136、无改删；自有组空、精确锁空。

root39项终审20261028-root-silu-r4-continuation-r1/root-silu-cuda-terminal-audit.json SHA870851c0be1c09d1400267483f87534a7c3feccf3c30dda67362e6bc837dd6ff独立复算原unit数学，明确仅局部门槛通过；CPU28项终审a1fcde78及此前CPU2+1回归保持。终态CUDA cache基线来自groupnorm/cache-after，不能采用bootstrap或旧r3 baseline。下一项是绑定这些真实收据/产品/终态cache的新原full no-hook后继，保持七数值文件原字节、29,171,712全logits含padding及原0.005/0.03、worker1800/comparator120。此刻尚无新完整模型结果，候选未采用main、LoRA不启动。Day2–5仍未完成，其余FAIL/预算保持。

## 2026-10-28 01:40 +0800 CUDA冷构建成功，只确认身份

20261028-qwen-silu-r4-cuda-bootstrap-successor-r1-prepared/attempt-01-cuda在401.88535046577454s内自然rc0，正式receipt SHA a45491fa9d8bdc674b69df90bd9ee85b3e78b0f10fe3b3253295d6865d436aca，状态PASS_COLD_BUILD_IDENTITY_ONLY，未执行张量/测试/模型。两次fresh审计231TU（动态统计、不固定旧count）、完整40启动/51导入后环境字段、source及原生stamp通过。has_cuda两阶段true/use_cuda两阶段1，环境use_cuda0保留，符合公开shim自动enable；不外推静态环境1启动。

新core SHA b1e8991ac775ed7c7e0fe897783d01440da7d09da783979f14e3f7c4cef6ac10、utils24b09003e71eca7fa94c451e62f2e120dc92d3896daeafc5da2a575c223bab15、stamp2e157b0b3964b471245705d2fd9544e6d5d66130b37bd9c3897ec685b16cf01c，均来自此次全新cache，不迁移旧core或写stamp。root终审root-cuda-bootstrap-terminal-audit.json SHA6b87523fec1cba3cbcf2d93bcacf5e5bca3631e38be296c14d2f17784e07f1d9复核产品SHA/mtime/size、source/env/audit/无tensor、自有两组空及精确锁无kernel holder。原CPU cold-stage FAIL保持，已通过CPU元数据及数值无需重跑。下一项是新CUDA-only数值后继，绑定上述实际产物/环境及NO_BUILD，继承原独立PT；numeric、维护2、activation1、GroupNorm邻近2分别单次放行。此刻无CUDA数值或原完整no-hook新结果，Day2–5及其他FAIL/预算不变。

## 2026-10-28 01:33 +0800 r4 SiLU CPU精确对拍与回归通过

新CPU metadata后继manifest f7d802ab、正式receipt 9c5c57a5e919acee38237f4eb07d10170b6bd635c171ba78f73127914e549bf0通过，NO_BUILD采用本次r4产物；完整导入前后环境/实际227顺序及ingredients exact，732文件cache无增删改，组空/锁空。原冷构建的环境误判FAIL保持原件。root14检查终审d60478ab记录此CPU元数据结论，不替代数值。

独立CPU数值后继20261028-qwen-silu-r4-numeric-cpu-r1-prepared/ manifest75abb723以原PT和旧JT10行逐值对拍，新JT10行完整。四个原23点unit低精度/native-AMP行的前向与输入梯度对独立PT和原量化数学均exact；BF16(4.59375)已正确为4.5625。正式numeric receipt5620f81d3474544630351771f973804adbf94d8e9412729b83ed6a4d6ebf88ce；非unit/FP32诊断无新增退化。原维护CPU两项与FP32 activation邻近一项分别2pass、1pass、0skip/xfail，receipts df832cde/5a7e74b3。fallback0、source/core/stamp不变，仅新增允许jit产物60/0/18，无改删。root28项独立终审root-silu-cpu-terminal-audit.json SHA a1fcde78a838d3d842cee4b3a76c33c607ae465a8cfb1b201efb17d78592dea8复算原unit数学并核对所有执行绑定。

CUDA新cold successor20261028-qwen-silu-r4-cuda-bootstrap-successor-r1-prepared/ manifest40a9c54d已由root唯一串行释放，CPU前置绑定实际PASS、新cache/不采用旧core、原900s上限。首次准入固定UUID空闲51296MiB，不按utilization。当前仅cold import/编译阶段，尚无CUDA数值或完整no-hook结果；不得把CPU精确对拍外推CUDA或模型。Day2–5仍未完成，其余FAIL与预算保持。

## 2026-10-28 01:24 +0800 CPU冷构建终态与工具环境误判

fresh fetch目标远端origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，已为main、原验收/fix/Qwen、storage-r3及SiLU-r4各HEAD祖先，无需整合。main起点298bb8422e24f50958c31e479ed867d2ed82909a，96项dirty及空index保存于未版本化20261028-root-silu-r4-continuation-r1/main-dirty-before.json。其余五树分别51/85/87/109/111项dirty，未清理或覆盖；根盘余约85GiB、data6余6TiB，共享GPU不作为正式性能证据。

此前203目标文件的进度已终结：20261028-qwen-silu-r4-bootstrap-r1-prepared/attempt-01-cpu完成227TU（190仓内/37生成），两次fresh-build审计passed。新core SHA 487a8fd8b8a8a4e4c534e0f29fd35f70427fb9983c1f8c87cc3f907901db8aa4、utils24b09003e71eca7fa94c451e62f2e120dc92d3896daeafc5da2a575c223bab15、stamp2078dac10c822a3330577720f82b3968ad8d7723a86328035219e7421d5f4f96。正式receipt SHA 485d77dfe06eac3b05254d7ded547e4c0bdec4ccf1e5c4887e666747b0aaa637及child00500cf191d9eedf6e30d0a45eba5c64a59b659a82c7834d735db26a365c9938保留COLD_BOOTSTRAP_FAILURE/rc1；约344s、未超时、没有测试或张量执行。工具在shim_identity_import后要求原HOME/TMPDIR，而冻结公开compat/shim/preflight.py明确重定向它们，因此post环境守卫失败。两次core/utils/stamp、compile-order、源码审计均通过，旧FAIL不改为PASS。

root纯stdlib14项终审root-cpu-cold-terminal-audit.json SHA da27fee8c2fdd2f69e59d11712b139ace1d21b2fcb0a5ee049ec32982804b061核对收据、产品SHA/mtime/size及source均exact，自有controller/child组空，精确锁inode32849971无kernel holder。下一项新metadata后继仅采用这次CPU产物/原cache并设置NO_BUILD，按公开shim精确推导完整环境，不重冷编、不泛放行变化。CUDA900s尚未启动，依赖有效CPU后继；之后独立SiLU数值、维护/邻近回归、原完整no-hook分别放行。numeric和nohook的r4准备包均缺产品/成功前置绑定、HOLD且无runtime。Day2–5未完成；T5/Vision原预算/禁止重试、BERT原gap/缺checkpoint、Day4–5稳定三轮/容量/峰值条件保持。

## 2026-10-28 01:10 +0800 SiLU前向/输入梯度独立CPU与真实CUDA旧失败

遵循此前离线因果审阅的SiLU-only优先顺序，未在storage修复后直接重跑完整模型。新`20261028-qwen-silu-current-r3-r1-prepared/`沿用source `2dad903f…`与已通过cores/env，manifest SHA `c3fdde08e7b42c517413572f66aa4b7159d9c4800e4c5f0042e8ee3ea009af9b`。root在offline comparator正负控/dry通过后唯一串行执行PT CPU→JT CPU→PT真实CUDA→JT真实CUDA，cap120/180/120/360s、无超时或重试。每侧/device10行含4原23点unit-cotangent低精度fixture（BF16/FP16、native/AMP）、4非unit补充和2FP32诊断；原unit/math门槛仍exact，补充行不杜撰容差。正式receipt SHA `6f895f7b07c1cf02ece16a68b8be1961538103ab87dfceb119bf32ef72d510c7`为`OLD_SILU_FAILURE_REPRODUCED_CPU_AND_REAL_CUDA`，不是修复或发布PASS。

CPU和真实CUDA各四个原unit行均独立PT与原量化数学的forward/input-gradient exact，JT两字段均有差异。BF16前向每mode3项差异、max_abs0.03125，4.59375点JT4.53125/PT4.5625；CPU BF16梯度每mode2项、CUDA6项。FP16前向每mode3项，CPU输入梯度9项、CUDA8项。每runtime实际device及native dispatch确认，JT fallback0；新源码或core无修改，CPU/CUDA各新增60个约定算子JIT产物、既有缓存无改删，四组空、精确锁无内核持有记录。root以标准库重新计算原unit量化解析值并核对独立PT，终审`root-terminal-audit.json` SHA `8aaa4665ca5d343ba8b001c9f31804adb1289d6ea113a87bd8928a23c4c211f9`，确认真实旧失败。

root据此创建新隔离worktree `torch-compat-storage-silu-20261028-r4`，HEAD仍`765f1791f…`，完整复制2804项r3文件/链接并逐项核验；相对r3仅activation.py采用此前已保存的8行SiLU FP32中间计算/一次输出舍入补丁，另新增原维护test_silu_low_precision.py原件，无删除。新source fingerprint `b3c6c32a66bd5243d5a739ccd15f5b651160d4a6df5c5487dda4a76491bdada4`；`20261028-qwen-silu-r4-source-r1/candidate-source-manifest.json` SHA `ea42f5e734a262eef9bebc34b45c54c7538011cc22bb31e944789f186aac0a65`覆盖2804 regular/2805 candidate records及compat链接。main/r3与旧cohort均未改，不叠加rsqrt/mean/T5修复。新root禁止迁移旧core或重写stamp。

`20261028-qwen-silu-r4-bootstrap-r1-prepared/` manifest SHA `c52ba6291fd214e379e42dc1d1d68c7afbbc7b0b7665804bddc8dc815783604a`在真实旧失败/root review/source全绑定后通过dry；CPU600、CUDA900秒分别继承原cold-stage manifest/plan，有明确路径/哈希/字段，不追加模型预算。逐device新cache，动态source/generator/compile_order/stamp检查；CPU包括合法indexing_schedule_codegen.cc，CUDA core/utils父子目录按既有compiler规则核验。本快照root正持有唯一CPU cold-stage，CUDA尚未释放；构建成功也只证明产物身份，修复后的独立PT/JT数值、维护/邻近回归及原完整no-hook均未验证。

原`20261028-qwen-storage-r3-nohook-r1-prepared/`仅离线准备，manifest `e1ae4a7a…`、dry通过但execution_hold明确缺SiLU old/new CPU/CUDA证据，attempt不存在；原数值worker/probe/support/comparator字节不改。待r4数值前置真正通过，须另建绑定新source/core/cache的后继，不能清改旧HOLD包或自动启动LoRA。Day4–5新只读审阅`20261028-day45-current-tool-review-r1/` manifest `899f35820ae42a33a9d03dea97f2b718a2691ee7c6d8a54323c1309b975596df`确认formal-r3只完成60s readiness、12样本0合格，未产生正式PT/JT性能行；旧source/head不能重绑。当前仍缺三轮代表格、真实容量、峰值与常用路径归因；T5/Vision原预算、BERT失败及checkpoint缺失保持。

## 2026-10-28 00:45 +0800 有界lifecycle独立CPU与真实CUDA通过

发布仍 **Not complete**。本轮fresh fetch记录`origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为main及原验收/fix/Qwen/storage-r3各HEAD祖先，无需整合。main起点`ef09b62332521c3f226dc1b73870bafea61ce1c8`，原96项dirty（99个文件哈希）及空index保存于新包`root-continuation-workspace-before.json`，独立终审确认原文件均未变；runtime源码未改。原验收/fix/Qwen树分别51/85/87项dirty，storage-r3树109项，均未清理。根盘约余87GiB、data6约余6TiB，GPU按固定UUID/空闲显存准入，不按utilization；共享负载不能证明性能。

新未版本化包`20261028-qwen-storage-r3-lifecycle-r1/`的manifest SHA `a8ea56ca354d83fb1f16264d16aa60f876b94b187477b4b6756ec57dd6ace3a3`，冻结source仍`2dad903f777e53c0c971d514e1bf1100be72d03b2203f9277ada3a9e817e4454`。root dry守卫通过后唯一串行执行PT CPU、JT CPU、PT真实CUDA、JT真实CUDA，cap分别120/180/120/360s，无超时/重试。正式`attempt-01/receipt.json` SHA `5319668de6bce14cfd9e45b02bf4c38295e06813d29d16c0930476945035915f`终态`PASS_BOUNDED_LIFECYCLE_CPU_AND_REAL_CUDA`，两device逐字段differences=[]。未重复原48或维护22，不自动扩为新的合并矩阵。

每runtime/device两个数值case各重复三次。非求导常量c=[-0.25,0.5]、两个求导权重的loss分别使用独立exp(c)/exp(2c)，或共享z=exp(c)与z.square()并删除z；先loss.item()物化，再公开no_grad/c.add_(1)，旧loss保持，梯度先于source/中间值读回。两梯度仍为旧exp(c)/exp(2c)，source写后[0.75,1.5]，独立PT及解析值均通过1e-12+1e-12*abs(reference)。JT另有每device两个kept边界：无关共享存储旁观图不重放，已执行的相关kept reader以既有精确RuntimeError在写前拒绝且target仍[2,3]。观测不读取kept输出；清理明确沿用维护fixture的_release_kept接口，不声称公开API。六次数值及两个kept清理后live_vars均恢复进入前0；每JT/device35份张量记录全部为对应实际device/native dispatch，fallback0。

沿用已验证CPU/CUDA core、utils、stamp及原cache/environment，JITTOR_NO_BUILD=1。实际CUDA import后has_cuda=true/use_cuda=1、环境use_cuda=0；不外推静态环境1启动。source、runtime compiler.files、core/utils/stamp守卫全通过，CPU/CUDA各仅新增16个约定后缀算子JIT文件，既有产物无改删；两个精确运行锁只允许owner记录更新，inode/权限/属主不变。四自有进程组终态均空，两个锁在/proc/locks无持有记录。root独立`root-terminal-audit.json` SHA `3e1fb7522a28bc52a5243d5046af761ac942da1c6a89cdab5f4522ab9cd852e4`重新核验完整结果/设备/liveness/cache/source及main原文件哈希。

上述最小lifecycle缺口已关闭。候选仍未采用main；下一项只准备绑定同source/已验证CUDA core的原完整no-hook worker与comparator，不改变参考、阈值、padding或全部logits范围。当前没有新完整模型或LoRA证据。原main no-hook FAIL10,338,462/29,171,712、T5唯一单micro超时/余51.29068389162421s且禁止重试、Vision余310.095694191s且无合法最终continuous、BERT15.425953905%>15%及缺11份PT checkpoint、Day4–5正式稳定窗口/三轮/容量/峰值均保持原状态。

## 2026-10-28 00:31 +0800 维护22项真实CUDA终态通过

发布仍 **Not complete**。隔离storage r3/source `2dad903f777e53c0c971d514e1bf1100be72d03b2203f9277ada3a9e817e4454`现已完成原48项CPU、维护22项CPU，以及同维护文件22项/device的真实CUDA验证；这是不同固定范围的证据，不把它们改称新的合并矩阵。新`20261028-qwen-storage-version-r3-cuda22-lock-metadata-successor-r3-prepared/` manifest SHA `a045fc77954a1aab6547e061c4cf51a7deee92ac1775b295558d02dee2bee9c2`；仅启动一个新JT进程，继承旧已通过独立PT CUDA22的原meta/JUnit/物理UUID/nodeids，不重跑PT或CPU部分。正式receipt SHA `9f0ca4514d13a30a79af3f9eae0fdb2c47dbf4f5ddf1c406a245f1cf6f3f4f0a`为`PASS_MAINTAINED22_REAL_CUDA_JT_WITH_INHERITED_PT`；JT meta SHA `d80ae430c0e110b2b1d6b414289d7f595f09871bc035b5e9b9fd5e8e549efb9b`、JUnit SHA `acc4f6cb9f2e0c8b6b0ae57f873c5487294d3694fd297c70f46acb79854d247b`。PT/JT各22 passed、22 CPU项deselected、0skip/error/failure，同选中nodeids；JT 66份test-return张量记录均为cuda:0、native dispatch=['cuda',0]、fallback_count=0。

显式采用本次真实CUDA core `16a21ef12a2b75f15d99b54c03c388628a3940dd8f8bb67867d4b1ceb12aa632`、utils `4eeb093bbabc70593da6138c67aef1d7b27562f6be9ec3e76b25b6f45dd46198`与stamp `93a33c63d94903185fd89ca970e00bc6e8d57eba1ba70c34b02b100d7d137b5f`。保持原采用环境/cache/runtime路径，增加JITTOR_NO_BUILD=1禁止core/custom-op扩展重编；只允许同CUDA cache/jit下约定后缀的新算子产物。两次import后分别实测jt/compiler/build_config has_cuda均true、flags.use_cuda=1、环境use_cuda='0'、math=strict；这确认公开composition的自动enable，不回填旧进程缺失值，也不宣称静态环境use_cuda=1启动通过。实际runtime compiler.files、1225候选源文件、779 stamp源签名、8 generators与231单元（194仓内/37生成）通过。原core/utils/stamp不变，新增64个允许JIT产物；编译/生成产物无改删。共享负载下的这次正确性执行不构成速度或峰值显存验收。

工具失败链全部保留。先前JT-only metadata-r1 receipt `e6470b9120ccf88c7620cd1c85e1f064e1e399277615f34e13741c5193b2dcba`在数值前因正常jittor.lock owner记录更新被误判缓存漂移；其已记录真实import值、构建检查无错误，不能将它改成测试PASS。冻结源码`src/runtime/lock.cc:91–108,156`在取得flock后ftruncate/pwrite持有者JSON，因此最终新包只对绑定的这个精确路径区分运行元数据，保留before/after记录，严格检查device/inode/mode/uid/gid，禁止替换或unlink锁；所有其他缓存文件仍逐字节/元数据校验。纯stdlib控制确认锁内容更新可通过、换inode或改core内容会被捕获。中间r2的差异文件在manifest冻结后生成造成工具哈希不匹配，执行前即停止、无attempt；原件保留，最终r3在生成全部文件后冻结并通过守卫。没有放宽数值断言、跳过case、改source或重冷编。

root独立终审位于未版本化`20261027-root-maintained22-continuation-r1/root-final-cuda22-terminal-audit.json`，SHA `ccc3810292fb45a4f0354e1f6e6872c145e4a91c6877ee7b88c26e075c7996a3`：同nodeids、完整JUnit、66份CUDA证据、fallback0、source/build/cache通过，自有组空，精确锁文件无内核持有记录。main本段起点`4a6988f403a23ac43d9a8b6eb00e92e24d71f083`为此前三文档提交；原96项dirty和runtime源码保持，r3候选尚未采用main。下一项仍须先关闭两组最小lifecycle缺口：多个无holder saved-root含不求导常量及重复图释放；compat eager-write面对kept旁观图的隔离及相关reader的拒绝/写入原子性。现有22不能代替这些证据，未自动启动完整Qwen或LoRA。原main no-hook logits FAIL10,338,462/29,171,712、T5单micro超时/余51.29068389162421s且不重试、Vision余310.095694191s且无合法最终continuous、BERT gap15.425953905%>15%及缺11份PT checkpoint、Day4–5正式稳定窗口/三轮/容量/峰值门槛均保持原状态。

## 2026-10-28 CUDA22冷构建完成，但导入后守卫先于测试失败

root核验维护CPU22成功receipt后，审阅并唯一串行释放`20261027-qwen-storage-version-r3-maintained-cuda22-r1-prepared/`，manifest SHA `70d45275e312b711854abf5766bb6a421038e01ca0aceaba8ab7a6fa6b0e032f`，候选source仍`2dad903f…`。采用新独立CUDA cache、固定nvcc12.4、环境use_cuda=0导入、公开scope1的拟议测试协议；GPU按UUID及足够显存准入，不按利用率。PT真实CUDA22 passed/22 CPU deselected/0skip，meta SHA `437763d913c6a8f6deddcebee49930d9f5823d6f355735664912ea4f52333b4e`。JT完成真实CUDA core及扩展后，在`child.py:57`的`not has_cuda or use_cuda!=0`守卫处rc1，未collection/执行任何维护测试；overall SHA `8540056b7993d6be89d78efd35c6048fede386fecf3d0499b3206d5e7dd73c5c`、JT meta SHA `f2fa68294297b07692f5369e065c12dabe43db7b75bb455580a62794cf6dce36`。旧child没有分别记录实际两值，因此不能把它当CUDA数值FAIL、推断CPU core或回填具体动态flag。

离线源码核验发现`compat/torch/installers/core.py:153–168`公开shim安装在CUDA可见时主动设置use_cuda=1；工具要求安装后仍0与既有规则冲突。另`compiler.py:371`先在jit_utils.cache_path生成utils，685才给core cache附CUDA子目录，旧`fresh_build.py`的同目录前置不适用于此CUDA布局。实际新core SHA `16a21ef12a2b75f15d99b54c03c388628a3940dd8f8bb67867d4b1ceb12aa632`、utils SHA `4eeb093bbabc70593da6138c67aef1d7b27562f6be9ec3e76b25b6f45dd46198`；utils虽与CPU helper内容相同，但来自本次新cache，不是采用CPU core。root纯stdlib事后检查得到779源签名、231编译单元（194仓内/37生成）和CUDA ingredients exact，旧guard仅报`core_utils_cache_mismatch`；审计SHA `f9c2d5902c5ea7693e30eb43bca71e99528e18cb2c9d0dc763155351ad9eef70`明确compile_order取自stamp，不能冒充未记录的runtime compiler.files观测。

正式失败receipt/source/tools不改，自有PT/JT组空、精确JITTOR_HOME下jittor.lock无内核锁记录；未重试、未重冷编、未采用main。metadata-only新后继仅离线准备：显式采用本次CUDA core/cache，继承已通过PT22，记录post-jittor/post-torch真实has_cuda/use_cuda，使用公开scope1并另行审阅；不静默叠加T5初始化修复，也不声称覆盖静态use_cuda=1启动。离线lifecycle审阅还保留两组最小缺口：多个无holder saved-root含stop-grad常量及图释放、compat eager-write下kept bystander隔离/相关reader拒绝；现有22不能替代这些证据。原48/维护22 CPU PASS保持，完整Qwen/LoRA、T5、Vision、BERT与Day4–5发布门槛不变。

## 2026-10-27 Qwen原48与维护22项CPU后继终态

发布仍 **Not complete**。本轮fresh fetch `origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`为main、原验收、fix、Qwen诊断及storage r3树祖先，无需整合；main起点`8a2e9b74e8c8d8c8cb7091eb829c500c53efe085`、原96项dirty及空index保留，runtime源码未改。现场、launch审阅与终态独立核验写入未版本化`20261027-root-maintained22-continuation-r1/`，根盘约余90GiB、data6约余6TiB。GPU共享负载不阻止显存充足的正确性验证，但不满足正式性能结论。

冻结storage r3 source `2dad903f777e53c0c971d514e1bf1100be72d03b2203f9277ada3a9e817e4454`的原48项CPU已在`20261027-qwen-storage-version-r3-original48-maintained-cpu-r1-prepared/`完成独立PT/JT逐字段对拍：每侧48项，differences=[]、真实CPU/fallback0；pair receipt SHA `83c8acf15651993820717e5949c9dff84da75821772e3c7634905a31d3b6b927`，PT raw `492877422e9d772235c35b7af02e155ab56d8a5c19f0527183abc9831a771abf`、JT raw `5cd46d3b8bf21a1a207561bdac339a2d292dc7c272893d52c56fc0d962f6fe3e`。随后维护22项在PT pytest注册`pytest_runtest_logreport(result)`时发生PluginValidationError，零collection/测试，JT维护阶段未启动；overall失败receipt `c7e50e5576df534a5fe54ea2a2f23fcfd1f48ca0d95deb89296b2b68ec7a28d1`保留。原48脚本/CSV未改、已通过部分不重跑，该CPU-only探针不能改use_cuda冒充CUDA。

新`20261027-qwen-storage-version-r3-maintained22-cpu-hookfix-r1-prepared/`仅修实际hook形参及体引用`result→report`、控制器缩为维护22-only；manifest SHA `59fc174277c06260a5893559225c772743f0ad0dd385a6379ccd3cbf45f280ff`。两独立pytest环境实际hookspec注册smoke先通过，不collect、不导入框架；root审阅后唯一串行执行，正式receipt SHA `17522991df431ab8003046c6ec74b95408ace1aef8d448bed0ecea06621088e5`为`PASS_MAINTAINED22_CPU_ONLY`。PT/JT各22 passed、22 CUDA项deselected、0 skip/error/failure，同nodeids，每侧66份test-return tensor记录确认CPU，JT native dispatch=CPU/fallback0。JUnit SHA分别为`3674b462ef042b0e1e8d595d9bf8615e731906b4865fc6b1e93335e7d8637054`、`2c267161e371acc3b51f40b95ea3eb4577b5743652837d15c8d69e3dea90cf21`；core `b58f1709…`、utils `4eeb093b…`、stamp `d49c7d30…`与1225源码守卫均exact，缓存从794仅新增20、无改删，自有进程组空。精确adopted-home jittor.lock在内核锁表无持有记录，未据此声称全机所有不可读进程无holder。

计数是既有维护文件22/device（原20+新2），并非原48扩到50或52。此时CUDA22包仅离线准备，须绑定CPU成功receipt、独立真实CUDA core与显存准入后另行释放；r3未叠加T5初始化修复，合法协议为has_cuda=1/use_cuda=0导入后公开scope(use_cuda=1,backend_fallback="error")，不证明静态use_cuda=1启动。候选未采用main，lifecycle覆盖、完整Qwen原no-hook logits、LoRA与所有发布FAIL保持。T5 `TERMINAL_INDEPENDENT_AUDIT.json` SHA `bb77e028dccef5667b44aa2c80dd401112f934f1f6f4691a7d283b13dca9aa5f`事后验证75 bindings与2817源文件/权限/软链接exact，不改正式receipt的退出校验false；唯一单micro超时、预算余额51.29068389162421s与禁止重试边界不变。

## 2026-10-27 23:29 +0800 T5单micro超时；Qwen原生修复双例CPU通过

Day2–5仍未完成。重新fetch origin/cgq_transformers=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，main及三个原验收/诊断树均已包含，无需整合。main HEAD c9d9633277ba0465bc9fa06f6ade798e16f34255、99项dirty/空index保留；根盘余91GiB、data6余6TiB。只备份/更新本三文档，未改runtime源码。

T5唯一20261027-t5-step51-singlemicro-ordered3-r1已从合法checkpoint50进入真实CUDA，但在210s工作deadline仍编译新kernel，controller核对自有进程身份后SIGTERM，worker rc-15/timed_out，终态FAILURE_RETAINED_NO_SUCCESSOR。receipt SHA a6b3aef207994256dc44a5b977f7ae6820f556fcdbff42e390a6fb6f932901dd；仅留下preforward参数签名，无first-micro NPZ、404 trace、完整梯度或比较，不能声称模型恢复/因果。manifest7bc020d31c9f9ccfaae1065659200317419a4860282a44e3a63c77d8044577b9/source715081d6不变。收费212.9110761990305s，原1800s账累计1748.7093161083758s、余51.29068389162421s，无finalization overrun、重试或自动后继。正式receipt的source_and_tools_verified_at_exit=false仍保留；root另写root-terminal-audit.json，重新验证完整bindings exact、自有组空、可见cache holder空。1458个拒读PID记录完整，因此不证明全机holder绝对空。启动后补查发现report直接序列化projection_dtypes/observed_autocast集合的工具缺陷；本次未到report终写，未动态触发，不把它当本次timeout原因，冻结tools不在线修复。纯标准库comparator实物控制8071c078…确认2803自比exact、原continuous/resume119差异、404/RNG2536/input5exact及旧protocol gate=false，仅为工具证据。

Qwen隔离storage r3 source2dad903f777e53c0c971d514e1bf1100be72d03b2203f9277ada3a9e817e4454、manifest3e7ebd441e60c93b2506b30447784091f919ff8eda178c2028586aa56e85038d，相对r2仅native retained saved-value发现边界和维护回归两文件变化，原生assign/update/rebind、导数、fusion不变。两case数值AST与r2一致，新CPU cache、不采用旧core；T5组清理与冻结绑定审计后，唯一启动attempt-01-cpu，每runtime600s。独立PT两例已PASS；JT完成227/227冷编，但fresh_build_guard将stamp.files中合法仓内backends/cuda/kernels/core/indexing_schedule_codegen.cc误认cache生成物，在两例执行前rc2，终态RUNTIME_EXECUTION_FAILURE、receipt b5293d93ece12a837c73468e14bfd1ef9f281a8820d3e18b40b4d84780106ae4、JT report f0c5034bc752239c32d48409060b9cf0f75e04c1ccc5dc77cfd53df2b2ce8122，cases为空。core b58f170982e3a46addaad973d128aa1e2058d0523a6d8cbc022c36a6ebf1a5eb、utils4eeb093bbabc70593da6138c67aef1d7b27562f6be9ec3e76b25b6f45dd46198、stamp d49c7d3076c7ca7d72494fb946ba2743b2f1639aaf19dcceee6a14d379f89bce均保留；root复验source/pre-post环境通过、自有组空，不能声称r3修复。下一项只在新工具cohort精确修正source/cache分类并绑定现有core，不改旧收据/stamp/source，不扩大原两例；不按活跃队列推导通过。原r2静默错梯度SEMANTIC_FAILURE保留；即使局部通过也不关闭原48/CUDA/no-hook/LoRA或发布门槛。


Qwen新metadata successor只修compile_order分类并显式采用本次r3 core，已PT/JT两例PASS_BOUNDED_CPU_ONLY，receipt da045adbf1cad6979e7cc99605d386c34052fe082d3c74a785d1408da8c0d84b；无holder与保留holder均返回原exp([2,3])梯度，source写后[3,4]、旧loss保持。真实CPU/fallback0，source/core/utils/stamp守卫通过；cache仅新增12、changed/removed0，自有组空。仅局部CPU修复证据，未采用main；原48＋维护22CPU正在离线准备，CUDA/no-hook/LoRA及发布FAIL不变。 metadata manifest49e68b189da399557231f048f3bd92122fc2f1c47c6737de24a330aa9d4cef39/adoption8627fef95506e5632ca0db4b00ffea8239c2af750dedeb8970ba364489904465；root-terminal-audit SHA92d6824135c8992346243323a96b78d2506c9ee0d1495a058153c2538611686d。独立T5终审bb77e028dccef5667b44aa2c80dd401112f934f1f6f4691a7d283b13dca9aa5f核对75bindings/2817文件/core/utils/stamp exact，257 preforward参数记录对continuous/resume双侧exact；无完整前反向、RNG或模型恢复结论，原receipt exit-guard=false未改。

## Status

**Not complete.** Day2–3 remains failed or incomplete at its original gates. Day4–5 has shared-GPU diagnostic data; r4 and r5 bounded waiters have both timed out without launching measurements. No release-valid stable-window result exists. Thresholds, seeds, comparison fields, failure evidence, checkpoints, caches, and worktree changes remain intact.

- Date: 2026-10-26
- Main continuation baseline: integrate/cgq-transformers-native-multicard-target at 765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe before this documentation update; all execution receipts and earlier snapshots retain their original source identities
- Candidate: release/torch-compat-day23-fix-20261021 at 9e0e187bb013e71d85111acfd668e46aeff9f9d2
- Remote sync: origin/cgq_transformers fetched at e5cffe1ae60c418f53c5bb1cdca9182577fd2b81; origin/2.0-refactor not integrated
- Owner: cgq / Jittor Torch compatibility maintainers
- Review when: candidate source, dependencies, fixtures, device, or acceptance protocol changes

## 2026-10-27 22:34 +0800：T5 同源 public norm flag 干预闭环；模型门禁未过

本轮 fresh fetch 的目标远端仍为 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，是主树 `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe` 祖先，无需整合。main/原验收/修复/Qwen树的 porcelain-uall 分别为102/51/85/87项，本地内容与index保留，未整合其他远端分支。以下候选均在仓外隔离树；没有将新候选补丁应用main。

T5 原 CUDA bootstrap SIGSEGV 已从保留 ELF 的 `.init_array` 和故障指令闭环：device initializer位于100/`0x714cc0`，fetch initializer位于145/`0x714e28`；`init_use_cuda → setter_use_cuda → sync_all` 提前遍历尚未构造的全局 fetcher，零next形成`0x10`解引用。`fetcher_to_free`也有同类风险。独立静态审阅及原件摘录位于未版本化 `20261027-t5-fetch-init-elf-review-r1/REVIEW.json`（SHA `415735f3a7e6ca35fbc893ea3cdff94d33002e19101e7ae5b1196aa76c4f370f`）；原失败receipt `994d36…` 未修改。fetch-init r2把两队列放入同一延迟构造的进程期owner，保留显式cleanup；新CPU44/fetch通过且CUDA `use_cuda=1`首导入通过。该source随后在首个ordered kernel的KernelIR序列化失败，独立于旧`plan_reduce_thread_order`崩溃。字符串`push_back(raw=true)`仍解析多语句并产生无`code`属性的非root容器；新r3仅在ordered分支采用既有BlockedReductionPass式opaque code/raw节点，未改全局parser或kernel数学。

最终隔离候选为 `torch-compat-ordered-sum-kir-20261027-r3`，完整2817文件内容身份 `715081d6f57cf8d2032e3e31549b2a59e3f630f003d08392e2f73fe948bb97e4`，manifest SHA `619800d17e5a4040db4db3b7934bfb4245d9091c8c637ce0b91b7aece8eb07ef`。CPU44完整数值/VJP/provider及fresh41fetch/twice-cleanup通过：r7 CPU receipt `e7a9c9a4f292ce382b1ab8abfe38b811482f44a2295be4656a64033d328a3cfa`。新CUDA core `b63e552b18a9721619cc4a5325b104352fbbc58eeb9f2b512e62ba1518fb0651`，保持原`use_cuda=1`；r8 CUDA数值receipt `0fb79f7b27a82026bf690a02731511e3a460b56303960f19edd0b3adde404d4f`通过全部44例、独立PT/FP64对照、全部VJP、provider正负例/compound graph、两个32次对抗归约各1hash、fresh41回调和两次cleanup。实际GPU UUID `GPU-73f08516-9bbe-5f79-7023-a8f9b38cfdc4`、fallback0、source/tools/core守卫通过、退出组与holder均空。此证据只覆盖primitive/初始化/fetch，非T5模型恢复验收。

原1800秒家族所有失败及开销均保留：r4 CPU工具将`location()`误写`host`（实际`cpu`）、r5 CUDA KernelIR失败、r6 CPU cold stage workdeadline及r7 CUDA cold stage workdeadline均保留为FAIL；后继显式审阅并采用相同source/core绝对cache，没有修改旧收据、放宽阈值或重置预算。到r8累计1457.1829295223579秒，余额342.8170704776421秒。`20261027-t5-public-norm-ordered-sum-r2`继承此余额，120秒cap（90work+30cleanup），已正常rc0退出：receipt `8c296ff75e44e8cfc04cb14ddfc56b2435e18e8d47bbba3ffa364a54ec12a10b`，report `0672cfb0e255dcd145999b0bb2fa927cc046e3fe6cf72d9061e750b033180747`。原32fresh public T5LayerNorm/VJP、同冻结输入/gamma/synthetic upstream/FP64oracle/BF16 autocast/para3/epsilon，flag3下96份数组的y/dx/dgamma各1hash且逐字节exact，两个ordered sum provider各32次实际调用，fallback0、source/tools/core未变、组和holder空。对FP64 maxabs分别9.916250398589455e-8、1.4252825968064098e-9、3.6461851529168143e-6；precision gate仍null，synthetic upstream并非原模型cotangent，不能声称模型因果或strict恢复通过。前向mean虽有marker，实际生成代码仍有atomic，不宣称前向mean无atomic。本次收费45.509478067047894秒，原家族累计1502.6924075894058秒、余297.3075924105942秒；同source/core/cache仅flag3→0的新32fresh对照 `20261027-t5-public-norm-ordered-sum-flag0-r1` 已执行并完整审计：manifest `a66ec3f73de25eff5980502ddf73e703b891d99f7f85192c427e21654c570e06`，receipt `524b348c3b8312963afd0b49ec30a4938bb43e1630dd21172c495c2dc4e8b784`，report `cae370314b8b799f16152856fa89c974ec209ef6dadbca1be0354c8430bc5929`。y为1hash，dx/dgamma各32hash，相对首次数值最大差分别4.656612873077393e-10与7.62939453125e-6；实际无ordered-sum provider，96数组完整、fallback0、source/tools/core守卫通过、组/holder空。控制的accepted=true只代表完整执行，分类明确VARIES；与flag3的各1hash构成同源公开norm的局部flag干预证据，仍不是原模型cotangent或strict恢复证据。收费33.10583231993951秒，原1800秒家族累计1535.7982399093453秒、余264.2017600906547秒，无预算重置或自动后继。独立离线审计 `20261027-t5-norm-flag-pair-independent-audit-r1/audit.json`（SHA `3ebf5dd240a1e23cf78156ef4e4f42a0e03cfade9a1f0f6f901c69d5d7f5d1d0`）重解析两臂共192份NPY、全部raw/hash/FP64误差、绑定与provider实物，确认仅compile option为3→0、环境仅工具入口和一次token差异。flag0任意重复对的dgamma最大差为1.1444091796875e-5，先前7.62939453125e-6明确指相对首个重复。旧T5双seed strict FAIL771/2088保持，不从r66/r70启动resume，不重训full100。

Qwen隔离storage-version r2仍未采用：source `b7e60367873449855de48a94134479dcaf493932a435b90218b00293d7e0cca2`，12文件补丁及48格只有静态准备。审阅发现无中间holder的`x.exp().sum()`可能在写入后反向重算旧exp；新两case控制保留/不保留holder。独立PT两case通过，JT冷编成功（core `52ba8db7c21caf8d655e02ae21831dbb16e970d8863b20db4e5eb353950e248e`）但工具强读合法namespace缺席的`torch.__file__`，在数值前失败；receipt `757337b63cc55b210d0ee7cb10d002ecc0a3750746e1ae3f00c8d7ed1936d029`，没有JT语义结论。metadata后继 `20261027-qwen-storage-version-r2-noholder-metadata-successor-r1` 已单次执行：PT两例通过，JT同core导入后因工具把合法HOME/TMPDIR派生当作漂移，在数值前停止（receipt `192697d352ca4bdb6a6eb378581f2923a68c8a2b617cc9e14e6a1198b6c37720`）。实际core/utils/stamp未变；HOME变为原runtime/home、TMPDIR变为原tmp/jt-8c98b3156c，其合法性正按源码核验，后继仅准备且不扩大数值范围。两次工具失败均无JT语义结论，旧收据保留。源码证明shim preflight应把HOME设为runtime/home，并由路径sha1推导短TMPDIR；新r2只区分pre/post import精确环境，72个错误字段负例保留。`20261027-qwen-storage-version-r2-noholder-metadata-successor-r2`（manifest `5b8e54414fcc68b5b696453e1900f07a30561655b8b3193092fb08545b994a2e`）已单次完成原两例，真实CPU/fallback0、同core/utils/stamp/source零漂移、无超时。receipt `8d88976f3033253a59d78dffb600acf747d073dbe8f1c0df749474b0ca873cc2` 为SEMANTIC_FAILURE：PT两例PASS；JT保留exp holder的控制PASS，无holder的`loss=source.exp().sum(); loss.item(); no_grad: source.add_(1); grad(loss,source)`却返回[e^3,e^4]而非[e^2,e^3]。旧forward仍e^2+e^3、source已[3,4]、shared version从0→1，因此这是可复现静默错梯度，r2候选不得采用。与只保留holder的对照形成依赖保存边界定位，不声称已修复；原48项/CUDA/no-hook均未执行，下一步仅在新隔离r3准备native写前saved-value保全修复。main assign/meta补丁仍7/8CPU及direct alias FAIL，完整no-hook FAIL10,338,462/29,171,712，LoRA未启动。静态假报警“重复for”已证实是两次sed边界行重印，非源码问题。

Vision既有continuous终态、310.095694191s余额与cache变化边界不变，未重试；五格仍FAIL。BERT原双seed任务gap15.425953905%>15%仍FAIL。Day4–5后继15项工具纯离线合约通过，新r2后继已按main文档提交c9d963327/相同46a8计算身份离线准备（18项manifest `e8616ca01d3a6d37ef0b4289cbcc5a54665f0b7e5d0ef3e98b550858fc56bfb1`），补齐整程外来PID/startticks/利用率时戳及采样生命周期守卫，readiness最多60秒；尚未绑定最终全树身份或运行新性能/容量格；共享GPU资源观察不能当正式性能或显存结论。Root随后将r2绑定到HEAD c9d963327及完整含dirty文档source `0844601329e98dcf8463c2069bd904737ffaedbf2a09a9e7f7445a607d65bb97`；readiness入口在NVML模块exec时因未预注册sys.modules["pynvml"]而KeyError，未创建runtime或启动观测/计算，外层失败收据SHA `9fedbae5e1aa39c5f0c926e4febc0d30194fde5bf911fdd074994bf7bc785f48`保留。新r3工具只修标准模块加载登记/失败清理，mock自查/错误传播/残留清理与离线合约通过，manifest `6e8e6aaf662d9433018bf707f8b730eb6bfbd974fa279d92f2a01f1709596e18`，同source独立重新绑定后只运行一次60s readiness。终态 `WAIT_TIMEOUT_NO_RELEASE_WINDOW`：12样本、0合格，末次free57471.664MiB/util83%，外来进程负载及缺少近期利用率证据均如实保留；completion SHA `0c3202499608173bc93793e6ea7f5edd9d32fb2eae865543c1a611fbac4a8456`。源码/工具全程exact，无模型、performance cache、measurement或自动重试。此次观测终态后才解除main全树冻结供证据更新；后继不可把该旧完整身份改绑到后来的文档/HEAD。整体Day2–5仍未完成。

## 2026-10-27 20:50 +0800：T5 CPU reduction门禁通过；CUDA bootstrap失败；Qwen别名与Vision仍开放

本轮重新 `fetch origin cgq_transformers` 得到 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，仍为主树 `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe` 的祖先，无需整合。主树脏改动和已有 index 均保留；当前 source-files manifest fingerprint 为 `46a8fbb7f15786462a530bb9a2a855a983ced8e796afeae3fe6b80362ebe2a74`，这是1223份纳入源码/构建清单文件的内容身份，不是Git SHA，也不是包含文档、HEAD及symlink的全树身份。验收worktree HEAD仍为 `9e0e187bb013e71d85111acfd668e46aeff9f9d2`，原树/fix树/Qwen树分别51/85/87项状态记录；main当前99项short-status记录。没有活跃的本任务worker、JIT holder或本地测试。

T5有界ordered-sum候选保持source `930732a5be98e897ff1dfbe49b184c97aa5130b110346e72b714ad44c9bc5e92`、原容差与44例范围。原始首个CPU数值stage因Jittor调用错误`dim=tuple`在首例前失败（PT44例通过）；修正为`dims=tuple`后，下一候选已完成35例，随后发现负例构造的普通transpose被物化为连续tensor，保留断言后停止。使用现有`transpose_storage_view=1`构造真实storage view的新候选以不同工具manifest及新输出完成CPU stage：44/44通过，包括ordered reduction provider路由和普通广播VJP，状态 `NUMERIC_AND_PROVIDER_PASS`，fallback=0；stage receipt SHA `11ff8c2c2b6bbc1c59cdadc8c36c438d0018a924f34b51ef8e93f693103e5b3a`。单独CUDA bootstrap随后在新CUDA cache完成228/228 core编译并输出 `CUDA enabled`，初始化路径于`sync_all`、`setter_use_cuda`、`init_use_cuda`中的Var向量增长处SIGSEGV，worker rc1、无CUDA数值运算、无cache holder；这不是原`plan_reduce_thread_order` step51 crash的复现。失败receipt SHA `994d36b3d1c6b62cc9d90871399ef63046844dc3dded0086ea5af2783438ebee`。原1800秒家族累计532.228476549s、余额1267.771523451s；r1/r2/r3失败目录、成功CPU cache和CUDA崩溃产物全部保留。下步先从已编译core符号与受控最小初始化路径离线定位CUDA开关期间的segfault，再审阅新cohort，不从失败stage自动续跑；原T5严格恢复FAIL771/2088、无step51连续trace/checkpoint的resume限制保持。

Qwen assign/storage问题重新CPU复核了4种Tensor/Parameter→Parameter/buffer场景。PT2.6要求holder/storage alias保持；r2在所有4类都观察到copy而非双向共享，receipt `20261027-qwen-assign-storage-alias-r2-prepared/attempt-01-cpu/receipt.json`。r3的patch与known-key/meta guard在隔离tree下8个定向测试为7 pass/1 fail，失败仍是Tensor→Parameter后source/target双向写未共享；r4独立直接Parameter()/detach()复核也失败。assign/meta/guard补丁的文件内容已在main（root与保存的patch/final逐字节匹配），尚未提交，且CPU失败仍在；不得声称修复完成，也不得将CPU旧结果视为真实CUDA验证。新隔离storage-version候选引入shared token写版本和反向读保护，并需先物化懒执行旧reader；它尚未编译/运行，需审查其普通native backward、延迟构图和autograd保存边界后才可单独放行CPU。main no-hook full logits在另一个旧source上仍FAIL10,338,462/29,171,712，不能启动LoRA。

Vision native Lt稳定选择的既有fresh-process及8项existing regression通过只关闭了局部kernel选择稳定性证据；新continuous恢复stage已终止失败：435秒worker deadline，receipt状态 `TERMINAL_FAILURE_STOP`，收费438.2823419321794s，原预算余310.0956941910554s。step62–67日志与checkpoint64 SHA `887b7f64fd74007ed61cdc5b4a030cce12477814940059a671e685a829190622`保留，但没有最终model report/JSON/NPZ，因此不得作为resume输入。JIT文件新增458项，既有`.cc`/core/jit_utils未变，但cublas `3834eb7feaa26b5c`的`.so/.d/.key`改变；复核确认该对象是前一PASS8阶段新建的，因旧key/dependency原字节缺失，重建触发原因仍不可判定。终态审计SHA `58bb5f0ee72273f9ad02ef70ca08d813fbd5384abf3ffab95e2a2c0b0f886b3a`；不得重试或将当前cache身份静默视作原身份。独立对r25 CNN/ViT BF16 full/prefix/resume的离线审计SHA `eb7726ca8a2cf6a4f19d1dc247fee1c5dc2011f50016b89da67db527328fc907`验证6个NPZ绑定：CNN首个已归档gradient_norm差在step63、loss到step64才差；ViT首个gradient_norm差在step67、loss到step73才差。逐步参数/全梯度未归档，故不能排除更早内部差异或归因。原Vision五格仍FAIL，BF16训练不走native Lt候选所拒绝的BF16或requires-grad路径。

Day4–5工具审阅确认旧formal-r6完成1×8机械pair（201输入/master、197梯度、19 AdamW updates、10 hot样本和独立memory/attribution），但绑定旧HEAD/content且共享负载，仅为诊断。当前source无可用release-valid正式三轮/容量窗口。新的BERT successor只在仓外离线准备：代表格拟B8/S256、PT/JT各三轮交替、完整phase/NVML identity与JIT/allocator稳定性守卫；容量候选尚未运行或冻结，B224/S512不得自动扩展/标签为near-capacity。当前GPU负载不能作为发布性能窗口。

主树CPU core维护检查已在先前source `995338ff8f112839ede40f9e67362210f1d3252df00883008a0c3bef86420ef7`完成：native101pass/43skip/1xfail，Torch227pass/3skip，合计328pass/46skip/1xfail、rc0、880.513255596s，`20261027-main-maintenance-guard-r3/core-attempt-01/terminal-audit.json`绑定core和未漂移源码。它不验证后来46a8 source的alias修改。当前三文档相对链接与layout通过；一次直接structure入口在collection前因默认Anaconda libstdc++缺`GLIBCXX_3.4.30`而rc3，无测试结果。没有据此次失败否定已有绑定structure结果，也不因纯文档改动重复整个门禁。

本轮仅正式放行T5 CPU44例和一个CUDA bootstrap stage；所有失败均按各自manifest/receipt收费且无自动后继。repository结构/layout或CPU核心维护通过都不替代Day2–5退出条件；BERT loss阈值、T5 strict recovery、Qwen完整no-hook、Vision五格与正式Day4–5要求未变。

## 2026-10-27 19:27 +0800：Qwen 最终加载补丁通过定向回归，完整 logits 仍 FAIL

最终 assign/meta/known-key guard 源码身份为 `995338ff8f112839ede40f9e67362210f1d3252df00883008a0c3bef86420ef7`。`20261027-qwen-meta-assign-guard-r3-prepared/focused-attempt-01/` 已串行完成 CPU 7 项、独立 PyTorch2.6 CUDA 公开片段与 Jittor CUDA 7 项，全部 rc0/PASS，公开字段 exact、fallback0、source/core 未变；收据 SHA `e85e0a70bd65f6df06f3b6abba15918b1948e14613320d1f44ca8b205a2741b4`。仅替换被指定的 holder，保留 Parameter trainability、buffer/source 元数据及未命中别名；copy 到 meta 为警告/no-op，meta 不再被归为 CPU/CUDA，known-key 替换失败显式报错。此前完整 CPU 33pass/4fail/1skip 仍保留；3 个 oneDNN 环境失败的新 r7 入口缺少 matching binding 且 NO_BUILD，尚不能直接运行，独立 storage 契约失败未修复。

新原 no-hook `20261027-qwen-nohook-meta-guard-r3-prepared/nohook-attempt-01/` 的 worker 与原 comparator 均 rc0，耗时231.353749990s，实际 cuda:0、fallback0、输入 exact、所有 logits 有限；**完整 logits FAIL：10,338,462/29,171,712 超原 atol0.005/rtol0.03，max abs3.8125**。有效位置4,480,257/22,030,720 超限、max abs0.421875，padding 5,858,205/7,140,992 超限；不能删 padding 或借有效 argmax 代替原门槛。probe loss2.691995620727539。原 CUDA core `67f8688496265966e216ebf82655966e106f7dbc087b7cdfc99218625af74882` 及 source 全程未变，无训练/自动重试/LoRA。comparison SHA `3fb7d7a84183c8b54cf2e3e77066c70537301af4342313d218acb353a078fc1b`，receipt SHA `aaf24b780e3b9e618355d4bc79ed840eba3134d711bdafd3da3710df286a55fe`，NPZ SHA `fd2074eca6da9281487b85c8850123c0da57761d80b638d215e91cf8ab044f4f`。本次只关闭 meta 装载执行阻断，完整前向前置仍开放。

离线审计 `20261027-qwen-pass-main-offline-audit-r1/` 用当前原 comparator 重比较曾成功的隔离候选 NPZ，logits 仍 exact/PASS；其87个 dirty 哈希与817个 core stamp source stat 均匹配，但该证据绑定 CUDA12.2.140，与主树 CUDA12.4.131 不同。两树 softmax 已字节相同，其余 SiLU FP32 opmath、rsqrt/mean/RMSNorm 等共41项 runtime 差异未建立逐块因果，不整包覆盖主树。下一优先对照是 SiLU 单块的独立 PT/JT CPU 与真实 CUDA 值/输入梯度；仅在局部旧失败/新修复证据成立后再决定一次原 no-hook，不自动启动模型。

Vision r2 隔离 native Lt 候选已准备，保留8 heuristics、按第一个 SUCCESS 选择、去掉墙钟排名；新 fresh-process 测试仅补可移植导入入口。manifest `1c4f68c67c38867dea3cd15454a2538368d43a25ca43327fa6eb6bfe7caebf9b`，尚无真实编译/CUDA/恢复/成本证据，原五格 FAIL 与余额1289.681100597838s保持。T5 ordered-sum 五文件候选 source `930732a5be98e897ff1dfbe49b184c97aa5130b110346e72b714ad44c9bc5e92` 仅静态准备，默认0不变，实际 CPU/CUDA/provider/norm VJP 未运行；余额1491.236827937s与原771/2088不变。

本次 fresh fetch 远端仍为 `e5cffe1a`、为 main `765f1791f` 祖先，未 merge/丢弃 dirty/index；原验收/修复/Qwen树 dirty51/85/87，main99条。现场无遗留本任务worker/cache holder，根盘约余42GiB、data6约6TiB，GPU均共享负载，不作正式性能结论。主树 CPU core 维护检查已在独立 `20261027-main-maintenance-guard-r3/` 启动，源码冻结；首个外部控制脚本语法失败发生在启动前并保留，v2通过语法检查后启动，无通过结论。发布仍 **Not complete**，BERT 原15.425953905%>15%、GC公共重算与Day4–5稳定三轮/真实容量/峰值均OPEN。

## 2026-10-27 18:42 +0800：Vision 同期反例、T5 primitive 与 Qwen 加载回归

Day2–5仍未完成。Vision 同期对照复用原source/core/cache/final67/eval65与GPU，只对私有Lt候选数作8→1诊断：两个新进程各32次，1024/1024跨进程输出raw-bit exact，全部与原continuous一致。三种新增Lt CUDA源码只有cand数组大小与heuristic请求数改变，逆替换后与原cache源码exact；stage-a新增12文件、b无新增，原JIT/core/jit_utils未变。随后在同cache上补无干预candidate8两个新进程，各自32次稳定，但1024/1024跨进程对比均FAIL527/650（两批263/264、最后batch1零差），max abs `5.960464477539063e-8`；a精确复现原fresh，b精确复现原continuous，也与单候选输出exact。这把Lt每进程计时选算法与该固定eval跨进程分歧联系起来，尚未记录具体algo ID/执行分支，未替代公开修复或原五格验收。控制终审SHA `edd2fe3ab37ef82f3a8eb6285734f8a0b0a8e5010afc4080a1eb521ff9e6d919`；同期baseline终审SHA `57c2e82c0f78af77b614acc16ee45bd15f9c9284feea8fe8431ca80a918c9006`，位于未版本化 `20261027-vision-lt-output-audit-r1/` 与 `20261027-vision-lt-baseline-audit-r1/`。两组新运行合计45.266462644s，原4200s家族累计2910.318899402162s、余1289.681100597838s，无自动后继、无性能结论。

T5 `20261027-t5-primitive-baseline-r1/` 保留原终态 `BASELINE_STOP_SCHEDULE_NOT_MATCHED`，PT rc0/JT rc1，无timeout/残留组/holder，source `efe225…`、core `6dfe…` 与73绑定未变。独立终审重新核对128份数组：PT dx/dgamma各1hash、稳定输出1；JT dx32hash、dgamma8hash，值0/1/2。输入是精确实数和2的FP32取消样例，不以PT稳定1声称实数精确或新增精度PASS。原guard唯一失败是要求初始预算524288字面常量；standalone为65536/131072，独立算出的实际ownership仍同原review（dx16×256、dgamma64×256），其余条件和两provider Count32均通过。独立审阅不改旧STOP、不为改标签重跑；它建立局部sum重复不稳定，不能认定原public VJP或strict恢复的唯一根因。终审 `20261027-t5-primitive-baseline-r1-terminal-audit/TERMINAL_AUDIT.json` SHA `f28966d88ad01fa2c73a47e2fe69d16f306c1c475ff681aacfe97258471caf4c`。本次收费81.709760715s，原1800s家族累计308.763172063s、余1491.236827937s；没有core修复运行/full50/full100/resume授权或新通过。

Qwen assign/meta分类修复在 `20261027-qwen-meta-assign-cpu-r6-prepared/` 完成独立PyTorch2.6 CPU分类与Jittor逐字段对拍、定向7/7 PASS。完整serialization CPU范围为33 PASS/4 FAIL/1 SKIP：3项conv前向因NO_BUILD下缺匹配oneDNN暖库失败，1项既有storage地址契约失败；唯一内嵌CUDA方法未执行，CPU补片段也因full失败未运行，因此完整CPU仍FAIL。storage实现与本次修复前AST字节一致、失败不经过load，但没有动态baseline复跑，归因边界仅此。真实CUDA `20261027-qwen-meta-cuda-nohook-r2-prepared/cuda-assign-attempt-01/` 已独立PT→JT完成，公开cases逐字段exact、定向7/7 PASS、fallback0/source/core未变，耗时20.074705601s，core SHA `67f8688496265966e216ebf82655966e106f7dbc087b7cdfc99218625af74882`。这些证据只验证当时冻结source的公开加载/分类；为防新增assign分支忽略known-key替换失败，正在追加显式错误保护，最终source须定向复验并重新绑定no-hook cohort。原main nohook失败及所有旧收据保留，LoRA原前置仍开放。

## 2026-10-27 18:10 +0800：主树加载缺陷、Vision 原 cache 回放与 CUDA 证据边界

Day2–5 **仍未完成**。本轮 fresh fetch 的 `origin/cgq_transformers` 为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，仍是 main `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe` 的祖先，无需整合；未 stash/reset/clean/push 或整合 `origin/2.0-refactor`。原验收、修复和 Qwen 诊断树 HEAD 仍为 `9e0e187bb013e71d85111acfd668e46aeff9f9d2`，本次记录 dirty/untracked 分别 51/85/87 项；这是现场记录，不能替代后继运行前的身份核验。根盘约余 51 GiB、data6 约余 6 TiB；没有清理证据。原先已暂存的三份文档及所有未暂存改动保留，写入前的 index patch 和文档 patch 在未版本化 `20261027-root-evidence-update-1810/`。

主树 Qwen 无 hook cohort `20261027-qwen-nohook-main-current-r1/` 已 rc1，失败发生在前向之前：`.to("cuda")` 后大部分基础参数仍被报告为 meta，lifecycle guard 拒绝继续。该 cohort 没有 initial-probe、comparison 或有效完成收据，不能借用另一候选的历史 forward/LoRA 结果补齐。冻结 source fingerprint `7fc9a18616f1c5b0da52c02be89a4127bc2a579f0c1944a59001c6115b5dab02`、original probe SHA `8ba6f26c138a44092822f9d7696ddbc352b67b3daa041ee903ac0c04be18fd75` 和 manifest SHA `ddd6b28aa5fd9df34920ef458aeeb55a5a9c036a8d41482e4c820870705bb93d` 保留。独立 PyTorch 2.6 与公开最小 CPU probe 证明当前 compat 的 `load_state_dict(assign=True)` 没有替换 meta holder，且会错误地改变未命中的 tied alias 和 checkpoint requires_grad；Transformers 4.56.2 的加载路径确实使用该公开接口。修复正在主树限定文件中验证，尚未取得当前修复的完整 CPU/CUDA 与 no-hook 模型验收。minimal-r1 的 Anaconda libstdc++ 缺 `GLIBCXX_3.4.30` 失败保留；系统 libstdc++ preload 后 minimal-r2 rc0/CPU/fallback0，report SHA `b98bee0e31a9395a7e3b2a7991f9bf200074671ad8e76251e932e06da501d2b8`。修复后 minimal-r3 的 assign=True 行为已改善，但不是完整验收；CPU regression-r4 因工具链漂移到 CUDA12.2、未命中 CUDA12.4 暖 core 而在测试前停止，没有 pytest 结果，不重跑旧目录。修正环境后的 CPU regression-r5 命中原暖core `6139d0de7189d736b2bea335aada20b0e822f81d3f195aeaaaf376b814398a75`，实际7项为6 PASS/1 FAIL、0skip、CPU/fallback0、source/core未变；新增assign/meta no-op/容器/源元数据用例通过，既有meta tied用例在load前发现 `is_meta=True` 同时 `is_cpu=True`。full阶段按失败停止，不能记CPU gate通过；设备分类最小修复待新绑定复验。

Vision `20261027-vision-r2-same-cache-replay-r4/` 用原 absolute cache、core `1dc6cefcd1527f5a830aa0e6b70a0a6262e12c7547c6fefbbe8d6ee5c54a64ad`、原 CUDA12.2 与 final67/eval65，在真实 CUDA、fallback0 下 rc0/6.727510105s。同一新进程的 32 次完整 eval65 逐位一致；对原 continuous 仍有 531/650 个 raw bit pattern 不同、max abs `7.450580596923828e-8`，对原 fresh 有 500/650 不同、max abs `5.960464477539063e-8`，仅两批 batch32 不同，最后 batch1 exact。`OFFLINE_COMPARISON.json` SHA `7add60df3f22c20af5578dd975200796dd312597d8c3670b6ad6bd323fb3b6ec`、执行 receipt SHA `d9c27f769107ded69266ca1346d83d41c13d7238133113094c0ef2cbd422738a`。r2/r3 的框架导入前 harness 失败均保留；三次合计收费 6.957669657s，原 4200s 家族余 1334.947563242s。Lt 路径的体积阈值和每进程计时选算法与 batch32/batch1 边界吻合，只是可证伪线索；尚无干预结果，原五格与 strict recovery 均 FAIL。

GC combined-r2 的 saved CUDA attempt `20261027-gc-r2-cuda-positive-r1/runs/jittor/attempt-01/receipt.json` rc1/67.917668s，明确拒绝 CPU-only saved foundation 在 CUDA 上运行，不能移除守卫并声称支持。独立 direct-only cohort `20261027-gc-r2-cuda-direct-r1/` 则完成 Jittor rc0/120.2298s、独立 PyTorch 2.6 CUDA rc0/2.7538s；actual cuda:0、fallback0、no_fuse0，source/tools 未变，mapped core `b56588b45c4793dbb7405e19c6f1fad2904a2198a9f7d5bde704b1d6566b8596`。comparison SHA `e22c3b92cd4d9e476e33de09839cee0a2a3f438305d9bb99d7f9c77870004269`，状态 `PASS_NATIVE_CUDA_PAIRED_MATH_ONLY`，24/24 原阈值对照通过；只覆盖 direct 数学，**没有** saved-CUDA、owner-safety、公开 checkpoint(False)、RNG/AMP 或显存收益结论。

BERT CUDA-r4 因冻结 handoff/manifest 不符在 attempt 前拒绝；r5 在 PT 计算后的身份守卫失败，错误是把含 used_memory 的整条 NVML 记录作为身份比较，未启动 JT/JIT。r5 没有保存两次 raw mapping，不能补造显存变化量。新 r6 仅将比较限定到 PID+UUID，保留每次恰一条/固定 UUID 守卫及两次显存观测；12 项纯 AST/mock 正反例和 11/11 工具哈希通过，source 仍 `779b35d0b5aa34fefe6b2c73f3b51998cd01b887036bae85d19d924ea89e2c34`。r6 已终态 PT/JT rc0、真实 cuda:0/fallback0、无timeout/残留组，mapped core `70385e793efda8c8c6588938bd2da611b77066233a48bb89c877559536f3041b` 从import到exit exact。原阈值下 output/loss均0项失败，input_grad 61440、weight_grad 589824、bias_grad 768项全部与PT逐位相同；retained contraction cotangent与公开output.grad exact。oracle report SHA `6000092fd2e1ff8df8d1949cc6eb37a4de7ff3700a880c290713704637852c37`、completion SHA `d132e991b53e498be8bb29f93cb3806fb0b7f2a57b7781d699c2adf5c180e5dc`。因此CPU9150/61440缺口没有在本CUDA最小案例重现，不能用它解释原CUDA任务失败或猜测性修CUDA；该案例使用synthetic activation/upstream、没有optimizer update。原双 seed task gap `15.425953905% > 15%` 与 Day4–5 正式代表/容量格、三轮、稳定窗口及峰值门槛不变。

### T5 原 provider 线程分工解释的纠正

`20261027-t5-reduction-ownership-review-r1/` 对原 frozen manifest 54/54、runtime artifact 16/16 与四组生成源码/binary/key 重新做纯标准库绑定审阅。原 review 的“两 warp”和“三 block”描述错误，旧件保留并由本段覆盖解释：dx 的静态布局是每行128个归约线程，完整active-mask warp分支产生四个无序部分和；非完整mask可能走逐lane atomic，实际分支没有launch trace。dgamma 的warp线程对应不同channel，无法warp合并，每channel有32个block的32个线程部分和。FP32取消反例证明这些布局存在顺序敏感机会，不证明原GPU的实际调度或首因。`OWNERSHIP_REVIEW.json` SHA `555ee1192fed11210b5b5f68314c8726ec0e8702ace11c18803fe1ffc2b454a8`，`REVIEW.md` SHA `31bd4c543f046376c38dd21e43cc350a75b44a16dbaf8a886150ed9736dee2bd`。该离线审阅收费0，原1800s家族仍累计227.053411348s、余1572.946588652s；下一项是两固定布局的有限primitive baseline，而非猜测性全局para4或full100。

## 2026-10-27 00:18 +0800：GC combined-r2 CPU paired math 对拍通过

GC combined-r2 的独立 comparator 已消费冻结的 positive/oracle 输出，并重新核验 source、tool manifest、bootstrap identity 与 63+22+1 项 CPU receipts。结果 `PASS_NATIVE_CPU_PAIRED_MATH_ONLY`：4 个场景（`nonlinear_fp32`、`nonlinear_fp64`、`matrix_fp32`、`view_fp32`）各自的 saved/direct 路径，分别对照 PyTorch 与解析梯度的 `loss/gx/gw`，共 48 项；FP32 阈值为 `atol=2e-7, rtol=2e-6`，FP64 阈值为 `atol=2e-12, rtol=2e-11`。输出绑定 source `b6ca987cddb259e67756cdcf540d2a70e891e0be1bf2f988684ad34497389889` 与 CPU core `555086064eba86660e90dbfaa83de4affb2bfdb6566d441b85cb0a59bad2a401`。

原 comparator 输出与 root-review 副本逐字节一致，SHA256 均为 `0d182a8baeb2f3d9651d07d089a4d7894227cb134809659d238198bf35afba1d`；两份文件分别位于未版本化的 `20261027-gc-generic-saved-combined-r2-comparison/positive-comparison.json` 与 `20261027-gc-generic-saved-combined-r2-root-review-r1/positive-comparison.json`。比较器明确记录 `release_acceptance=false`、`public_false_wired=false`、`allocator_peak_or_net_memory_claim=false`。因此此结果只关闭 paired native CPU private-math 比较，不能推出 owner-safety、public `checkpoint(False)`、真实 CUDA、RNG/AMP、峰值显存或模型门禁通过。

Day2–5 仍未完成；原 Qwen、T5 strict recovery、Vision、BERT 15% task-loss 与 Day4–5 稳定三轮/真实容量边界/峰值门槛保持原状态。尚无新的真实 CUDA GC 结果；后续若进入该阶段，必须使用冻结 source 的独立 cache 与新输出目录，并重新核验运行窗口，不能把 CPU paired math 结果当作设备验收。

## 2026-10-27 00:08 +0800：GC combined-r2 CPU stages 完成，独立 PyTorch 对拍待审

Day2–5 **仍未完成**。GC combined-r2 内容 SHA `b6ca987cddb259e67756cdcf540d2a70e891e0be1bf2f988684ad34497389889`，tool manifest SHA `d7a7b196d483ff6f09ce6e66e485d8e2ad2b3631cf0ef2ae8728f2fc6fa8a5bf`，bootstrap 编译出的 CPU core SHA `555086064eba86660e90dbfaa83de4affb2bfdb6566d441b85cb0a59bad2a401`。全部 stage 均绑定同一 source/tool/core；actual device 为 CPU、`fallback_count=0`、`no_fuse=0`，每项收据 `accepted=true`，但统一保留 `owner_safety_accepted=false` 与 `release_acceptance=false`。

- Bootstrap：rc0，41.395883s；receipt SHA256 `035086da9e1eba9032fdc038a47bda2947f23312d7b06866ffa84be705345aa9`。
- 原 63 项 CPU 回归：63 collected、0 skip、0 failure，rc0，98.275326s；receipt SHA256 `f81baaf17a4e4b75f62dd918aea823c85f05afab75dde429e2475b4c1d47f53f`，JUnit SHA256 `cf1ca3d3261a002f14e9a5aaa8dbaaba77d1d25abe21c9fbb14171747a1e62e9`。
- priorview 回归：22 collected、0 skip、0 failure，rc0，3.607991s；receipt SHA256 `dd01ea9f6790ae444015933d16ed91415fc1816358376e2b98101002c3fbaf49`，JUnit SHA256 `deddddcf246653067ea1cb335e72fb1f4290c232b80e9ece19c3cc273bad6073`。
- swap 回归：1 collected、0 skip、0 failure，rc0，1.567859s；receipt SHA256 `0cad901382547b5d2a4e57902cdeb68a16f8c688bcb7c2086391a1145b60dd56`，JUnit SHA256 `712e513e4d9ebf719286af570ab2bb45d884b781de1008a7bce75413749342b9`。
- Positive diagnostic：`nonlinear_fp32`、`nonlinear_fp64`、`matrix_fp32`、`view_fp32` 四个 saved/direct CPU 数学场景，其 loss 和两组梯度均通过 worker 内既有容差断言；rc0，4.658710s。receipt SHA256 `ae181d0d46270332109c663f3c7b34adc5b0de70d69ef76f4f80f8ea314ed9e4`，`positive.json` SHA256 `e2d9aa4ac57e32fba097357a837f7d49def2111203e09e239cb895ee91011ff3`。
- 独立 PyTorch oracle：PyTorch `2.6.0+cu124` CPU 进程生成同四场景的 loss、input gradient 与 parameter gradient；rc0，1.820663s。receipt SHA256 `28175cba097a8fe301d8f83be3338f692f048e39e47e822fcdef935f7d9dedbb`，`oracle.json` SHA256 `f5ef5a01d3cb5b9af301e1d5ce38a95c1ac736247ec6a5cb3372f640180d7e90`。独立比较器尚未执行；root 将在独立输出目录按原阈值对拍，未取得比较收据前不宣称 paired math 通过。

这些结果只覆盖冻结候选的 native CPU 基础和 private math 层，不验证 public `checkpoint(False)`、Python/native owner 释放全路径、RNG、AMP、真实 CUDA 数值、峰值显存或模型行为；没有改变任何 Day2–5 原失败与阈值。所有运行组和 JIT cache holder 已复核为空。下一步为 root 离线完成独立 PyTorch paired comparison；其后如通过，由 root 单独安排 real-CUDA 数值回归窗口。原主树源码未由本轮修改，94 项 dirty/untracked、历史失败证据、HEAD `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe` 保留；fresh remote `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81` 为该 HEAD 祖先，未整合 `origin/2.0-refactor`。

## 2026-10-26 23:20 +0800：structure终态、T5公开反向复现、BERT r7与GC combined r1失败边界

Day2–5 **仍未完成**；本节新增的结构门禁、局部数值诊断和GC CPU回归均不替代原模型/发布门槛。主树仍为 `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`，原 dirty/untracked 与空 index 保留；本轮重新 fetch 的 `origin/cgq_transformers` 为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，是所核验六树 HEAD 的祖先，无需整合。现场快照在未版本化 `20261026-root-resume-134637/ROOT_REACQUIRE_1513.json`。

主树提交前 Torch-mode structure session 82548 以 rc0 完成，**1396 passed、4 skipped**，pytest报告700.57s，1400 collected/1398 executed。两项 skip 为当前解释器未安装 Jittor、两项为未安装 pytest-xdist；日志明确这些文件没有由本次session证明。进程、独立 cache FD holder 与记录的锁均已清空。它只证明该结构门禁；CUDA toolkit 12.4.99 与既有 release cache/toolchain 不同，也不构成发布验收。终端收据保存在未版本化 `20261026-main-structure-2250/ROOT_TERMINAL_RECEIPT.json`。

T5 public norm 诊断在冻结候选 `source efe225153ddd2bb404f3dbef174fe22efe60319dd8057c4e5ae4a548b297765b`、CPU source core `6dfe0645db0cdd9d4b30d38165a32a580896b19064557cb73056a96b62e24645`、真实 CUDA `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5` 上完成唯一32-repeat attempt。结果 `PUBLIC_NORM_REPEAT_DIFFERS_DIAGNOSTIC_ONLY`，rc1/46.376770237926394s，真实 `cuda:0`、fallback0、无timeout、组/cache holder清空。固定的 public T5LayerNorm forward `y` 为1个hash、全exact；`dx` 为32个hash，每次相对repeat0有235–561/27648项不同、并集1285、最大差 `4.656612873077393e-10`；`dgamma` 为32个hash，每次520–569/768项不同、并集761、repeat0最大差 `4.5299530029296875e-6`，任意两次最大 `5.7220458984375e-6`。独立cache静态审阅把两条sum绑定到生成CUDA里的 `atomicAdd`：dx末维REDUCE=4两warp写同row，dgamma前两维REDUCE=3多个block写同channel；没有独立profiler launch trace或干预，故仅是强机制吻合。该输入使用明确synthetic dy，原内部cotangent未归档，**不能**定位原模型首个因果算子。终审 `RUNTIME_REVIEW.json` SHA `36f7a447a22ddc2dad1e3c8349408078e2c550536906f49bc2bafe4deb204b09`；provider/reduction离线审阅 SHA `82f6dace43ee6ed573ce2ce51029ae1ab6be366fc7f12bdf3297227c37b24c97`。原1800s家族累计227.05341134802438s、剩余1572.9465886519756s；未重置旧账本。原strict `FAIL771/2088` 不变。

BERT r7 保留mapped-core守卫后完成一次900s上限CPU JT control，source `779b35d0b5aa34fefe6b2c73f3b51998cd01b887036bae85d19d924ea89e2c34`，实际core `7fee20dd13762c46a3a70adbf76317fb2a0b39bad0d6c736784842b99f1e0ef7`，两次BF16 forward、一次backward、11 arrays、0 update/fallback，source/core/exits exact。原PT对比 `FAIL_LIMITED_LINEAR_FORWARD_GRAD`：input grad **9150/61440**项超原逐项 `atol=0.01, rtol=0.05`，max abs `0.375`；两个forward/loss、weight grad和bias grad在各自原限值内。只读raw-array分析显示差异覆盖80个token行：绝对差median `0.01709`、p95 `0.078125`、RMSE `0.03755`、269项符号不同。NumPy float64 dot中，先将upstream和weight各自BF16舍入、点积后将结果BF16舍入，与PT有61427/61440项exact且全符合原阈值，JT仍有9150项超限。证据支持把缺口收窄到 BF16 Linear input-gradient contraction/reduction路径，但不能区分operand/result cast位置与累加顺序，也不代表真实后端复验。将准备一个有界隔离CPU中间量诊断；未获得新终审与root窗口前不运行。BERT双seed task gap仍为 **15.425953905% > 15%**。comparison SHA `f34e155549ff724a6517fcc6c61f6137f260b01f0ed4dde3ed81c4fcf44f5a81`。

GC combined-r1 离线绑定 source `05512469a799643a05f5d8bf09515d9fac4578e959c30e53b8de348d6a83635e`、工具 manifest `01615801526ff6ba388f2b56a2a131a14b1b3e6ac27305f99af6e19a69f25527`、artifact manifest `1a77d672d5363e190226049c4a7a58a9fe3656cbc956a18ae46eb8fde56d02a9`，11/11最终只读审阅通过。单独bootstrap rc0/41.302s，实际CPU core `27b820b0646a4f0294cb2d912010649eabf65f69990b86e008d9088c04905999`、fallback0/no_fuse0；原63项测试 rc0、**63 pass/0 fail/0 skip**、98.649s，与bootstrap身份完全相同。新增prior-DataView的22项中 **21 pass/1 fail/0 skip**，rc1/3.686s，source与工具未变、组清空。失败仅 `test_last_view_release_allows_swap_and_real_gradient`：合法swap移动saved输出后，replay报 logical value 178 的saved version改变。源码路径复核成立：`VarHolder::swap` 对两侧调用 `remat_check_mutation_cpu_s1`，该函数无条件调用 `native_saved_note_mutation`，使被移动Var版本递增；这是holder交换的伪mutation。r1历史收据保留，未重跑；positive/PT和owner-safety验收未放行。combined-r2当前仅离线准备修复，不得把r1的63通过写成combined全通过。

三份文档更新前，production源码未改。原结构/布局门禁结果、T5/BERT历史FAIL、Qwen与Vision门槛、Day4–5无共享负载有效配对窗口的边界保持。

## 2026-10-26 22:45 +0800：weak-frame CPU 回归通过，T5 public backward 已冻结离线计划

GC weak-frame 候选 `torch-compat-gc-weak-frame-20261026-r1` 只把 native saved slot 对 frame 的 shared ownership 改为 weak ownership，并在操作时 lock；Python GC-visible owner 仍强持 frame，活动 unpack/replay 期间用局部 strong reference，过期 eviction 不释放 storage，过期 mutation/unpack/stats 显式拒绝。source `651e1b69a497e7e48df8e84ba76df9d2fcf7439749a0734ae7cd090910af83ec`，工具 SHA `cd69ca8785a7c84c076af987854ebf6adb34a22115586307e694d347ef1e87e8`，离线语法审计覆盖18个 C++ TU 与生成 binding。唯一 CPU core `6adb8518a7438c38734ab1da42d6c99afbc985daf706e99b7c419fbaee3210ea`。import-only bootstrap 41.467947857s；未改的原63项测试 **63 PASS/0 FAIL/0 SKIP**，103.285086088s，原 lazy-lease frame 计数失败已通过。公开 False、原 strong Python owner、lease storage 和原测试文件均未改。

同一冻结 source/core 下，4个正向数学场景和真实 CPU/fallback0 通过；独立 PyTorch 2.6.0+cu124 CPU oracle 成功，离线 comparator 对direct/saved、PyTorch/analytic四场景完成48项原阈值比较，状态 `PASS_NATIVE_CPU_PAIRED_MATH_ONLY`，比较 SHA `405528e3e1456ff87e65b910255a1844b7fea26d4da33d1e82fff0908fcf6c83`。完整runtime终审 SHA `8831db7fd562fc5fbcd81c2cd42a03d8844d6ee4d8994acc9b3e43a4c9f74bba`；原63测试终审 SHA `b89ab471d68ab44a51b00d466b3c0c75fb7997495d74144943c36ac1709eba60`。全部source/core身份一致、进程组、cache FD holder 与proc锁空。日志保留原维护测试 lived_ops/vars 诊断增量，因此不把63项通过扩大成所有native cache/storage均释放，也不声称内存收益。

另一个 prior-DataView adoption 隐藏owner环仍未解决。本次没有混入generic saved r4；该r4已有22项DataView反例和guard但没有运行。weak-only r1 与 r4 source delta可组合，但必须创建新绑定的新候选，再运行bootstrap与原63+新增22项共85条CPU测试；运行计划 SHA `58b9fbece7517cb4ca440d02031f8609450f9e618bd453d9d6b6a9e049371974`。公开checkpoint(False)、RNG/autocast/CUDA和峰值门禁仍OPEN。

T5 fresh-repeat-r2原失败、budget及输入边界见本结果22:31节。新离线队列 `20261026-t5-public-norm-backward-r1/` 选取原404事件 `encoder.block.0` 输出event_0015作为 `(2,18,768)` FP32 norm输入，从checkpoint逐字节提取gamma及原`epsilon=1e-6`，并固定独立synthetic upstream `dy_i=(((17i) mod 61)-30)/32`。原内部cotangent未归档，此测试只经公开T5LayerNorm与公开autograd.grad验证该固定数学问题的32次重复raw-byte确定性，并分别报告FP64解析误差；无新precision gate，不证明原模型首个backward因果。24项离线检查及controller verify-only通过，无framework import/claim/runtime/accounting。manifest SHA `ef224a883e72a164714026caf4bea48919ea9be00c0d77ce68b4af88cc972ece`，worker SHA `e86059550fadcb33f56f2a280e2932fce6fb6e5c96129809c256c67d3554fbfa`，controller SHA `4b0b24cc871886b96660bcdeed3c3a47269acaf5a902400ffbd0c64c40335e9c`。拟议一次120s含30s清理，延用原1800s家族累计180.676641110098s/余1619.323358889902s及receipt `492b2ae94486708d02d207c7bc8f3d63b9603eef9c45a8ed41fb05fb18f1e321`；仍须root另行释放GPU/JIT窗口，未运行。

BERT BF16 r6身份失败仍保留：14:27:47 UTC worker rc1，编译后actual mapped-core守卫在数值前拒绝磁盘core变化（43f1…→7fee20dd…），jit_utils字节未变，无forward/backward/数组。16/16终审 SHA `7d7ea372ef73d1b2b77a5e8e2f4e8238602a7f0bf3152197d92c0715afb54e84`。旧expected来自r4失败后磁盘seed，不是成功primitive runtime anchor。离线ELF/source-stat审计正在核实路径重定位差异并构造保留mapped-core守卫的新cohort；此前两失败与原22 PT oracle不改。BERT双seed任务差15.425953905%仍高于原15%门槛，仍FAIL。

## 2026-10-26 22:36 +0800：weak-frame 原 63 项真实 CPU 回归全通过

GC `20261026-gc-weak-frame-r1/` 独立 bootstrap rc0/41.467947857s，实际 core `6adb8518a7438c38734ab1da42d6c99afbc985daf706e99b7c419fbaee3210ea`。随后原63 tests的唯一attempt rc0/103.285086088s，**63 passed、0 failed、0 skipped**，无timeout、组空、原worker已不存在；JUnit由root独立逐项核验，actual identity与bootstrap逐字段exact、core磁盘hash一致，source651e1b69/tools退出守卫通过。原lazy-leases最终frame计数断言现已通过，原两次backward/延迟lease/无backward GC/失败replay回归均保留。tests receipt SHA256 `928770c9193718cbaa1a35169aed1a48f0e96f9c1503588ce87ac97983dd9fc5`，bootstrap终审 `b31b88925b1ae5af572e1dbcb76af386b80a36d49b2ac0a19b2c119d39b2f2d5`。真实CPU/has_cuda=false/no_fuse=0/fallback0不变。

本候选还未混入priorview guard，owner adoption缺口仍OPEN；原测试日志中的lived_ops/vars诊断增量不隐藏，不据63PASS推定所有native cache/storage释放或显存收益。root仅释放后继positive180s、独立PT60s及双成功后离线比较，失败即停，不自动重试。public False/RNG/AMP/CUDA/模型峰值与Day2–5仍未通过。

## 2026-10-26 22:31 +0800：T5 反向重复差异确认，BERT r6 身份守卫失败保留

Day2–5 **仍未完成**。重新 fetch 的目标仍为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，是 main、两验收树、Qwen、GC r3/r4/weak-only 与 BERT 候选八树 HEAD 的祖先，无需整合。main 仍 `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`，94 个 dirty/untracked 路径及空 index 保留；未整合 `origin/2.0-refactor`，未 stash/reset/clean/push。新现场保存在未版本化 `20261026-root-resume-134637/ROOT_REACQUIRE_1429.json`，根盘约余71 GiB、data6约余6 TiB，无清理。

T5 `20261026-t5-fresh-repeat-r2/` 的唯一 180s fresh repeat 已结束，controller/原 worker rc1 数值失败，postprocessor rc0，状态 **DIAGNOSTIC_NUMERIC_FAILURE_RETAINED / SELF_REPEAT_DIFFERS**。新 fresh 对原 continuous 有 **63 个梯度数组、34,017 个元素不同**，max abs `5.960464477539063e-08`；encoder norm25、decoder norm37、shared1，shared 仅一元素。新 fresh 对旧 singlemicro baseline 有 **77 个梯度数组、1,385,954 个元素不同**，max abs `7.722713053226471e-05`。两 fresh 的404完整动态事件及数组、5输入、257参数签名、2,536 RNG原始叶与4边界摘要、Trainer/RNG context均 exact；raw/returned loss、logits exact。新 fresh 对原 continuous 的404前向事件也 exact。实际同物理GPU/cuda:0/BF16/fallback0、level3/compile7413，独立NVML见2324 MiB；仅一次training_step/forward，零update、零norm hook/retain。source `efe22515`、core `6dfe0645`、工具/原worker/adapter/旧收据退出重验通过，所有controller/worker/postprocessor组与cache holder均空。

T5本次计费 **78.605056156s**，原1800s家族累计 **180.676641110s、余1619.323358890s**，未重置预算。只读终审 `RUNTIME_REVIEW.json` SHA256 `c7d526406fc797e357d56650fbe6c8bdb21d9e048385dec9eeea81d3a63e23dd`，receipt `492b2ae94486708d02d207c7bc8f3d63b9603eef9c45a8ed41fb05fb18f1e321`，repeat comparison `1d025b9be9a8435ce3571ba2894637cfdaf0d426846bbdcb44a7d24a6e82b68d`。下一步只离线准备固定归档norm输入与显式synthetic upstream的公共反向重复诊断；内部cotangent未归档，叶梯度分布不能确定原轨迹首个因果算子。原capture仍未放行，原771/2088 strict FAIL及旧baseline失败保持，不重训50步/full100。

BERT `20261026-bf16-linear-reentry-r6/` 唯一CPU JT control于 **14:27:47 UTC rc1**，无timeout、组空、cache及controller锁释放。同一r5绝对cache完成228 TU与backend重编译后，原worker的actual-mapped-core守卫拒绝新core：磁盘SHA从 `43f1e01e…` 变为 `7fee20dd13762c46a3a70adbf76317fb2a0b39bad0d6c736784842b99f1e0ef7`，jit_utils字节不变。仅started.json，尚无sidecar/runtime identity/数组、前向或backward；不能算数值失败或通过。错误文本“successful primitive core anchor”为通用工具文案；旧43f1实际是r4失败后磁盘cache seed，**不是成功JT primitive的已加载身份**。source779b、r5/r6工具、原22份CPU PT oracle、原r4cache均exact；原r4/r5/r6失败保留，无自动重试/比较器/后继。独立失败终审16/16，SHA256 `7d7ea372ef73d1b2b77a5e8e2f4e8238602a7f0bf3152197d92c0715afb54e84`，completion `62acd20c2a3453f7da395002e59bd1da1479d25a5cab2a09322445aa271dfc77`。后续先离线审构建依赖/配置，才可提出保留actual mapped-core守卫的新cohort绑定；原BERT任务gap15.425953905%>15% FAIL保持。

GC weak-only候选已实际实现并独立代码审阅：`torch-compat-gc-weak-frame-20261026-r1` 从stride-only r3派生，source `651e1b69a497e7e48df8e84ba76df9d2fcf7439749a0734ae7cd090910af83ec`，唯一源码delta为slot.frame强引用改weak；GC-visible Python owner仍强持Frame，unpack/replay活动区间持局部strong，过期evict不释放storage，mutation/unpack/stats显式拒绝。原63 tests、strong owner、lease storage和public False不变，未混入priorview guard；18 TU/生成binding仅语法通过，等待独占CPU bootstrap和原63真实回归。另r4 prior-DataView guard已离线冻结，实际新增 **22** CPU反例（下文21为较早计数），source `b9f2ea67`，尚无runtime。两个生命周期缺口分别处理，不以任一局部通过宣称owner安全。main生产源码未改，Vision五格/769字段FAIL1、正式性能三轮/代表/真实容量/稳定窗口及公开GC/RNG/AMP/CUDA/峰值仍OPEN。

## 2026-10-26 22:15 +0800：stride 修复实测 62/63，两个生命周期缺口保留

GC stride-only 隔离 r3 已在真实 CPU、默认 fusion、fallback0 完成 bootstrap 和原 63 项回归，source `2c208246843a1add0a71e1252bdcdc1382e21e9438fce22eb71668e65c98c7cd`、实际 core `dae33a763b4c0014d54231198ecab2da1b2e4557900b150012ece0b868b93ee0`。bootstrap rc0/40.934714455s；tests **62 passed、1 failed、0 skipped**，rc1/93.221210401s、无 timeout、组空、source/tools 守卫通过。原 20 个 stride 异常及 4 个 live 基线下降断言本轮均未再出现；唯一剩余 `test_lazy_leases_survive_dropping_original_graph_and_callable` 在数值及 lease 检查通过后，最终 frame/owner/lease 为 `(3,0,0)`、基线 `(2,0,0)`。不能据此放行 positive/PT 后继或 owner 安全。权威 `20261026-gc-generic-saved-r3/runs/tests/attempt-01/receipt.json` SHA256 `3b940abb507048ed5c6e533512b0c64c9732853665e5443582d4f2866d73238f`，原 63 tests 未改。

全新进程原样单跑该测试，已复现 frame 从 `(0,0,0)` 到 `(1,0,0)`；只在原测试外包 `no_fuse=1` 的独立诊断仍同样失败，故不能归因于 fusion 或 VarRelay 独有路径，也不能用关闭 fusion 制造通过。原始隔离 worker 已完成，外层事后导入 helper 曾遇 `ModuleNotFoundError: bindings`，原件和错误保留，另做标准库终审确认真实 worker rc1、组空、source/core exact。证据为 `20261026-gc-r3-lazy-lease-diagnostic-r1/{result.json,TERMINAL_AUDIT.json}` 与 `20261026-gc-r3-lazy-lease-fusion-control-r1/{receipt.json,result.json}`。Frame slot 强引用弱化仅为已审阅设计方向，尚未实现或验收。

另一个 owner 缺口已有有效独立复现：priorview v3 先 `.numpy()` 物化再看实际 CPU location，direct control 完整通过；saved case 丢弃普通 roots 后 callable/ndarray 均仍活、native counts `(1,1,0)`，只清除 `prior_view` 后两弱引用释放且 counts 回 `(0,0,0)`。v3 rc0/1.516524194s、fallback0、source/core/tools exact、组空；连同两次旧工具失败累计 **3.719155585/120s**。权威 `20261026-gc-r2-priorview-diagnostic-v3/runs/attempt-01/receipt.json` SHA256 `6c2268827774e8d360d5c29ecdca491d6e47a79bec80c22d5ab96cb624105aa8`，root 独立终审在 `20261026-root-resume-134637/GC_PRIORVIEW_V3_TERMINAL_AUDIT.json`。观察边界仅为 callable/ndarray weakrefs 与 native counts，没有 native Var weakref 证明。新 r4 正离线实现“有旧 DataView 时在所有受影响 holder 重绑前拒绝采用 frame”的修复及 21 个反例；未 runtime，原 63 tests、main、原 r2/r3 与剩余 frame 计数缺口均保留。

BERT r4 CPU JT control 于 13:50:38 UTC rc1，尚未模型构建，原因是工具读取合法缺席的 public `torch.__file__`。r5 修订仅允许 Jittor public namespace 缺文件元数据、原生 PT 仍强制合法文件路径，复用原 r4 22 份已完成 CPU oracle 原件，并将完整 cache 复制到独立新路径。r5 唯一 control 又于 14:10:32 UTC **rc3**，导入阶段要求 fresh process 重载 `jit_utils`，仍无前向、backward、11 数组、primitive anchor 或数值对比。source779b、a307 工具/104外部绑定、原 oracle 守卫通过，组空、锁空；缓存只三项依赖记录/锁变化，core 与 jit_utils 二进制字节不变。r4/r5 失败终审分别在各自 `CPU_JT_FAILURE_TERMINAL_AUDIT.json`；没有按日志自动重试。r6 仅准备同一 r5 绝对 cache、未改 r5 worker、独立输出与原 oracle 的一次有界重入，未执行。原 BERT 任务 gap15.425953905%>15% FAIL 不变。

T5 fresh-repeat-r1 在模型构建前因 manifest 缺原 worker 所需 `original_trace_observer_sha256` 退出，worker/postprocessor 均 rc1，无 capture、无 SELF_REPEAT 数值结果。其 44.709051474s 加原 57.362533480s，累计 **102.071584954/1800s，余1697.928415046s**，旧失败未改写。新 fresh-repeat-r2 完整继承原 26 字段及其值/类型，AST 枚举与删除/改值反例通过，原 worker/adapter 字节不变，manifest `85932450a1a2fae22ad0195b1050f96b805eaef0588c604dd1f8a6ba36e7e83b`；root 独立核验后仅释放一次 180s（含30s清理）的 fresh repeat，目前须查该 cohort 的实际 receipt，不按本段推断完成。仍不放行原 capture，不重训50步或 full100。main 生产源码未改，Vision 原 FAIL/余量及正式性能三轮/代表/真实容量/稳定窗口、公开 GC/RNG/AMP/CUDA/峰值均未闭环。

## 2026-10-26 21:46 +0800：GC CPU 回归失败分类与独立 stride 修复

Day2–5 **仍未完成**。本轮 fresh fetch 目标仍为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，是八个已核验工作树 HEAD 的祖先，无需整合。main 保持 `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`，94 项 dirty/untracked 包含原 91 项及本任务三份文档；index 为空。原验收/fix/Qwen 树分别为 51/85/87 项，保留所有工作。现场、进程启动身份、GPU 与磁盘快照位于未版本化 `$JITTOR_LAB_ROOT/_state/release-day23/20261026-root-resume-134637/`；未 stash/reset/clean/push、未整合 2.0-refactor。

GC generic r2 的 63 项 CPU 回归已于原 `runs/tests/attempt-01/` 结束：**39 passed、24 failed、0 skipped**，rc1、88.349513978s、无 timeout、组空，source/tool 守卫通过。原 supervisor 在 rc1 时未解析 tests 字段，故原 receipt 的 `tests=null` 保留；数量直接取原 JUnit 与 `JITTOR-SESSION-COMPLETE collected=63 executed=63`。20 项为同一 `Storage stride rank must match shape` 异常：`replay()` 无条件将合法的空 contiguous stride sentinel 传给要求显式 rank 的 setter。其余 4 项为 GC live 基线从 `(1,1,0)` 降到 `(0,0,0)` 的断言，尚不能仅凭整轮结果断定是级联或证明 owner 安全。新隔离 r3 只在非空显式 stride 时调用 setter，原 63 tests 与其余 2,807 源文件不变，source `2c208246843a1add0a71e1252bdcdc1382e21e9438fce22eb71668e65c98c7cd`；目前仅离线冻结，未编译、未运行，未改 main 或原 r2。

原 priorview attempt 在对 native Var 取 weakref 时失败，尚未进入 saved case。修正后的独立 v2 只 weakref 真 Python callable 与 ndarray，仍在 direct control 的 `loss.location()` 断言失败：断言早于 `.numpy()` 物化，原生 location 在无 storage 时返回 `none`；该次也未进入 saved case。v2 rc1、无 timeout、组空、source/core/tools exact，计费 1.153659915s，加原 1.048971476s 共 **2.202631391s/120s**。两份失败及原工具均保留，没有形成 owner 环运行证据。下一修订只调整物化与 device 观察顺序；已知先 view 后 adopt 的 owner 缺口仍 OPEN。

BERT r4 独立原生 PT CPU oracle 已 rc0，32/32 终审通过：两次 BF16 forward、一次 backward、零 update，11 数组及 FP32 input/weight/bias 梯度齐全、原 master exact。权威 `20261026-bf16-linear-fix-r4/CPU_PT_TERMINAL_AUDIT.json`，SHA256 `654f787a6acceadf626f5b1a8cb0035226ddceb5509bcceabcc6fa55bf3beb2b`。唯一首编窗口现给一次 Jittor CPU control，worker `3565828/start163146791`，独立 correctness cache、原 900s 上限，无自动 public/native/CUDA 后继；运行终态必须再查 receipt。尚无 JT 数值对比或 BERT 任务修复结论。T5 baseline 77 个梯度不同及其 1800s 家族计费、Vision 原 FAIL 和余量、AdamW 局部 launch 改善、正式性能与公开 GC/RNG/AMP/CUDA/峰值的未闭环状态均保持。

## 2026-10-26 21:28 +0800：优化器标量对照通过，T5 单 microbatch 基线失败

Day2–5 **仍未完成**。本次 fresh fetch `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，是 main、原验收、fix、Qwen、T5 r2 及新 optimizer 候选 HEAD 的祖先，无需整合。main HEAD 仍 `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`，91 项原 dirty/untracked 与本任务三份文档增量保留、index 为空；未 stash/reset/clean/push 或整合 2.0-refactor。本次现场与调度入口位于未版本化 `$JITTOR_LAB_ROOT/_state/release-day23/20261026-root-checkpoint-resume-1310/`。main 生产源码未改，最近 1398 passed/2 accelerator skips 的结构门禁继续保留原执行身份，不为这次文档增量重复全套。

新 AdamW shared-scalar 隔离候选绑定完整 source `bb1a5fa297a4b76a5f4ca58bfe110a93a5edd3381d8bbc80cd9bb5fb31e82b9b`，原基线 `062c8e11f04fe4ded8e3710e7bf7ac8f137d92a1cbb470ddcb501f95f667b78e` 保持不变。实际 CUDA 五步的 63 个参数/梯度/moment 数组与原 Jittor 基线逐字节 exact，并全部满足独立 PT 原 fixture-exact/其余 atol=rtol=1e-5 门槛；三参数均真实更新。相同三步 optimizer-only profile 的 4B scalar kernel 从 **72 降到 30**，关联 device events 同为 72/30。Torch TestAdam **19 passed/0 skip**（166.77s），native test_optim_core **34 passed/0 skip**（564.72s），两阶段 rc0、fallback0、source/core 守卫通过且组空。权威 `20261026-adamw-shared-scalars-r1/{CANDIDATE_COMPARISON.json,ROOT_FOCUSED_TERMINAL_AUDIT.json}`，比较 SHA256 `1e277d56f4e5bcfceec9acbc17fd973559f448af216f2eef6e0c4bf12d2b72dd`。独立只读复核再次核对三方数组、trace 及缓存只活一 group/step 的语义；这是局部数值和 launch 修复证据，尚未合入 main，也没有模型热态 A/B 或峰值收益。r6 共享小格与正式三轮/代表/容量/稳定窗口门槛保持。

T5 singlemicro r1 未运行且缺少原 404 hooks/张量生命周期，原件保留；独立 r2 按 AST 复用原 wrapper，17 项离线检查和 root verify-only 通过，最终 manifest `1fc8baf03485afdc84303252238919483f83fc9b69613f696665daf4872a5f2a`。r2 baseline 于 21:24:31 启动、**21:25:28 rc1 数值失败**：原 Trainer/GA2、checkpoint50 的 257 参数、5 个 inputs、公开四 owner RNG 恢复回读及原 404 动态前向事件均 exact，恰好一个 training_step/forward/backward，零 optimizer update、没有新 norm hook/retain_grad，但原 260 字段中 **77 个梯度数组不同**。原 source `efe22515`、core `6dfe0645` 和工具退出守卫通过，组空、无 timeout；权威 `20261026-t5-singlemicro-replay-r2/runs/baseline/{receipt.json,report.json,vs-original.json,trace-vs-original.json}`。同一 1800s 分配计费 **57.362533479929s**，余 **1742.637466520071s**，没有重置预算、修改参考或放行 capture。该失败不能归因于静态 capture 后 mode，也尚未定位具体反向算子；旧双 seed strict FAIL771/2088 保持，未重训 50 步或 full100。独立离线三方梯度/RNG 审计进行中。

BERT public Linear r2 的实际 CUDA 控制已记录同 AMP64 下 no_grad 为 FP32、grad-enabled 为 BF16；原 BERT 失败只记录 non-BF16，不能追溯补造其 dtype。三文件修复候选 `779b35d0b5aa34fefe6b2c73f3b51998cd01b887036bae85d19d924ea89e2c34` 与 r3 工具已离线冻结，r4 正准备按 device 共享且顺序使用的独立 correctness cache，避免三个 correctness cell 重复冷 core；原 forward/全梯度预算及所有设备/零 fallback/精确 nodeid 守卫保留，尚无修后运行。原 BERT selected-best gap15.425953905%>15% FAIL 与原 two-update 的零更新失败不变。

GC generic r2 source `a380d0c60691d93d3c9b337cbc46bfa85606aa1477100f7700491dc374792d56`、tool `648bc5a2f85c0cdf86eab1b5637f86a22f87c394cd509cc08c7c2c1268318a7c` 已冻结，18 TU 和 generated binding 仅语法通过；原40测试不变+23反例尚未运行。只读审查另发现“先取得普通 .data view、随后 holder adopt frame”的隐藏 owner 环边界，仓外 weakref/GC/native live-count 最小工具已准备，但尚未实测，不能以未来63PASS关闭 owner 安全。T5 失败退出且 cache holder 清空后，仅释放一次 CPU bootstrap900s；worker3558549/start163047387 的编译进行中，后继 priorview-repro120s 须独立审阅 bootstrap 收据再放行，没有自动过渡/重试。公开 False GC、RNG/AMP、CUDA、默认 fusion 与峰值仍 OPEN。Vision 的原五格/769字段 FAIL1 及原累计2858.094767101/4200s、余1341.905232899s保持，不启动新模型或重置预算。

## 2026-10-26 20:50 +0800：T5 梯度失败、BERT BF16 入口失败与 Vision 重复性收口

Day2–5 **仍未完成**。本次再次 fetch `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为 main、原验收、fix、Qwen 树 HEAD 的祖先，无需整合；main 开始 HEAD `765f1791f3b00f9129fb3ec6342dadf2c9dc2dbe`、完整内容 `062c8e11f04fe4ded8e3710e7bf7ac8f137d92a1cbb470ddcb501f95f667b78e` 与上轮冻结相同，91项原dirty/untracked保留、index为空。现场证据见未版本化 `20261026-root-resume-1231/snapshot.json`。下面20:08的“运行中”“未运行”均为历史快照；未重复full100、改变门槛或自动重试。

T5默认3 continuous/fresh resume均rc0，分别20:22:11/20:22:46结束，合法checkpoint50和source `efe22515`/实际CUDA core `6dfe0645`守卫通过。20:23:28终审为 **DIAGNOSTIC_STEP51_DIFFERENCES_RETAINED**：2803数组中119不同，404动态前向事件全部exact，protocol errors为空，前向前参数签名及源清单exact；静态capture后mode不同不能解释hook时行为。权威 `20261026-t5-ordered-mean-r2-1854/step51-default3-prepared-r1/{root-final-audit.json,paired-diagnostic-v2.json}`，比较SHA256 `6bd6e4ef70b3845bb6f00cb3c3b4078058fd2275862a5b39507276e57334c7c0`。119项全为梯度：62norm、54encoder矩阵、relative-bias/decoder-wo/shared各1；全局max_abs **0.0016089007258415222**位于encoder block4 norm gamma，不能统一称为微小误差。54矩阵16,098,319个差异元素全部在BF16格点，其中4,884,293个超过2ULP；尚未定位具体反向算子。历史双seed strict **FAIL771/2088**与旧reference失败保留，未合入main。

T5离线清单证明可不重训50步：checkpoint50全部257参数的shape/F32/rawSHA匹配本次前向前签名，原归档完整5inputs及4边界各634个RNG叶的摘要重新计算exact。`20261026-t5-backward-offline-r1/` manifest `bc35f3a45e8d5d17d84be3b153aa89a67375129a5d2a51bd24ce6a86da5b16fd`，50绑定/8离线checks通过。新单micro工具正在准备，复用原RecordingTrainer/Trainer.training_step；baseline先证明257梯度/rawloss/returnedloss与归档exact，否则停止，root审阅真实收据后才放行最小norm反向捕获并检查retain_grad观察效应。新诊断总上限1800秒、两阶段各900秒含清理，不继承或重置Vision预算，无自动过渡/重试，不手写loss缩放，尚未运行。

BERT两update工具已新绑定 `prepared-765f1791f`（原tool manifest `7a94b0ab…570a01`）。独立PT于20:13:17 rc0，17项终审通过，3837数组文件/8,321,258,420字节均重哈希，84次实际BF16/CUDA query、201参数/梯度及402moments齐全；第一步lr0无位移、第二步lr6e-6有真实更新。JT于20:23:37启动，却在 **20:30:51 rc1**，首次initial-fixed-eval的layer0 query hook观察non-BF16并拒绝；零次backward/optimizer update，没有配对数值结果。实际core `99959ad7b4bc433490da2fb3dd5e7e5005652790a3e965226bc7eb4858110ab4`、退出source守卫通过，原组空，queue为 `JT_CONTROL_FAILURE_RETAINED_NO_RETRY`、`main_source_freeze_active=false`。原hook抛出早于保存dtype，实际dtype/amp/选路无法恢复；静态确认cuBLASLt `compute_dtype_name`漏读BF16位64，与本次失败的因果仍待最小CUDA实测。只读审计 `20261026-perf-1311/bert-two-update-live-observer-r1-765f1791f/BF16_ENTRY_FAILURE_AUDIT.json` SHA256 `4df45a29f51765da1731a9d9bd9c53990f1fe1330764fadde78ab312fc7637c9`。原失败工具/cache不改，no_grad/grad-enabled两forward工具仅准备并正复核公开入口。原task-loss gap **15.425953905%>15% FAIL**及optimizer/scheduler原比较字段不变。

Vision v5在原cache/core `128db287`/jit_utils `6064ff34`、原最终40参数/eval65输入、默认3/seed821/实际CUDA/fallback0完成两fresh进程各32次：进程内及跨进程1024比较全部exact，均与旧continuous原件exact，与旧fresh仍527元素不同。repeat-a/b计费75.367760770/13.315828691秒；独立profile模型rc0，但postcheck误把含`shapes:`注释的FileName当路径，sources为空，**STOPPED_FAILURE_RETAINED**原收据保留，无重试。计费12.029656805秒后，原累计 **2858.094767101109/4200秒，余1341.905232898891秒**。`20261026-root-resume-1231/{VISION_V5_TERMINAL_AUDIT.json,VISION_V5_PROFILE_OFFLINE_AUDIT.json,VISION_V5_SELECTED_FORWARD_PATHS.json}`离线解析并逐项hash核实59份生成源码、421个现代op记录，profile输出亦与repeat exact。选中attention是`mea_fwd::forward<8,4,false,0,false>`（batch32，10次）及`<8,1,false,0,false>`（batch1，2次），forward命名空间无atomicAdd；同文件backward定义含atomic不能推断前向非确定性。三项cuBLASLt linear源含计时选算法，但原算法ID未归档，仍仅假设。原769字段FAIL1及五格FAIL不变，没有分配后继模型运行。

Day4–5 r6仍绑定原HEAD30085673/source bdae0ec0且仅共享1×8单轮诊断，正式代表尺寸/三轮/真实容量/稳定窗口未过。新 `20261026-adamw-scalar-minimal-r1/` 绑定完全同源隔离baseline，原AdamW设置不变：3小FP32参数、5步完整参数/梯度/一二阶矩63数组，原optimizer TOL1e-5，初值/梯度要求exact。PT已rc0并真实更新全部3参数；JT于20:42:30单次900秒启动，新独立correctness cache仍在扩展JIT，尚无完整数值或scalar-launch结果。另2步预热/3步profile只检验197×3×8模式能否在3参数复现，不是速度A/B。GC generic saved-value独立候选正在离线完成，10 TU语法通过不等于功能验收；原r4私有17pass和公共False/RNG/AMP/CUDA/默认fusion/峰值OPEN保持。

## 2026-10-26 20:08 +0800：CPU 与 GC 私有回归收口、默认 step51 执行中

Day2–5 **仍未完成**。本次恢复再次 fetch `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，为所检查工作树 HEAD 的祖先，无需整合；main 为 `8eb88dbdedf2d28a4af5f8ea2724372a29180b74`，原 91 项 dirty/untracked 与空 index 保留。原验收、fix、Qwen、T5 r2、GC r4 的实际分支、HEAD 和状态分别归档于未版本化 `20261026-root-resume-1155/snapshot.json`；没有 stash/reset/clean、push 或整合 `origin/2.0-refactor`。以下较早“未运行”“窗口交接”只对应原时间。最近生产源码结构证据仍为 1398 passed/2 accelerator skips，不为本次文档增量重复运行。

T5 r2 的受影响原生 CPU 回归已经实际结束：**15 collected/executed/passed，0 skip/0 fail，111.924542 秒，rc0**；CUDA 隐藏、has_cuda=false、无 Torch 导入、实际 CPU，完整源码与工具、worker/core/cache 身份不变，进程组与 holder 均空。receipt SHA256 为 `e4dea024098b8c13a90337e815316538d0130d6e34277b03edfe44a2021fe560`，实际 CPU core 为 `4d116d9df7775133f0f0033aca52356c3f30d5e65c9626895edc9d478d1785da`。它与此前 17 项真实 CUDA、两新进程各 32 次默认3局部回放及独立 PT 证据共同满足当前候选的局部前置；未合入 main，历史双 seed strict FAIL 771/2088 不变。

在这些硬前置通过后，`20261026-t5-ordered-mean-r2-1854/step51-default3-prepared-r1/` 于 19:49:43 +0800 实际启动默认3有界 step51 campaign，保持原 seed821/batch2/GA2/BF16/名义100-update schedule，compile key7413；每个连续/恢复 worker 上限3600秒，捕获 step51 后停止，不运行 full100。恢复进程只会在连续段 rc0、passed capture、完整 trace 和合法 checkpoint50 均存在后启动，无自动重试。当前复核 supervisor PID3475041/start ticks162459728、continuous PID3475053/start ticks162459887 均匹配，指定 GPU UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5` 的实际 worker 存活、CUDA 已启用，模型算子编译产物持续增加；20:07已观察至少21次实际训练更新，loss/grad_norm有限、LR依原schedule从0升至3e-5。初始probe实际位于`continuous-output/initial-probe.npz`，SHA256 `3fff39ac564ef90d9f4e15d89fcb3605203d2b7c21dd9b98ebdc52bf55022bac`；此前只查capture目录的absence判断已纠正。**尚无step51或合法checkpoint50，不能计为恢复通过**。源码身份为 `efe225153ddd2bb404f3dbef174fe22efe60319dd8057c4e5ae4a548b297765b`，campaign tool manifest 为 `c9d07eb40ad66771aeaabd61b89c7eddfa184b967499a0678af038f93a85f6ce`。另启动仅观察本campaign的有界离线finalizer（PID3497250/startticks162559854，最晚13:52:43 UTC退出），不导入框架、不启动/重试模型；两侧终态成功且组空后才运行比较器并写`root-final-audit.json`，guard/数值失败原样记账。九项离线adjudication控制确认即使工具exit0，字段减少或任一protocol/array/event/signature差异也不能通过。先复核实时 process/execution/status，再决定下一步，不重复启动；两侧成功后须显式查看 `compare_pair_v2.py` 的 protocol errors、array/event differences 和参数签名，不能用其 exit0 代替数值判断。

GC r4 已在独立候选真实执行，**17 tests PASS/0 skip**，positive、独立 PT 和9项解析比较均按原 `2e-7/2e-6` 通过，actual CPU/fallback0，source `997d1acc6960d2897128b0eaa4c0b048113f4faeae34fa5bb1ea4d64443a845b`、tool `d8e97d98fd45fa1def776a01779b1f8681893084cbaa4f826f0ab8e113524a0a`、core `63aac1cf8645835cb27654c6d1d5630643fdd0a15b1943d6f28ac1e55441b487` 守卫通过，所有自有组退出。四文件修正让实际梯度遍历遇到 Ready remat 值时标记一阶策略，不再仅依赖 saved-value reader；add/sum、mul、tanh 三负例均按声明拒绝高阶，普通无 remat 的线性图仍返回正确零二阶。r3及后继原负例FAIL原件保留。权威 `20261026-gc-native-s1-r4/private-cpu-s1-verification.json`；真实小图计数为6 allocation/48B释放、1次重放/3次saved read，**不是净显存或完整step峰值节省**。公共 HF `use_reentrant=False`、RNG/AMP、CUDA、默认fusion及模型/显存门禁仍OPEN，未移入main。

Vision v4 bootstrap4 已 rc0/110.243977741秒，实际新 core `128db287831b48ab25cab2af2fac44af711524b8936b9d78c18cfaffe1eacdc2`、jit_utils `6064ff343b8c0df12e53818f8c057914c396d87748b50c9e072666c2c40cb4f6` 通过source/settings/inventory/40个生成文件、RUNPATH及进程映射核验。repeat-a 却在原180秒上限触及 **timeout/rc-15，181.116214559秒**，仍在编译原模型算子；没有完整 eval65、32份数组或终态 fallback/参数 hash，repeat-b/profile未启动，无新数值或kernel归因结论。v4新增active291.360192300秒，原累计 **2756.63591191615/4200秒，剩1443.3640880838502秒**，未重置。权威 `20261026-vision-eval-replay-r1/{root-status-v4.json,terminal-audit-v4.json,TERMINAL-v4.md}`，source/fixture/seed/tool终审通过、组空/无cache holder。新v5已在同一实际cache路径完成离线准备（`20261026-vision-eval-replay-v5/`），manifest SHA256 `d49a88f442ffce9c769d63d9eaa4c33abf98bea5ad39ee07bb9189719df651bf`，11项离线检查及verify-only通过，无framework/模型/JIT/队列。当前1431项partial cache已只读归档并逐项校验，archive SHA256 `d63361ddd26a53e8c90b5424281f3e75643497a880143d60c893ec43d62a08b3`，归档实测保守收费0.7456089190673083秒后累计2757.3815208352173/4200秒，剩1442.618479164783秒。首repeat-a900秒、repeat-b180秒、独立profile180秒加各30秒清理和30秒拒绝调用reserve后仍余62.618479秒；每阶段需独立收据/窗口，任何失败停止，不扩预算。尚未执行；原769字段FAIL1 `final/eval/logits`及五格FAIL保持。

Day4–5 r6小格的机械pipeline通过与共享负载边界保持上一节结论，原执行身份继续绑定 HEAD30085673/source bdae0ec0，不转绑文档提交。新增离线分析 `20261026-root-resume-1155/R6_OFFLINE_ATTRIBUTION.json` 核验原row/trace/profile哈希和每个调用序号：3次update的 `optimizer.step()` site195恰有 **4728个4B scalar array与对应CUDA kernel，即197参数×3步×8**，占7995个kernel的59.13696%，对应kernel duration sum8.412524ms（23.81742%）；host scalar launch合计32.725080ms。原native attribution记录3步profile wall310.233908ms，其中Python/graph177.220402ms、operator launch103.759119ms。这是明确的最小复现目标，尚无匹配干预证明其造成未profile的PT/JT耗时差距；不将profile的层级CPU/CUDA行相加，也不替代三轮、代表尺寸、真实容量或稳定窗口。

BERT两更新工具仍仅离线准备，33项原preflight与新增只读hook审查均完成，无framework import/模型/JIT/队列。审查核对Trainer/Accelerate的callback、GA2、裁剪/step/scheduler/stop与BF16 hook调用顺序，未发现首次执行必然中止的确定性冲突；`initial_lr`、`verbose`、`_get_lr_called_within_step` 元数据差异仍保留原字段exact比较，不能消掉差异制造PASS。独立报告为 `20261026-perf-1311/bert-two-update-hook-static-review-r1-8eb88dbde/STATIC_RUNTIME_HOOK_REVIEW.json`，SHA256 `21e3b8ae23edc51b0e5dd31426903bc072bcb6764464fb46385f9f290191d0d1`。本次main文档/HEAD变化将使旧 `prepared-8eb88dbde` 完整绑定失效，后续执行必须新命名重绑，旧准备不改写。原task-loss gap15.425953905%>15%继续FAIL，实际两更新诊断尚未运行。Qwen独立候选原前向/训练/恢复通过范围保持15:26节与原adjudication，不外推main，也不关闭公共GC。

## 2026-10-26 19:36 +0800：默认归约局部通过与显存采样闭环

Day2–5仍未完成。再次fetch目标origin/cgq_transformers仍e5cffe1ae60c418f53c5bb1cdca9182577fd2b81，八个相关工作树HEAD均已包含该历史，无需merge。main HEAD30085673abb3cf3eb5d2fad6a5bb90aa13c16301、91项dirty保留；未stash/reset/clean/整合origin/2.0-refactor。详细现场已保存在live-snapshot.json；PID与GPU只对应采样时刻。

T5 row-mean原r1的17项真实CUDA结果为15pass/2fail，原source3290d483、core6187ed5a及全部原件保留。lazy para0失败由launch_bounds256与实际p2上限1024不一致触发；独立最小过程默认宽度失败、显式1024通过。新r2只在隔离复制树让level0 p1/p2使用同一cuda_thread_num；另一失败是profiler把两次调用聚合到Count2的一行，测试改为求Count而保留单reduce/数值所有权断言。17方法/32重复/FP64/原容差不变，新增128/512launch宽度子例。r2 source efe225153ddd2bb404f3dbef174fe22efe60319dd8057c4e5ae4a548b297765b、tool16f70a97f88cef00f3af1c72f5d0913c519ed401a37721b6d799f1b850c8aa26、实际core6dfe0645db0cdd9d4b30d38165a32a580896b19064557cb73056a96b62e24645；bootstrap63.970s、全部17CUDA tests221.190s且零skip，source/cache/core/GPU/fallback0守卫通过。两fresh native各32次冻结r38默认3回放，五阶段进程内/跨进程exact且finite；独立PT32回放满足原局部atol1e-5/rtol1e-4，output maxabs9.5367431640625e-7。权威为20261026-t5-ordered-mean-r2-1854/replay-comparison.json。这是局部重复性通过，CPU与新默认step51尚待直接执行，旧双seed771/2088 strict FAIL不变；未复制main/启动full100。

GC隔离r3实际CPU bootstrap/14tests/positive/PT与9项解析比较通过，fallback0，source473f35d7/core9a210490身份一致；小图真实释放6allocation/48B、1次重放/3次saved read。48B不是净显存或峰值节省。新三项一阶策略探针rc1/4.407s：mul/tanh正确拒绝高阶，add/sum一阶exact但返回数学正确的零二阶而未按S1承诺拒绝；原FAIL保留。r4仅隔离准备四文件修正与原14+新增3项，source997d1acc，未运行；公共False、RNG/AMP/CUDA/defaultfusion/memory仍OPEN。证据根20261026-gc-native-s1-r3与20261026-gc-s1-next-design-r1。

Vision eval回放v3已在原300sbootstrap2上限超时，300.946s/rc-15，source/fixture/seed退出守卫通过、组空，未执行eval/repeat/profile；剩余原累计预算1734.7242803839035s，不重置。缓存绝对路径改变使stamp/逐对象key/generated内容/RUNPATH变化，复制不等于warm cache，重链后也不能预设旧core字节hash。机制报告与单次cold-aware后继设计保存在20261026-vision-eval-replay-r1；原769字段FAIL1 final/eval/logits和五格FAIL均不变。

性能r5采样失败的原error/sample未保存，无法补造其值；后继实测确认独立JT venv缺pynvml，固定增nvidia-ml-py12.560.30后纯NVML及公共native sampler通过。r6保留原采样/allocator守卫，增加验证前rawmemory落盘和sidecar绑定。新PT小格19update/27项终审PASS，JT亦rc0/28项终审PASS，原冻结comparator为MECHANICAL_SMALL_PAIR_PASS，最终source/tool守卫通过、两组与holder为空，19:35:14解除main冻结；两端仅原1×8/单轮诊断，不能替代代表尺寸/三轮/容量/稳定窗口。该轮main完整内容冻结bdae0ec0与HEAD30085673保留原执行身份，不重绑到后续文档提交。JT rawNVML error=null/samples14/peak3,238,002,688B，allocator live2,469,275,648/reserved2,731,540,480B；实际CUDA trace34,348events/7,995kernels。PT/JT hot median为0.111462s/0.151168s、观测ratio1.356，CV约0.226/0.152；不作正式速度或峰值显存结论。最初终审误把torch.Tensor要求为jt.Var同对象，已保留并只按已冻结frontend子类型契约修正审计谓词，计算脚本、守卫和比较器未变。权威20261026-perf-1311/v3-small-memory-evidence-r6-30085673/PAIR_FINAL_AUDIT.json。BERT原selected-best gap15.425953905%>15%继续FAIL；firstmicro train/dropout/GA2与旧eval probe不是同轨迹因果对照，未重训。

当前串行首编窗口交T5受影响CPU单次900s原15项测试，入口20261026-t5-r2-cpu-regression-r1/root-status.json；须重查真实receipt和/proc，不能按此快照推断完成。GC r4排其后，Vision v4单次cold-aware工具已离线准备而未运行，承接原预算；T5默认3 step51工具仅准备，需CPU通过后再启。main没有合入这些隔离候选的新改动，没有重复full100或放宽门槛。

## 2026-10-26 18:40 +0800：Vision 恢复残差与性能显存采样失败

Day2–5 仍未完成。主树已有本地提交 `6f61c185595242ba95dd711ee16c2ba53ba4ad2e`（公共优化器清零缺失梯度与设备回归，未 push）；下方旧“未提交”只对应原快照。再次 fetch 的 `origin/cgq_transformers` 仍为 `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，是六个相关工作树 HEAD 的祖先，无需整合；main 91、原验收 51、fix 85、Qwen 87、Vision 95、T5 mean 90 个 dirty/untracked 路径均保留，没有 stash/reset/clean 或整合 2.0-refactor。性能 r5 两端运行期间完整主树冻结，最终源码守卫通过后才解除；旧收据继续绑定原执行身份。

Vision r2 的实际 CUDA bootstrap、61→67/save64 连续段和新进程 64→67 恢复段均 rc0，耗时分别 399.642、425.654、13.014 秒；原默认 level3、无 module hook、seed821、配置及数据不变。新 checkpoint64 的 SHA256 为 `887b7f64fd74007ed61cdc5b4a030cce12477814940059a671e685a829190622`，model/40 个 momentum/scheduler/scaler/RNG/cursor/config/fixture 审核通过；两 worker 与 bootstrap 的 source/cache/core 精确相同，实际 GPU UUID `GPU-261ac14d-3f54-8ca9-ebaf-2830b8992cb0`、CUDA:0、fallback0，核心 SHA256 `1dc6cefcd1527f5a830aa0e6b70a0a6262e12c7547c6fefbbe8d6ee5c54a64ad`。

原七项零容差 comparator 仍 **FAIL：769 字段中 1 字段不一致**，为 `final/eval/logits`，max_abs `5.960464477539063e-8`。两个 batch32 中分别 263/320、264/320 个值有 bit 差，尾 batch1 的 10 个值 exact；最终 40 参数、40 momentum、9 scheduler 字段及 loss/accuracy 逐字节 exact，无 buffer 字段。history65–67 的 30 个统计字段 exact，但它们不含逐元素梯度或每步参数，不能外推这些值已对齐。原两个 artifact audit 均通过，compare rc1 后所有进程自然退出，没有重试、cost 或完整矩阵。权威入口为仓外 `20261026-vision-model-recovery-r2/root-status.json`、`same-runtime-exact.json` 和 `20261026-vision-eval-replay-r1/bitwise-audit.json`。这是新的有界恢复失败；原五格模型 FAIL 不变。下一项是相同最终参数/原 eval65 输入的两新进程各32次无 hook 回放，并另做一次 native profile 定位实际 kernel。ViT 实际为17 tokens；候选缺少 main 短行 FP32 softmax 修复，但 fused attention 可能绕过该路径，尚未认定残差根因，没有猜测性复制补丁。

Day4–5 新 small/profile r5 绑定 main HEAD `6f61c185`、完整内容身份 `be0ebfed643527af36f44fc07a3cdda3a777cc8cc17e2a0a857d7526a2fd8967`、工具 manifest SHA256 `a4eb68daa513ae6eafdbcf48e621e423b8989259cf3a5945726f3afa1bf6657e`。原 r3 计算脚本字节未变，两端使用相同 GPU UUID 与新的独立 performance cache，199 FP32 master、2 buffer、197 梯度、两 pooler 排除项和 objective 均保留。独立 PT 一轮 rc0：首次真实 AdamW update、2 warmup、10 unprofiled hot、3 memory 和3 CUDA attribution update 完成；原生 trace 可解析，65,398 events、8,289 kernel events。JT 已越过旧 zero_grad(None) 失败并生成10个 hot provisional 样本，但独立 memory 阶段的 NVML observations 守卫失败，18:38:20 自然 rc1，无 final row 或 CUDA attribution，不运行 pair comparator。原始 completion 确认主树源码未变，组与 cache holder 已空。权威根为 `20261026-perf-1311/v3-small-post-zero-grad-r5-6f61c185/`；provisional 不替代完成收据，共享负载也不构成发布速度或显存结论。下一项先查明 profiler 当前 NVML schema/采样事实，保留失败，不删采样要求或原字段、不无变化重跑。

T5 五文件 row-mean 候选仍绑定 `3290d4839ef3399f784a4e3ef479b1f06191b5a89c58e30a17ba4954cca104be`。独立静态复核未确认算子 bug，但查出测试 worker 核心身份与跨进程数组比较缺口；已在仓外工具 v6 补 bootstrap/source/cache/core、pytest worker/fallback/NVML UUID、两 native replay/core 精确核对和独立 A/B/PT 比较器，旧 v5 全部保留。v6 manifest SHA256 `e80bb72214d5b28a595c2e6fd7d0abf33ab23e64f70b4d898e9ef30755e6e514`，语法与干运行通过。性能退出后仅启动新的 T5 bootstrap（900 秒 bound）；尚不能把编译进度计作17项 CUDA 回归、32次回放或默认 step51 通过。后继仍需实际17测试零skip、两新进程与独立PT比较、受影响CPU回归及有界默认3 step51，禁止 full100 或从 r66/r70 启动 resume。

GC 从旧 shortcut blocker 推进到隔离 `gc-native-s1-r2` 的真实私有 CPU 候选：原生 metadata/value lease、last-use eviction、saved-value reader 和按 backward invocation 重放，mul/tanh 使用原反向图，不嵌套 `jt.grad`、不开 no_grad/keep_graph、不清 finished；相关正负测试代码已写，尚未执行。语法编译仅验证翻译单元，不证明梯度、释放、重算或显存效果；冻结 HF `use_reentrant=False`、公共 GC/RNG/AMP/CUDA/默认融合门槛仍 OPEN，不能用 S1 私有能力代替。BERT 原 selected-best gap 15.425953905% >15% 继续 FAIL，Day4–5 原三轮、10样本、稳定窗口和真实容量边界继续开放。

## 2026-10-26 17:52 +0800：公共优化器修复与最终设备回归

18:08增量：本批完整 Torch-mode structure 已通过 **1398pass/2accelerator skip/exit0**，1400 collected，pytest230.24s、supervisor231.516s；完整内容身份 `14c7d22b1ba59bd7a3ca6ec45e22e1f744f00bf02081b201ecb5abd52bfd1f5f`、HEAD37021507，前后未变，终态收据在 `20261026-root-continuation-1713/gradscaler-fixture-fix/structure-cpu/receipt.json`。layout、MANIFEST、相对链接与diff检查通过。Vision独立CPU bootstrap及实际15测试/15格矩阵均PASS，CPU worker/core/source exact、fallback0；随后只释放CUDA bootstrap-model1，尚无新模型恢复结论。不同阶段仍绑定各自执行时身份；下面旧“尚未运行/未提交”只对应17:52快照。


Day2–5 尚未完成。恢复现场后重新 fetch 的目标仍为 `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，已为 main/原验收/fix/Qwen/Vision/T5-mean 六树 HEAD 的祖先，无需整合；未 stash/reset/clean、未合入 2.0-refactor，全部本地工作与旧失败保留。主树 HEAD 仍为 `37021507860392f33565d59c4907b6acaae47a99`；本节最终 CPU 扩展回归的完整内容身份为 `f76fcc31181f0ce5130cc8cff8573f8092e8c7c22ab7955e3a0ca35edda63e51`，该身份取于本文更新前，不能重绑到稍后的文档或提交。

性能工具发现的 `zero_grad(set_to_none=False)` 崩溃已在公共 AdamW 最小用例复现：第一次更新后，原生 zero_grad 对合法的缺失梯度调用 zeros_like(None)。修复仅在 `python/jittor/optim/base.py` 原循环跳过 None 两行；compat 的梯度槽位和未使用参数不补造零梯度，既有 tensor/holder/cache 规则不变。新增回归将 CPU/CUDA 分为真实方法，保留 False/True 连续清零、参数先后顺序、延后启用、状态步数、holder identity、独立 NumPy AdamW 轨迹与实际 device/fallback 检查。

修复后独立 PyTorch 2.6 CPU/CUDA 各十条记录均通过：参数、梯度、loss exact；一阶矩最大差 `1.1920928955078125e-7`、二阶矩 `7.450580596923828e-9`，原 `atol=rtol=2e-6` 不变，实际 CUDA:0/fallback0。原生 focused 为34pass/1accelerator skip；CPU core 合计328pass/46skip/1xfail/0fail。最终CUDA公开回归 **8pass/0skip/exit0**，最低 accelerator executed>=1 规则保留；最终CPU扩展 **34pass/1CUDA环境skip/exit0**，完整源码前后相同，进程组为空。仓外权威入口为 `20261026-perf-1311/zero-grad-unused-fix/FINAL_RELEASE_AUDIT.json` 与 `20261026-root-continuation-1713/gradscaler-fixture-fix/focused-cpu/receipt.json`。本批尚未提交，不把这些结果写成模型或发布通过。

原扩展测试中的 GradScaler fixture 在 scale 前 unscale，被当前独立 PyTorch 同样拒绝；修正仅显式 device 并初始化 scale，原数值、一次 item 及 inf 跳步断言不变，生产 GradScaler 未改。独立 PT 控制实测有限梯度 [8,-4]/[2]→[4,-2]/[1] 且更新一次，inf/NaN 跳步且参数不变。CUDA 首次 rc3、为保护归档 core 而在实际计算前停止的旧缓存编译，以及7断言PASS但设备计数0造成的 gate exit1 均保留；最终采用独立 correctness cache 与真实 CPU/CUDA 方法解决，不改最低门槛、不使用性能 cache。CPU core 的原 cache-list-equality 字段为 false，是新增两份 cfg core；另份 preservation audit 逐路径证明原四份及最终全部六份核心 SHA256 不变，原 receipt 未重写。

Vision 后继工具 r2 仅纠正 r1 的CPU选择与身份核对：15项真实CPU测试必须零skip，加原15格广播矩阵；worker与bootstrap逐项核对 source/cache/core/shim/GPU，原模型、seed821、默认3、无hook、61→67/save64/fresh64→67、全部字段和七项全零阈值不变。新计划 SHA256 `39f14347de3342dd91cd1a147d05ddf0a028b9691f9f62c9bca078a3949bab7d`，78文件/11脚本，累计剩余3241.122788784094秒；串行窗口现仅释放CPU bootstrap，任何新结果以 `20261026-vision-model-recovery-r2/root-status.json` 与终态收据为准，原五格FAIL不变。T5 row-mean候选 verify-only通过、内容身份 `3290d4839ef3399f784a4e3ef479b1f06191b5a89c58e30a17ba4954cca104be`，尚无实际编译/修复证据。

GC静态核对确认冻结 HF runner 明确传 use_reentrant=False。现有 no-grad Function 原型只给显式参数接梯度；补额外module slots仍不能充当原非重入门槛。非重入原图保存值释放/重算设计与最小原生候选另在隔离lab推进，尚未运行，也不把23项静态检查或现有CPU正例计作公共GC通过。CPU RNG/AMP/live-memory controls仍未运行，旧冻结工件不改。BERT原任务loss FAIL、Day4–5原三轮/十样本/真实容量门槛继续开放；旧r4 small绑定已因修复过期，不执行旧绑定或旧r5 waiter。

## 2026-10-26 16:44 +0800：原生重复性收敛与新公共优化器失败

Day2–5 仍未完成。主树已有本地提交 `37021507860392f33565d59c4907b6acaae47a99`（未 push），只提交已验证的 planner/短行 softmax 修复、对应测试、清单三行与本批文档；混合清单其余十二行和所有无关 dirty 工作均保留。下方 15:26 的执行回执继续绑定旧 HEAD，不重写到新提交。本节之后的原生候选和 GC 原型都在独立工作树或 lab 运行目录中，尚未合入主树。

Vision 的 r2 首轮原生回归为 13pass/3fail，三个失败由 profiler 名称的子串误选引起：测试标签自身含有 `ordered_broadcast_grad`，因此误选了普通融合 kernel。保留 r2 失败及所有原件后，候选仅修正测试的完整 `choices` 键解析，并用锚定 `#define opN_name` 的声明列表检查单个 reduce；数值断言、种子、容差、重复数和循环 AST 未变。r3 计划 SHA256 为 `10d1d3b7f19a4a66c4a6e8a51458403a5ba08e924dbe55357df55365176ad7c3`；16 项原生测试全部执行，**16pass/0fail/0skip，420.07s**。随后两个新 Jittor 进程各对 cls_token、position_embeddings、conv_bias 三份冻结 cotangent 重复32次，进程内及跨进程 exact；同一 CUDA 的独立 PyTorch 对照最大绝对差依次为 `1.1920928955078125e-7`、`8.940696716308594e-8`、`2.384185791015625e-7`，原 `atol=1e-5/rtol=1e-3` 超限数均为零。实际 CUDA、fallback0，运行核心 SHA256 `7bf882ef91b705d143767624ac5627930bee0df4488379f1f99a0ad84393c1b8`，源码/输入绑定通过；16:15:54 自然结束，无遗留进程。仓外权威入口为 `20261026-vision-native-validation-r3/terminal-verification.json`。这些只关闭原语重复性局部诊断，原五格 Vision FAIL 不变；下一项为默认3、无 hook 的有界模型恢复及成本门禁。

GC 已从静态设计推进到真实 CPU 可行性：`20261026-qwen-causality-1300/gc-feasibility/cpu-first-r3/` 的两探针于16:29:04自然完成，29项能力源码未变，原型所有观测均在CPU、fallback0。显式传入输入和参数的正例精确得到 \(g_x=[13,17]\)、\(g_w=[7,9]\)，调用时梯度模式为 `[1,0]`，即一次无梯度前向与一次反向重算。闭包负例按预期得到捕获参数梯度 `[0,0]`，数学所需为 `[2,3]`，证明该原型不能用于任意闭包或模块参数。终态核验 SHA256 为 `3fef5043a04935652b6c531177e9526b6979fcf20bd0af973c21f14d6b3aa1e0`，实际 CPU 核心 SHA256 `b3af6ee8bab6456ff34f6c1cc350d14ab1e716546cd8b9fcb4616e7535444819`。r1 原型误读 CPU 构建中不存在的 CUDA TF32 flag 的失败、r2 未执行稿及 r3 首次 import-only rc3 都保留；修正只按实际 device 查询相关 flag，未用默认值吞掉缺失项。RNG、AMP、真实 activation live-memory、CUDA 和公共 GC 集成仍未执行，GC 发布门禁继续 OPEN。

Day4–5 新小规模工具在主树 HEAD `37021507`、完整内容身份 `c51928b12fa837e62a77e6f7b49768292d6287a1a3132c2a9ad38f672099caab` 下实际执行。r2 的 Jittor 入口在模型前无条件读取 Torch namespace 的 `__file__` 失败；新 r3 仅修 metadata 并保留全部计算/配置/fixture/更新阶段，manifest 为 `1262d25387b0fe715f3e57f30456acf0ca73e43ff8d5543b27987a4f0e173817`。独立 PT 的 b1/s8/FP32 小格完成真实更新、十个热态样本及分离 memory/CUPTI 阶段。Jittor 首个 update 后，在下一次 `optimizer.zero_grad(set_to_none=False)` 触发 `optim/base.py::zero_grad` 对 `None` 调用 `zeros_like`，16:41:26自然rc1退出，无热态样本或完整配对结果。该目标原本不使用两项 pooler 参数，其197项梯度覆盖条件保持不变；不得给 pooler 造零梯度、改目标或跳过 zero_grad 绕过失败。新公共路径问题已进入最小复现与核心/compat 边界定位，旧 r2/r3 失败和性能缓存独立保留。失败回执最终源码 guard 通过后，才解除主树完整冻结以修复；后续运行必须建立新源码身份。

T5 的 level4 有界 step51 结果与历史默认配置 strict FAIL 保持原口径。新增原生 row-mean 候选仅在独立工作树准备，完整复制并核验固定修复树85项 overlay，尝试在 CUDA/连续单末轴/FP32或FP64/实际非空 fold/执行level3或4时隔离归约并使用既有 block ownership；没有改全局默认、模型 hook 或公开确定性承诺。八项 public CUDA 回归、原 r38 冻结边界32次回放及有界 launcher 只完成离线准备，入口为 `20261026-t5-ordered-mean-1554/DESIGN.md`；尚未编译、执行或形成修复证据，不启动 full100。

## 2026-10-26 15:26 +0800：此前证据与源码边界

本节取代下方较早现场快照作为本日结果入口；历史失败的原始收据、日志、checkpoint 和 cache 均保留。Day2–5 **未完成**，不以仓库门禁或单个候选的局部门禁代替发布验收。本逻辑任务开工已 fetch `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`，它是相关 HEAD 的祖先，无需整合；未整合 `origin/2.0-refactor`。主树验证基线为 `395949d7239764baa0a614a3ac8b2f204d6e41bb`；四个原有工作树及其 dirty 工作保持独立，新建 Vision 候选只复制修复树完整 overlay 后再修改，没有覆盖原树。

| 验收项 | 最新直接证据 | 尚未满足的边界 |
| --- | --- | --- |
| Qwen BF16 公共前向 | 无 hook、无模型补丁，29,171,712 logits exact，原 atol=0.005/rtol=0.03 通过，实际 CUDA/fallback0 | 仅绑定 87-path 独立候选，主树不继承全模型结论 |
| Qwen LoRA 双 seed | 821/823 各 full102 与 checkpoint51 新进程 resume51 完成；每 seed 930 项原 strict comparator 零容差通过；原 selected-best 双 seed loss gap 0.02542264% <15%；17 个协议字段全部匹配 | GC API 实际仍直接调用 fn，缺少 activation recompute/显存节省，不能标整项发布通过 |
| T5 BF16 | 原真实 planner 崩溃已最小复现并修复；有界 level4 step51 新 cohort 2,803 数组与 404 动态事件 exact，参数签名/源码相同 | 543 静态 mode 快照不同；原 reference comparator FAIL，历史双 seed 771/2088 FAIL 保留；没有重跑 full100，尚非默认配置恢复通过 |
| Vision | 广播反向归约 level3 重复不稳定、level4 稳定的因果诊断；v2 原生候选隔离准备 | 原五格 comparison FAIL；candidate 尚需真实编译、原语/模型恢复与成本验证 |
| BERT BF16 | 同源同权重真实 CUDA 初始 eval forward 与全部 201 参数梯度按原预算通过；仅0更新局部诊断 | 原 selected-best 15.425953905% >15% FAIL 不变，checkpoint 选择差异与定向 probe 不替代原任务指标 |
| Day4–5 | r5 已超时，无测量；新性能工具有合同及 PT smoke | 无完整稳定三轮配对、真实容量边界或可信的常用路径退化归因 |

Qwen 权威收据为仓外 `20261026-qwen-causality-1300/lora-campaign/qwen-adjudication.json`，SHA256 `26fe28c4d8d744be6c9201f6192e001741c68e57be5f7002de423759b728748a`，状态为 `FROZEN_QWEN_FORWARD_TRAINING_RECOVERY_TASK_ENVELOPE_ARTIFACT_GATES_PASS_GC_RECOMPUTE_OPEN`，`overall_release_acceptance=false`。候选 HEAD 为 `9e0e187bb013e71d85111acfd668e46aeff9f9d2`，87 个 dirty source 的 fingerprint 为 `441ddd71f35d22da9826fdfcdb699110e3087eae8b621562411a1049e3a5c342`。六个新 checkpoint 每个均核对 703 个 optimizer/scheduler/scaler 原始签名和 112 个 adapter tensor；每 epoch 101 个样本恰好一次，112 个最终 trainable 全部真实变化，save/reload/merge/generation/cache 满足原预算。任务指标 PT loss 均值 2.4566192627、JT 2.4559947252；样本标准差分别 0.0000350662 和 0.0014415908。控制器较早的 `TASK_METRICS_PENDING` 已由离线 adjudication 覆盖；四个正式 worker 均已自然退出，无需重复训练。GC 的函数签名与 config 不构成功能证据，原警告和 `GC-FUNCTIONAL-GAP.md` 保留。

T5 的生产核心复现为 `20261026-t5-reduce-repro-1315/`：合法 control 可执行，空 tn range 在旧 core 的同一 `plan_reduce_thread_order` PC 崩溃；修复让空 range 中性，并在所有 reduced loops 已消除时保持 IR 不变。主树与修复树只应用已验证的 planner 修复和定向回归。第一次新有界 campaign 在 update50 后的 CUDA eval 编译中撞到 1800s supervisor 预算，rc-15，没有 checkpoint/step51，已单独保存 `terminal-audit.json`；这是 supervisor timeout，不能归为再次 SIGSEGV。第二次 `20261026-t5-step51-fixed-r2-1434/` 复用同源、同 fixture/seed821/compile key7404 和 warm cache，使用新输出与 3600s bound，continuous 与 fresh resume 均 rc0，checkpoint50 合法、实际 CUDA/fallback0、源未变。权威 `paired-diagnostic-v2.json` SHA256 `c51ec052bbc2b6af5ebdcff10581f932e93077367017b488b88fbce085c72cfe`，状态 `DIAGNOSTIC_TENSORS_EQUAL_STATIC_MODES_DIFFER`；静态快照取于 delegated training_step 前，不能替代实际动态 mode。旧 v1 与 unchanged retained-reference comparator 的 FAIL 原件均保留。

主树本批只移入 planner 与短 FP32-row WarpSoftmax 两项原生修复，各有真实 CUDA 回归；WarpSoftmax 梯度加入现有 inventory。冻结校验绑定 2,216 个源码/测试/工具文件，fingerprint `ac2835e5bbd458bd9a9119efee107c8036a8785dcd45fe9247483f45630a8af3`，验证前后未变。CPU core 为 native 101pass/43skip/1xfail，Torch 227pass/3skip；受影响结构 30pass/0skip。主树 CUDA 全部两文件为 15pass/1skip，actual use_cuda=1/fallback0；唯一 skip 是入口未配置独立 Torch oracle，该项已在明确 native 入口、独立 PyTorch 2.6.0+cu124 上补验 1pass/0skip，实际 CUDA/fallback0，forward exact、backward 满足原预算；首次继承 shim 路径变量造成的导入冲突 rc1 另存，未计算模型或改源码。完整 Torch-mode structure：1398pass/2skip（均为accelerator类别），1400 collected，246.84s，exit0，source unchanged。主树 core SHA256 `80cd0a83180d746aad4bfba465964754f2f18973877bf0c2628ad803994874b3`；日志与 guard receipts 在 `20261026-main-affected-verification-1340/`。这些是受影响仓库维护证据，不能迁移 Qwen 独立候选全部模型门禁至主树：87 个候选 dirty path 中 85 个与主树不同或缺失。

BERT 配对收据为 `20261026-perf-1311/bert-r10-diagnostic/paired-comparison-r1.json`，SHA256 `6450853c9862eb13bd95ec1edc7577a3141f9d90e6b8e465331efec7ef484b26`，`release_acceptance=false`。两端绑定同一完整主树身份 `288c975daa0b87de33ef6f35ecb166e520552bfa444d0b8657ec26eea084039f` 和旧 HEAD `395949d7`，仅在 Jittor 最终 source guard 完成后才解除全部文件冻结。201个 FP32 master 均逐值/原始字节 SHA exact；BF16 eager、eval但梯度开启、TF32关闭、2×40 原输入，观察36个BF16 QKV、12个FP32 softmax输入，指定CUDA/fallback0。logits max_abs=0.00390625；loss PT=0.900390625、JT=0.90185546875，均过原0.005/0.03；109,483,778梯度元素无超限，max_abs=0.009765625，原梯度预算0.01/0.05。12层score与11层probability仍有额外中间诊断超限，不能宣称 attention exact、task-loss 原因已定位或训练改进。首次 cache bootstrap rc3 与第二次 from_pretrained meta 装载 rc1 保留；成功r3使用公共构造器+同一safetensors严格装载，权重/配置身份核对通过，不把装载前置失败当数值FAIL。此结果只覆盖初始局部 forward/grad，没有 optimizer update。

下一步按串行首编顺序推进 Vision v2 的 pre-fix copy contract 复现、独立 cache 实编/真实 CUDA 与默认配置有界模型恢复，另对通用 native GC 先建立梯度/RNG/AMP/真实重算和 live-memory 最小证据。Day4–5 正式测量继续要求冻结源码、独立性能 cache、真实 optimizer update、预热后至少10样本、三轮交替、同步统计与 allocator/NVML peak；共享卡上的正确性或 profiler 诊断不改变该退出条件。不创建无限 waiter，也不干预其他用户进程。

较早现场与诊断记录保留如下；其中 PID、运行中和未启动状态均只对应原记录时间，不能覆盖本节。

## Workspace and Live State

At the 06:07 +0800 refresh, the main worktree was at `91b543c60e60fc3d19b2f7407c4a8f631385c47c` with 99 dirty/untracked entries. The original Day2–3 worktree remained at `9e0e187bb013e71d85111acfd668e46aeff9f9d2` with 51 entries; the repair worktree remained at the same HEAD with 83 entries. The separate Qwen diagnostic worktree also remained at that HEAD with 85 entries, matching its frozen source identity. `origin/cgq_transformers` was fetched at `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`, an ancestor of the main HEAD; no merge was needed. No stash, reset, clean, overlay, or origin/2.0-refactor integration was performed. The 77 dirty source files bound to Qwen r44 retain their recorded hashes.

T5 r66 continuous capture PID 2788927, started at 05:56:01 +0800, exited with signal 11 after producing only `continuous-output/initial-probe.npz`; there is no step-51 trace, checkpoint-50, or continuous checkpoint receipt. The crash PC symbolized to `jittor::plan_reduce_thread_order`; its failure receipt is retained as `continuous-execution-failure.json`. The frozen candidate check passed at HEAD `9e0e187bb013e71d85111acfd668e46aeff9f9d2`, 51 dirty entries, fingerprint `9816957e611838974709f567a1c1df10fc86125884906edd8e26db4b7036d519`. Qwen r67 Jittor stage replay PID 2803257 started at 06:09:50 +0800 on UUID `GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d` with a new JIT home; first core compile was at 118/228 at the 06:12 poll. The PyTorch stage completed and exactly reproduced the frozen capture output. Day4–5 r5 waiter PID 2616242 started at 02:09:02 +0800. At 06:12 +0800 it remained WAITING; the latest sample had 49,409 MiB free and 100% utilization on UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5`, with zero qualifying samples and no launch/measurement/completion receipt. No repository pytest or suite process is active. Disk availability is about 99 GiB on `/` and 6.0 TiB on `/data6`; no files were removed.

Filesystem space was about 100 GiB available on / and 6.0 TiB on /data6. No files were removed. Checkpoints, failure evidence, logs, caches, and worktree edits remain preserved.

## Day2–3

### T5 BF16

The existing r36 and r38 step-51 captures remain the newest localization evidence. r36 captured the requested dropout and encoder block 0 FFN boundaries: earlier dropout/attention events were exact, then encoder block 0 DenseReluDense.wi input differed at 1,449 elements with maximum absolute difference 1.430511474609375e-06. Its formal comparator remains FAIL because the pre-forward signature does not match retained r6. r38's dynamic trace found its first archived difference later, at encoder block 1 DenseReluDense.wi input (1,377 elements, max abs 3.5762786865234375e-07). The historical strict recovery failures remain 771/2088. No full100 was repeated.

r65 repeated the frozen encoder block 1 layer-norm input 32 times on actual CUDA with fallback_count=0. At para_opt_level 3 the reduction varied across 29 hashes and the output across 13; at level 4 all 32 repeats were bitwise stable. Level 4 still differed from the archived continuous and resume FFN wi inputs at 4,227 and 5,604 of 27,648 elements (max abs 9.5367431640625e-07), so this establishes repeatability, not trajectory alignment or a repair. The FP64 replay did not improve reference agreement. r66 attempted a new frozen step-51 continuous capture at para_opt_level 4, compile key 6604, on GPU UUID `GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d`. It exited by signal 11 in `jittor::plan_reduce_thread_order` after writing only an initial input/logits/loss probe (SHA256 `2608a604101809e8098bb9de9b9e3b6dfe53a16fff438dea7bf84e41f60b62a8`). No step-51 trace, checkpoint, device/fallback capture receipt, or resume prerequisite exists. This failed attempt does not change the historical T5 FAIL or justify retrying from a nonexistent checkpoint.

### Qwen BF16

Qwen r44 completed the original unhooked BF16 CUDA full-logits comparison from a fresh PT/JT run using the frozen input and source manifest. Both worker commands exited 0; the comparison receipt is FAIL_RETAINED and records full_training_launched=false. Both sides used actual cuda:0 with fallback_count=0. Inputs and loss met their unchanged comparisons, but logits failed: 7,015,339 of 29,171,712 values exceeded atol=0.005, rtol=0.03; maximum absolute difference was 2.25. On valid positions, 2,756,985 of 22,030,720 values exceeded budget and maximum absolute difference was 0.3125. The loss difference was 0.0038883686 and remained within budget. Only the logits key failed. The comparison SHA256 is d604c30238a3a6c27d2adf5a283ed38af05d0396599088dd3019d46cf60eb770; execution receipt SHA256 is eadd2e696d142817bc357b8249319f7c109f5de39ee7224e947fd3f97a34deee. No LoRA full/resume was launched.

r46 continued from r23/r26 on the frozen r22 tensor `[2,96,16,128]`, reducing the earliest mismatch to the CUDA mean accumulation order. In PyTorch 2.6's `Reduce.cuh`, this case uses 32 lanes, each accumulating four values at offsets `lane + {0,32,64,96}`; `block_x_reduce` then combines warp offsets 1, 2, 4, 8, and 16 before `MeanOps` divides by 128. A new explicit expression using that order matched native PyTorch exactly on all 3,072 outputs in both the independent PT process and Jittor shim process. Jittor's native `mean(-1)` retained 1,046 mismatches with max absolute difference `1.4901161193847656e-8`. Both processes used actual `cuda:0`, fallback_count=0, the same frozen input/manifest, and candidate HEAD/status identity `9e0e187bb013e71d85111acfd668e46aeff9f9d2` / `a061f1cd85aa91037f50720ce2a50a5211843f16feeed7809871ae724398c41e`. This is a reduction diagnostic only; it does not change the public implementation, prove full-model logits acceptance, or authorize LoRA. The r44 full-logits FAIL remains the release state.

r50 is the latest unhooked full-logits gate. With the unchanged `atol=0.005, rtol=0.03`, it remains FAIL: 6,690,070 of 29,171,712 logits are outside budget and maximum absolute difference is 1.953125. The original failure and comparison fields remain intact; no LoRA full/resume was launched.

r54 then compared the public RoPE module and a direct replay of the frozen Transformers source formula. Both captures reproduced their own frozen same-runtime input and logits exactly, used the same A800 UUID `GPU-ecb44f7d-870b-b9a4-a3d6-10301bae6068`, actual `cuda:0`, fallback_count=0, BF16, seed 821, and candidate HEAD/fingerprint `9e0e187bb013e71d85111acfd668e46aeff9f9d2` / `d8b57cb5da41cbbf1a59c1310cc1204d79a0b9394aa66b4f1774cdec0bd927dd`. The 113-capture comparison had no input mismatches and first differed at `rotary_emb/output/1` (sin): 162/12,288 values, max absolute difference `3.725290298461914e-9`. Direct replay found exact `position_ids`, one differing `inv_freq` value at index 37 (`0.0003398208355065435` PyTorch vs `0.00033982080640271306` Jittor), 81/6,144 differing `freqs`, and 162/12,288 differing `sin`; `cos` was exact. Layer 2's first BF16 main-path mismatch remained `self_attn.o_proj`: 59/196,608 values, max absolute difference `0.0009765625`.

A separate 64-value CUDA probe decomposed the default `1 / (base ** exponent)` initialization with `base=1,000,000`. Jittor's float32 `pow` and reciprocal matched host `math.pow` rounding for all 64 values; independent PyTorch differed from that host result at 4 power values and 3 reciprocals. This identifies a backend floating-point difference in the RoPE frequency initialization, not a proven Jittor correctness defect or a source repair. Replacing the expression with `exp(-x * log(base))` was not a general match. r54 applies no release thresholds and is diagnostic only: r50 remains FAIL, no source changed, and LoRA remains blocked until the original no-hook gate passes.

r55/r57 are retained harness-only failures: r55 computed its CUDA replay but interpreted `dispatch_context`'s `(backend, fallback_count)` as a per-tensor list; r57 stopped before model loading because the artifact directory was absent from `sys.path`. Neither changed candidate source or produced release evidence. Corrected r56 injected the frozen PyTorch `inv_freq` array into the Jittor replay; `freqs`, `cos`, and `sin` then matched the PyTorch arrays exactly on actual `cuda:0`, with fallback_count=0. This confirms the 1 ULP `inv_freq[37]` explains the tiny source-formula replay differences.

r58 then ran the frozen BF16 evaluation forward without hooks, replacing only the shared rotary buffer in process. The 85-file source identity and HEAD remained frozen, all inputs were exact, the model was on actual `cuda:0`, forward hook count was 0, and fallback_count=0. The resulting complete probe is byte-identical to r50's Jittor probe (SHA256 `9941776443c3fd77353d5c7f1ded5c94f1b179cab7ee18e4ad5603627df74100`): the unchanged `atol=0.005, rtol=0.03` gate still fails 6,690,070/29,171,712 logits, max abs 1.953125. The valid-attention diagnostic remains 2,472,467/22,030,720 values over budget, max abs 0.34375. The `inv_freq` discrepancy has no measurable effect on this fixture's BF16 logits and is not the cause of the large gate failure. No public source changed and no LoRA full/resume was launched.

r61 replayed the frozen r60 layer-0 attention-softmax input on actual CUDA in independent PyTorch and Jittor processes, with fallback_count=0 and unchanged 85-file source identity. Each runtime reproduced its own captured output exactly; the cross-runtime softmax outputs differ at 56,251/294,912 elements with max absolute difference `3.5762786865234375e-7`. This is a small local diagnostic and does not change the r50/r58 full-logits FAIL or permit LoRA.

r67 completed the staged softmax replay on actual `cuda:0`, with fallback_count=0 and unchanged 85-file source identity. The maximum, shifted logits, and exponentials were exact against the independent PyTorch capture. The first difference was the denominator: 1,290/3,072 values, max absolute difference `1.9073486328125e-6`; the primitive and fused outputs differed at 53,939/294,912 and 56,251/294,912 values respectively, both with max absolute difference `3.5762786865234375e-7`.

r68 replayed the reduction over the frozen exponentials. The explicit 32-lane accumulation (`i`, `i+32`, `i+64`, then pair offsets 1/2/4/8/16) was exact with PyTorch native sum on all 3,072 outputs in both runtimes. Jittor native sum remained different at 1,290/3,072 outputs, max absolute difference `1.9073486328125e-6`. This confirms the local denominator reduction order as the earliest observed divergence; it is not a source repair or full-model acceptance.

r69 materialized the frozen exponentials and replayed normalization in a new output directory, reusing the completed r67 JIT cache sequentially. It ran on the same A800 UUID `GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d`, actual `cuda:0`, fallback_count=0; the candidate stayed at HEAD `9e0e187bb013e71d85111acfd668e46aeff9f9d2` and the same 85-file fingerprint. Its native sum and native primitive exactly reproduced the r67 Jittor outputs. Its explicit tree sum matched PT native/tree sums exactly, and tree-normalized output matched PT's explicit primitive softmax exactly. PT primitive vs PT fused softmax differ at 53,939/294,912 values; the r69 tree-normalized vs PT fused comparison therefore compares distinct algorithms. The `native_fused_*` fields are invalid because the script applies `softmax` to already exponentiated values; the report and script remain preserved and are not treated as model evidence. This confirms that the local denominator reduction order explains the explicit primitive mismatch, while the remaining fused-kernel arithmetic is separate. None of r67–r69 changes the original r50/r58 full-logits FAIL or permits LoRA full/resume.

### Vision and BERT

The r18 missing `torch.__file__` wrapper failure remains historical entry-point evidence. r19 stopped in preflight with `STOPPED_PREFLIGHT_NO_NEW_LAUNCH` because a prerequisite source artifact differed; no model run was launched there. The corrected measurement entry was exercised by the new r25 cohort, which completed all five smoke/full/resume cases and artifact audits on real CUDA with fallback_count=0. cnn-fp32-821 passed same-runtime exact comparison but failed cross-runtime strict with 160 items; vit-fp32-821, cnn-bf16-821, and vit-bf16-821 failed same-runtime exact with 100, 245, and 143 items; cnn-fp32-823 passed same-runtime exact but failed cross-runtime strict with 177 items. Vision remains FAIL; all receipts, comparisons, checkpoints, and logs are retained.

BERT BF16 remains FAIL: the frozen two-seed task-loss gap is 15.425953905%, above the unchanged 15% limit. The older r1 cross-task checkpoint audit retains 11 unavailable original model/optimizer checkpoints (5 BERT, 3 Qwen, 3 T5). This is distinct from the retained r10 BERT/FP16 audit: its six independent-PyTorch checkpoints were already audited at 804 numeric arrays each and their migrated originals are currently reachable. The 13:09 recheck reused those unchanged audits and verified receipt hashes and array file sizes; no model was retrained. Seeds and comparison fields were not changed.

## Day4–5

The r3 protocol measured real AdamW updates with matched FP32 weights, inputs, device, optimizer, and synchronization. It recorded 10 hot samples after warmup, model-ready/first-step/steady timing, allocator allocated/reserved peaks, NVML process peaks, actual cuda:0, and fallback_count=0. Representative batch 8, sequence 256 had three alternating rounds with JT/PT median ratios 1.409, 1.510, and 1.341; p95 ratios were 1.398, 1.478, and 1.427. Near-capacity batch 64, sequence 512 had only two rounds, with median ratios 0.746 and 1.132 and p95 ratios 1.147 and 1.136. Representative allocator allocated peaks were 3,006,131,712 bytes for Jittor and 3,041,185,280 for PyTorch; reserved peaks were 3,322,937,344 and 3,282,042,880. Near-capacity allocated peaks were 21,715,963,392 and 21,845,145,088; reserved peaks were 22,012,755,968 and 22,368,223,232. These measurements are retained in day45-bert-release-r3.

r3 summary status is DIAGNOSTIC_SHARED_GPU. The repeated representative slowdown is a common-path regression signal, but shared load and the incomplete near-capacity round count prevent a release conclusion or a code-level attribution. No source change was made from these measurements. r4's bounded 1,800-second wait ended WAIT_TIMEOUT_NO_RELEASE_WINDOW without a measurement launch.

A new queue was prepared at /data6/cgq/jittor-lab/_state/release-day23/20261026-r45/day45-bert-release-r5. It has a separate performance cache and output root, binds repair HEAD 9e0e187bb013e71d85111acfd668e46aeff9f9d2, dirty source SHA 4f28ab7c26e2ede3239cbad0ae0336c92d492dcfcf87d792c342850314625cee, and weights SHA 86fa7113367fd0d231a76571ca1b73a1b24166af33558fee960636ce56337ed7. The waiter requires utilization <=10%, free memory >=60000 MiB, and five consecutive five-second samples, with a six-hour timeout. If admitted, it runs 10 samples after four warmups for both representative and legal near-capacity grids, with three serial PT/JT alternating rounds and first JIT serialized. At 02:34 +0800 the waiter remained WAITING after 120 observations; the latest target sample was 50,422 MiB free and 99% utilization, with no launch or completion receipt. The queue does not claim acceptance until measurements, summary, and memory/device audits complete; shared-load output remains diagnostic.

At the 06:12 +0800 live check, waiter PID 2616242 was still the same `wait_and_launch.py` process. Its latest sample had 49,409 MiB free, 100% utilization, and zero qualifying samples. No launch receipt or measurement child existed, so no new Day4–5 performance data were produced. The waiter remains active with its original six-hour deadline and unchanged source identity.

The common-path performance issue is not marked fixed. After a stable-window result confirms a representative slowdown above 20%, inspect the affected training path, make the narrow repair in the repair worktree, then rerun its impacted correctness gates and matched performance cell with a new frozen source identity.

## Artifacts

- Qwen r44: /data6/cgq/jittor-lab/_state/release-day23/20261026-r44/qwen-full-nohook-r1/
- Qwen r46 reduction tree: /data6/cgq/jittor-lab/_state/release-day23/20261026-r46/qwen-reduction-tree-r1/
- Qwen r50 full-logits gate and r54 RoPE diagnostic: /data6/cgq/jittor-lab/_state/release-day23/20261026-r50/qwen-full-nohook-r2/ and /data6/cgq/jittor-lab/_state/release-day23/20261026-r54/qwen-rope-replay-r1/; r54 comparison SHA256 `022b806e18f8bb0bbc9c70758b72ca228fd3eddfbb128a6e779281f328f08ff6`
- Qwen r55/r56 inverse-frequency counterfactuals: /data6/cgq/jittor-lab/_state/release-day23/20261026-r55/qwen-rope-invfreq-counterfactual-r1/ and /data6/cgq/jittor-lab/_state/release-day23/20261026-r56/qwen-rope-invfreq-counterfactual-r2/; r56 report SHA256 `54ea73ccdaae4337d4813db80cc2d8169aaf11aa3ec0c4b3f981a47f8a20c76b`
- Qwen r57/r58 unhooked full-logit counterfactuals: /data6/cgq/jittor-lab/_state/release-day23/20261026-r57/qwen-nohook-invfreq-counterfactual-r1/ and /data6/cgq/jittor-lab/_state/release-day23/20261026-r58/qwen-nohook-invfreq-counterfactual-r2/; r58 report SHA256 `0ecbfddae860640883d9549a74ce49f05e3ac0d6f18796c8a8675a91978d649f`
- Qwen r60/r61 attention-softmax capture/replay: /data6/cgq/jittor-lab/_state/release-day23/20261026-r60/qwen-attn-softmax-capture-r1/ and /data6/cgq/jittor-lab/_state/release-day23/20261026-r61/qwen-softmax-replay-r1/
- Qwen r67–r69 staged softmax, sum-tree, and materialized-output diagnostics: /data6/cgq/jittor-lab/_state/release-day23/20261026-r67/qwen-softmax-stages-r1/, /data6/cgq/jittor-lab/_state/release-day23/20261026-r68/qwen-softmax-sum-tree-r1/, and /data6/cgq/jittor-lab/_state/release-day23/20261026-r69/qwen-softmax-tree-output-r1/
- T5 r65 level-3/level-4 reduction A/B and r66 failed step-51 continuous attempt (including `continuous-execution-failure.json`): /data6/cgq/jittor-lab/_state/release-day23/20261026-r65/t5-norm-reduction-level-ab-r1/ and /data6/cgq/jittor-lab/_state/release-day23/20261026-r66/t5-step51-para4-r1/
- T5 r70 failed step-51 continuous attempt and retained failure receipt: /data6/cgq/jittor-lab/_state/release-day23/20261026-r70/t5-step51-para4-r2/
- Day4–5 r3 diagnostics: /data6/cgq/jittor-lab/_state/release-day23/20261020-r3/day45-bert-release-r3/
- Day4–5 r4 timed-out wait: /data6/cgq/jittor-lab/_state/release-day23/20261020-r4/day45-bert-release-r4/
- Day4–5 r5 wait and future measurement: /data6/cgq/jittor-lab/_state/release-day23/20261026-r45/day45-bert-release-r5/
- T5 and Vision evidence: see the 2026-10-25 report and original Day2–3 report

## 06:24 Live Refresh

The main worktree is `integrate/cgq-transformers-native-multicard-target@91b543c60e60fc3d19b2f7407c4a8f631385c47c` with 96 dirty/untracked paths. The original, repair, and Qwen diagnostic worktrees remain at `9e0e187bb013e71d85111acfd668e46aeff9f9d2` with 51, 83, and 85 dirty paths. A fresh `git fetch origin cgq_transformers` recorded `FETCH_HEAD=e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`; that SHA is an ancestor of all four HEADs, so no integration was needed. No stash/reset/clean or `origin/2.0-refactor` integration occurred. `/` has 98 GiB available and `/data6` 6.0 TiB; no files were removed.

Qwen r67–r69 are complete diagnostics. Their CUDA stage reports and the r69 harness caveat are recorded above. Offline comparison of retained arrays confirms r69's native primitive exactly matches r67 Jittor primitive, while its tree-normalized result exactly matches PyTorch's explicit primitive (both have 53,939 differences from PyTorch fused softmax). No Qwen full-LoRA process was launched. T5 r66 remains an incomplete signal-11 attempt with no step-51 trace or checkpoint; do not launch resume from it. Vision r25 and the original BERT BF16 gate remain FAIL, including the unchanged 15.425953905% two-seed gap against the 15% limit.

The only task-related live process is r5 waiter PID 2616242, verified from `/proc` and its command line as the `wait_and_launch.py` started at 2026-10-26 02:09:02 +0800. Its latest retained sample at 06:24 has 49,409 MiB free and 100% utilization on target UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5`; the qualifying streak is zero and no launch, measurement, or completion receipt exists. The six-hour waiter remains active to its 08:09 deadline. The r67/r68 JIT cache has no active holder; the r69 process exited. All GPUs currently show 99–100% utilization with 23.8–58.9 GiB free. No repository test process is active. Day4–5 remains unaccepted; r3 is still shared-load diagnostic only.

## 06:47 Live Refresh

The main worktree remains `integrate/cgq-transformers-native-multicard-target@91b543c60e60fc3d19b2f7407c4a8f631385c47c` with 96 dirty/untracked paths. The original, repair, and Qwen diagnostic worktrees remain on branch heads rooted at `9e0e187bb013e71d85111acfd668e46aeff9f9d2` with 51, 83, and 85 dirty paths. `origin/cgq_transformers` was fetched again at `e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`, an ancestor of all four worktree HEADs. No stash, reset, clean, branch integration, or file cleanup occurred. `/` has 98 GiB available and `/data6` has 6.0 TiB. No repository test process is active.

T5 r70 continuous capture remains live as PID 2815893 (started 06:28:59 +0800) with child PID 2816187 (started 06:29:10 +0800), using the frozen original Day2–3 worktree at HEAD `9e0e187bb013e71d85111acfd668e46aeff9f9d2`, 51 dirty entries, fingerprint `9816957e611838974709f567a1c1df10fc86125884906edd8e26db4b7036d519`. `freeze_candidate.py verify` returned `FROZEN_SOURCE_PASS`. Its JIT cache is isolated at `/data6/cgq/jittor-lab/_state/release-day23/20261026-r70/t5-step51-para4-r2/jittor-home/`; the only lock holders are PIDs 2815893 and 2816187. The run generated `continuous-output/initial-probe.npz`, SHA256 `2608a604101809e8098bb9de9b9e3b6dfe53a16fff438dea7bf84e41f60b62a8`, identical to r66's probe. It is still compiling the first float32 `setitem` CUDA kernel on UUID `GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d`; at 06:47 that GPU showed 55,677 MiB free and 100% utilization. `continuous-capture/status.json` remains `started`; no step-51 trace, checkpoint, or completion receipt exists. Do not start a duplicate continuous run or resume without both required artifacts.

The r5 Day4–5 waiter is PID 2616242, still the original `wait_and_launch.py` started 02:09:02 +0800. Its latest sample at 06:47:20 +0800 is 49,409 MiB free and 100% utilization on UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5`; its gate still requires utilization <=10%, free memory >=60,000 MiB, and five consecutive five-second samples. The qualifying count is 0/5, and there is no launch or terminal receipt. Its six-hour deadline is 08:09:02 +0800; leave the waiter alive and do not alter its frozen repair worktree.

No Day2–3 gate changed: Qwen r50/r58 full logits remain FAIL and LoRA full/resume remains blocked; T5 r66 signal-11 failure remains retained and r70 has not reached step 51; Vision r18 failure remains retained and r25 remains FAIL; BERT BF16 stays FAIL at 15.425953905% versus the unchanged 15% limit. Day4–5 r3 is still diagnostic shared-GPU evidence only; r5 has not launched. No release acceptance is claimed.

## 06:49 Terminal Refresh

T5 r70 exited with signal 11 before step 51. The execution session returned exit code 1. Its requested GPU UUID was `GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d`, and `nvidia-smi` observed PID 2815893 on that GPU during the run. Candidate freeze verification passed at HEAD `9e0e187bb013e71d85111acfd668e46aeff9f9d2`, 51 dirty entries, fingerprint `9816957e611838974709f567a1c1df10fc86125884906edd8e26db4b7036d519`. `addr2line` resolved `jittor_core.so+0x4c47e0` to `jittor::plan_reduce_thread_order(...)`; line information is unavailable (`??:?`). This is the same reported function as r66, but no more specific cause is established. The process was last seen with a live JIT compiler and exited before producing a trace; no checkpoint, step51 capture, fallback receipt, or completion receipt exists. The existing capture `status.json` remains `started` because the crash path did not finalize it.

r70 wrote only `continuous-output/initial-probe.npz`, SHA256 `2608a604101809e8098bb9de9b9e3b6dfe53a16fff438dea7bf84e41f60b62a8`, matching r66. The cache remains at the r70 run path and has no active lock holder. The manually recorded `continuous-execution-failure.json` captures the observed process identity, signal, symbolized function, and missing artifacts; it is not a model-result receipt. Do not start resume or repeat the same capture before diagnosing this crash path.

At 06:49:08 +0800 r5 remained alive as PID 2616242, WAITING with 0/5 qualifying samples; its latest target sample remained 49,409 MiB free and 100% utilization, with no `wait_receipt.json`, launch receipt, or measurement child. `/` has 98 GiB available and `/data6` 6.0 TiB; there is no storage-driven cleanup need. No task-related train/test process remains except the r5 waiter. Day2–3 gates retain their failures, and Day4–5 remains unaccepted.

## 07:09 Verification Refresh

`bash tools/check_repo_layout.sh` passed (`documentation governance OK: 246 active Markdown files`; `repository layout OK`). The required Torch-mode structure command completed with `JITTOR-SESSION-COMPLETE collected=1400 executed=1398`, `1396 passed, 4 skipped in 650.77s`. The four skips were reported as two cases with no Jittor installation in the nested interpreter and two cases requiring unavailable `pytest-xdist`; no failures were reported. The first attempt without a runtime override stopped during pytest configuration because Anaconda's `libstdc++` lacked `GLIBCXX_3.4.30`; the successful rerun preloaded `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` and used the isolated JIT/TMP root `/data6/cgq/jittor-lab/_state/torch-compat-acceptance/20261026-structure-r1/`. These repository gates do not change any Day2–5 release result.

At 07:09:34 +0800 the r5 waiter was still PID 2616242, started at 02:09:02 +0800. Its latest target sample remained 49,409 MiB free, 100% utilization and 0/5 qualifying samples on UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5`; there was no `wait_receipt.json`, launch receipt, or measurement child. The six-hour deadline remains 08:09:02 +0800. The structure pytest process has exited; no task test process remains. Day4–5 remains unaccepted and all Day2–3 failures retain their original thresholds.

## 13:05 Continuation: r5 Terminal State and Remote Sync

r5 `wait_receipt.json` records `WAIT_TIMEOUT_NO_RELEASE_WINDOW`, ending at 2026-10-26 08:09:08 +0800 after 4,024 retained observations. There is no launch, measurement, or completion receipt and the old waiter PID is absent; the earlier WAITING entries are historical snapshots. No old waiter was restarted. A fresh fetch records `origin/cgq_transformers@e5cffe1ae60c418f53c5bb1cdca9182577fd2b81`, an ancestor of all four worktree HEADs. Main HEAD is `395949d7239764baa0a614a3ac8b2f204d6e41bb` (91 dirty/untracked paths); original/repair/Qwen trees remain `9e0e187bb013e71d85111acfd668e46aeff9f9d2` with 51/83/85 paths. No merge, stash, reset, clean, or cleanup occurred. The pre-fetch and synchronization hashes are retained under the unversioned `20261026-continuation-1257/` artifact directory. New work targets a reliable T5 reduction-crash reproduction and independent Qwen/Vision localization; all original release failures remain open.

## 13:13 Verify-Then-Fix Progress

A direct minimal call into the retained production `plan_reduce_thread_order` reproduced SIGSEGV at core offset `0x4c47e0`, identical to the symbolized r70 offset. A valid-range control returned `[0, 1, 0]`; changing only `tn0` to `get_thread_range_log(thread_num_left, 1)` crashed after `EMPTY_START`. This verifies the empty-dimension access in the real core, rather than inferring causality from generated text alone. A narrow fix now treats eliminated ranges as neutral and skips shared rewriting when no reduced thread range remains; CUDA regression tests are prepared. The original tree/cache and r66/r70 failures remain unchanged. Evidence and source transition are retained under `20261026-t5-reduce-repro-1315/`; fixed-core and fresh step51 verification are still required.

Two new Vision fresh-process checkpoint61-to67 captures find only three raw gradient tensors different at step62 (cls token, position embedding, and patch-projection bias), before scalar norm diverges. The corresponding fixed-upstream public/native broadcast backward paths each produce 32 distinct hashes in 32 actual-CUDA repeats at the default optimizer level; replaying the same frozen raw gradients through the norm helper is stable. This proves a matching generic reduction repeatability defect, but does not close either original Vision recovery or cross-runtime gate. Raw traces, generated atomic kernel hashes and limits are retained in `20261026-vision-1305/HANDOFF.md`.

Qwen has a new isolated counterfactual: a fixed-warp fused-softmax arithmetic replay matches all 294,912 frozen probabilities, and replacing only that operation in-process yields exact agreement on all 29,171,712 full unhooked logits at the unchanged gate. This establishes a causal arithmetic path; it is not yet public-production acceptance. The separate Qwen tree is validating a general short-row FP32 CUDA softmax repair and its gradient regressions.

The performance audit confirms r3 live allocator peaks cover only 25.28%/25.43% of the 80-GiB device, so its near-capacity label is unsupported. A new external protocol candidate corrects first-step placement before warmup and freezes source contents plus HEAD; 13 contract checks and an independent-PyTorch CUDA update smoke pass. Jittor and paired/stable-window acceptance remain unverified. See `20261026-perf-1311/HANDOFF.md`; no old waiter was restarted.

## 13:31 Public Qwen Forward and Fixed Planner Regression

The public short-row FP32 CUDA softmax repair is now verified in the separate Qwen worktree (87 dirty source paths, fingerprint `441ddd71f35d22da9826fdfcdb699110e3087eae8b621562411a1049e3a5c342`). The original no-patch/no-hook worker and unchanged comparator pass: all 29,171,712 logits are exact, loss difference is 0.0003712177276611328 within the original budget, inputs are exact, actual device is cuda:0 and fallback_count=0. The comparison SHA256 is `487c70cdc94bb64b4566949c103a6a14f6bfe5760347886500163520f4e8a55e`. Seven CPU/real-CUDA forward/backward, independent-PyTorch, large/log and FP16 regressions pass without skips; the two added tests are included in that seven. Only this forward prerequisite is closed. The new LoRA two-update smoke has started cold compilation; its first cache-bootstrap rc3 is retained and a new bounded output label is used for the documented reload retry. No LoRA full/resume pass exists yet. The validated two-file source delta was copied into the main tree only after checking each main file was byte-identical to its Qwen base; integration hashes are in `20261026-continuation-1257/softmax-main-integration.json`. The main tree as a whole has not been release-accepted.

The fixed planner now passes all nine CUDA regression tests with actual use_cuda=1, fallback_count=0 and unchanged 85-path repair-source fingerprint `7a4414bc8e7e71dff1ccaa83a1c46f7b50a27a7a95b121a152f9ba2dfc5d906e`. The rebuilt core SHA256 is `566f196a958503133ab249cb2c0bec1816d020ad78c6ab5aadac87de0702f997`. The first run retained 8 passed/1 failed: the new synthetic IR test incorrectly declared thread_num instead of emitting ParallelPass's assignment anchor. Correcting only that test construction gives 9 passed/0 skipped in 2.91 seconds; mixed empty/real ranges exercise apply as well as plan, all-eliminated ranges leave IR unchanged, and real singleton reductions are checked numerically. Logs and both source snapshots remain under `20261026-t5-reduce-fix-1310/`. This validates the compiler repair, not T5 training recovery.

Vision fixed-upstream A/B and independent PyTorch are complete on the same actual CUDA device: default level3 produces 32 hashes in each of three shapes; fixed level4 and independent PT each produce one hash. Level4 gradient max absolute deviations from PT are 1.192e-7, 8.941e-8 and 2.384e-7, with no original FP32 short-gradient budget violations. Frozen norm replay is exact and stable at both levels. Source drift and fallback audits pass. These diagnostics remain separate from original Vision acceptance; no global default level or public deterministic policy was changed. See `20261026-vision-para-ab-1316/HANDOFF.md`.

A further Day4–5 tool audit found no usable r3 per-kernel timing trace, repeated PT-then-JT order rather than alternating order, and Jittor-only memory profiling inside the timed samples. No slowdown attribution or performance fix is claimed. New external v2 protocol tooling passes 26 positive/negative contracts and a real-PyTorch CUDA smoke with complete device/UUID/update/source guards; its Jittor branch remains unexecuted. A separate unexecuted v3 candidate separates unprofiled hot timing, memory observation and attribution profiling. These tool improvements do not turn old diagnostics into release measurements.

## 14:03 LoRA Smoke Closure and New T5 Pair

Qwen `lora-campaign/821-smoke-r2.execution.json` has rc0 and `smoke-r2-status.json` is PASS. The unchanged 102-update schedule was stopped after exactly two successful optimizer updates, with zero skipped updates, actual cuda:0, fallback_count=0 and unchanged 87-path source fingerprint. All 112 adapter tensors (573,440 elements) changed from their frozen initial values and remain finite; max delta is 5.00678334e-6. Adapter save/reload, greedy generation and cache checks are exact; merge/reload max_abs 0.0002942085 and global relative 9.46e-6 satisfy the original 0.001/5e-5 limits, with exact generation and zero loss delta. Report SHA256 is `57a0f0491a5794d95f96615920cc668d8f645a1cb27a392e3f4103fe93925de6`. The original rc3 cache-bootstrap failure remains intact. This is bounded workflow/lifecycle evidence; full102, fresh resume51 and the original task-metric comparison remain pending.

The original gradient-checkpointing configuration was retained. The public shim emitted its explicit warning that it runs the wrapped function directly and keeps activations: actual activation recomputation and memory saving are not established. The release requirement for at least one gradient-checkpointing cell remains open. No model/operator patch hid this limitation.

T5 fixed-planner continuous capture started at 14:01:40 +0800 in `20261026-t5-step51-fixed-1320/`; PID 3077840/start_ticks 160371398 was verified live, using actual CUDA on assigned UUID GPU-a49e1cb8-b822-0e7c-f399-7ed6c935cc3d. Its 85-path repair-source manifest remains frozen. This is a bounded step51 continuous/fresh-process resume diagnostic, not a full100 retry; the controller requires a passed continuous capture plus valid checkpoint50 before launching resume. The retained historical comparator and a new-cohort-only all-array/event comparator are separate; original strict 771/2088 failures are unchanged.

Main integration now includes the new WarpSoftmax backward reference row. Before correction, the AST inventory reproducibly found only that new gradient unlisted; after correction, six targeted inventory contracts pass. The route points to the real CUDA/FP64-NumPy short-row forward/backward regression. The main CPU/core and affected pytest checks are prepared, not executed; two isolated copies of the old 103MiB warm structure cache preserve the original bound cache. Main has a different HEAD and runtime implementation from the frozen Qwen candidate (including missing candidate-only rsqrt header/workflow runner); isolated Qwen evidence cannot be transferred to the entire main tree.

The BERT r10 offline audit confirms both seeds' 198 microbatch sample cursors and 100 learning-rate values match. The original selected-best loss gap still fails at 15.425953905%; PT seed823 retained checkpoint50 after an accuracy tie while JT selected checkpoint100. Same-update loss comparisons are diagnostic only and do not replace the original fields. A frozen original-input real-CUDA forward/all-gradient probe is prepared but unexecuted. Vision now has an unapplied native broadcast-reduction design with source bindings; no production deterministic repair or original-model recovery pass is claimed.

## 2026-10-29 task-scoped compatibility evidence

The HEAD-only r12 candidate is based on 3dedae001372b6be106ae6700bcf10ebc7faebac.
Its patch SHA-256 is
250c43dffca948b5e027ea20f65b0e4141ecffc54b039f410398f6f6b69718c5;
the r12 delta from r11 adds a persistent subprocess test for CUDA enabled
before import. The main df45 CUDA run passed that node, and a separate run
passed the same r12 test and helpers on the actual r10 CUDA core and runtime,
with setup/call/teardown passed and no skips. The r10 native compile order has
device.cc before fetch_op.cc. This is mixed-source evidence, not a complete r12
cold build; the child asserts unchanged fallback, not a zero initial count.
The r10 CUDA candidate separately passed its pre-import and GPU functional
stages. These are distinct builds and results.

The r11 CPU scheduler focused run passed 16 selected nodes using the actual
r10 native core and r11 scheduler/helper overlay, with use_mkl=0. A separate
CPU tiny BERT run used that same mixed identity: eight updates completed and a
new process resumed updates 5–8 from step 4, with the final 25 parameters
matching continuous training. This was a synthetic, dropout-free fixture, not
a GPU or long-training result. Its fixed evaluation loss increased from
0.696845 to 0.767809 and accuracy fell from 0.25 to 0.125; no convergence
claim follows. Retained earlier failures and the original release thresholds
are not replaced by these focused results.

Main source df45, a separate identity from the HEAD-only candidate, completed
a fresh single-seed 821 BERT CUDA two-epoch run configured for BF16 with 100 updates, 198
microbatches, finite 201-parameter gradients and fallback 0. Its epoch mean
microbatch losses were 0.354515 and 0.208091 (historical independent PyTorch
0.354172 and 0.196566); fixed evaluation loss/accuracy were 0.477709/0.787879
versus PyTorch 0.483606/0.787879. This single-seed result does not replace the
original two-seed selected-best loss-gap failure of 15.425953905% versus 15%.

A fresh process restored df45 checkpoint 50 including model, optimizer,
scheduler and RNG state, verified 603 named CUDA tensors at restoration, and
completed updates 51–100 with the same sample and learning-rate sequence,
finite gradients and fallback 0. Its continuation mean loss was 0.207975
versus 0.208091 for the continuous run; evaluation was 0.478523/0.787879
versus 0.477709/0.787879. Strict final state equality failed: all 201 model,
201 first-moment and 201 second-moment tensors had differing hashes, whereas
all 201 step scalars matched. The offline numerical comparison reported global
relative L2 differences of 4.46e-5, 0.0505 and 0.00580 respectively. No cause
is established; numerical allclose is diagnostic rather than an acceptance
threshold. These observed results demonstrate bounded functional continuation,
not identical trajectories or a performance claim.

Independent terminal audits and their SHA-256 values are retained at:
- /data6/cgq/jittor-lab/_state/release-day23/20261029-main-unified-cuda-focused-r1-hold/independent-terminal-audit.json:
  3585f23ab5330b3a39960dfe5590ecf7f02e55dc863218abaa1e79217b77276f;
- /data6/cgq/jittor-lab/_state/release-day23/20261029-headonly-r10-fetch-queue-candidate-r1-hold/cuda-cold-hold/gpu-functional-independent-terminal-audit.json:
  4b30b1babe55b273519b6d9f1f7e172e303788a8397a502db185627cd9f6d590;
- /data6/cgq/jittor-lab/_state/release-day23/20261029-headonly-r11-lambda-cpu-focused-terminal-audit-r1/terminal-audit.json:
  efea8cd356d3aa8d73a25bbd0a8fb34d1769d68cd6f30b9d07dc4a2cf72f3f2e;
- /data6/cgq/jittor-lab/_state/release-day23/20261029-headonly-r11-tiny-bert-cpu-model-terminal-audit-r1/terminal-audit.json:
  9118537ee24255fe75c3756e4f2070446f129a38cf843c633c6f66decb890239.
- /data6/cgq/jittor-lab/_state/release-day23/20261029-headonly-r12-startup-cuda-node-terminal-audit-r1/terminal-audit.json:
  c3cebb96e05632e97e05951a3d7a0ef13c14c24eea8b94c23ba0de7ad69ae3f1;
- /data6/cgq/jittor-lab/_state/release-day23/20261029-main-df45-bert-bf16-learning-trend-plan-r1-hold/independent-terminal-audit.json:
  96f990e7895121b37db7e5b15b728c97839a88883517203e0f129bb1f4821265;
- /data6/cgq/jittor-lab/_state/release-day23/20261029-main-df45-bert-bf16-freshresume-plan-r1-hold/independent-terminal-audit.json:
  f3edc6ba4b820d0d163d70b7f99b37552188211fafaac64c6961e1cc32a2729f.

## 2026-10-30 Bidirectional cross-CUDA `Tensor.to` regression

Added a parameterized regression for both CUDA ordinal directions, cuda:0→cuda:1 and cuda:1→cuda:0, complementing the previously tested cuda:1→cuda:0 path. Both cases executed and passed (2 passed, 0 skipped). The test checks CUDA FP32 values, the first-order gradient [4, 6], source/target/gradient placement, and an unchanged zero backend fallback count. The worker owned GPU UUIDs `GPU-2be51dc7-d99a-7307-869b-30ed5888e701` and `GPU-b6f59824-d1a3-cd6f-8100-9149728fc8eb`.

The run was bound to main-tree HEAD `387a096ef85f44dd9d097bfaf78d5a36176abff5`, 2,815 files, source fingerprint `95eb54ba17b771ece4ed585c2036f19565075b66c858e599972fbee011f76558`, and test SHA-256 `0cf34fae4fb8399340b6560be962e61748aecd7de8c3f11c4c47d3e5b2294ca7`. It reused the R7 dual-GPU cold-built core (SHA-256 `e65333a9c7cf11d9865bc8ae8ba415afab130014b47e3592b2c9c14edf735b3c`). The worker exited naturally with code 0 in 14.642 seconds; pytest reported 2 passed in 8.78s. Independent terminal audit passed at `/data6/cgq/jittor-lab/_state/release-day23/20261030-torch-cross-cuda-bidirectional-r2-hold/root-terminal-audit.json`, SHA-256 `76721c11cb6f6b89f40d291248ebd0b06ce5e18e0b75a42103172d3c64393ad8`. Controller receipt SHA-256: `07cb30c47e870098b573beafda70336b9261c80247372dcfe34373f9eecf82c7`; worker result SHA-256: `8410601cefeae69cece6089c1a47a3dd25b2c59a34b2b8b8a189ff5ab15519e4`; raw log SHA-256: `1490d91288f4018d485002bd67c11c2273604d7ca085a14296bf788725cbc151`.

The candidate cache grew from 980 to 988 files, with 8 additions, no removals, and only the expected `jittor.lock` modification; protected caches were unchanged and the execution processes and lock were clear at terminal audit. The first isolated-worktree static review stopped before pytest because that worktree omitted 79 dirty source overlays present in the main tree. No test ran in that attempt; the hold package remains under `20261030-torch-cross-cuda-bidirectional-r1-hold`. The successful retry was bound to the full main-tree snapshot.

This result covers only a tiny FP32 forward copy, one reverse gradient, and placement checks in both ordinal directions. Shared GPU utilization was high, so no performance claim is made. Other sizes and dtypes, streams, CUDA graph capture, and training remain untested. The remaining Day2–5 scope and Day4–5 performance/memory gates remain open.

## 2026-10-30 Day4–5 window probe

Five read-only samples at approximately five-second intervals from 11:21:28 to 11:21:49 UTC showed the designated UUID `GPU-2a503ab8-d84d-af1c-62aa-46f15c7977e5` at 100% utilization in all samples, with 17,993–18,711 MiB free. No timing workload was launched. This is a short readiness snapshot only; the formal paired performance and memory gate remains open.

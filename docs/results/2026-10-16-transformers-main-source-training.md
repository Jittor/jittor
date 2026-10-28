# Transformers：主线整合后的完整来源训练复验

- 状态：独立主线整合测量 CPU ① / CUDA ③完整范围通过。
- 日期：2026-10-16（10月15日启动，跨日完成）。
- 实际测量源码：`09e9e8eb8f67f4b32e3b46e05174013c1216983f`，全部正式进程同一 clean 提交。
- 开工同步：fetch 得 `593249c5f0e82febb931234aa35003d0d41c8b33`，整合已包含该远端；冻结提交仍为 `09e9e8eb8`。
- 维护者：Jittor Torch compatibility maintainers。
- 复查条件：框架、compat、模型/runner/training probe、来源采集/比较器、上游/wheel或实际 provider 变化。

本报告是[原 B4 完整来源矩阵](2026-10-15-transformers-installed-source-training.md)整合进主线后的
**新独立测量**，不是给旧原始产物更换提交号。与旧 B4 的 1562 个运行源码相比，本次实际绑定
1571 个文件：11 个内容变化、9 个新增。严格评分器拒绝旧源码产物直接冒充当前树是正确行为；
原工作树、原 B4/v5 数值和全部来源身份保留不动。

## 环境与实际执行

从指定整合提交创建独立 worktree、新 candidate Python prefix 和空 CPU/CUDA benchmark cache，
未复制旧 JIT 二进制。首次编译串行；全部 controller、worker、HOME/JITTOR_HOME/TMPDIR/XDG
及原始产物均在外置 lab state。已封官方 Torch oracle/provider、Transformers wheel 安装只读复用。
本次没有修改计算实现、case、共享 runner/training probe、collector/评分器或数值预算。

固定 Transformers 4.56.2 官方源码 `cd74917ffc3e8f84e4a886052c5ab32b7ac623cc`，
官方 wheel SHA256 `79c03d0e85b26cb573c109ff9eafa96f3c8d4febfd8a0774e8bba32702dd6dde`。
每个正式 worker 核验完整2173个实际安装文件、官方 METADATA/RECORD，以及1571个框架、compat与选定工具源码文件。
独立 oracle 是完整官方 Torch `2.11.0+cu128` / NumPy1.26.4；完整11842个 wheel payload 在运行前后
全量重哈希，真实二进制 `_C` SHA256仍为 `981997eb5ab8ce1f39bdad04b1d8150e4787f5e144b47bc343eb193aea1f740d`。

CPU双方显式关闭CUDA，候选oneDNN/MPI/NCCL关闭；CUDA全部80个正式进程使用同一张RTX4090，
校验实际输入/输出/参数/全部请求梯度驻留、driver指针属性与物理UUID。
候选NVCC12.2.140/GCC11.5.0，实际映射runtime API12020；oracle为12080，分别保留真实cu122/cu128
库路径、哈希和NVIDIA发行metadata，未重标provider。CUDA双方保留原TF32开启策略。
本次单进程/单设备，不声明MPI、jtrun、NCCL或多rank能力。

## 完整范围与原预算

保留全部十个case：BERT、CLIP双塔、GPT-2、Llama、RoPE2048、Mixtral四专家/top-2、T5、ViT、
Whisper和真实32768token Longformer。每case/设备的base0、输入扰动7/19、三步训练均有独立
真实Torch与候选进程，共**160个正式worker、7854组数组比较**。三步手动SGD、lr0.001、
step输入seed1/2/3、完整初始权重、全部参数及浮点输入梯度、loss和参数轨迹保持原样。
Whisper保留base/扰动89参数梯度和训练88可训练+1冻结；CLIP/ViT像素梯度、Whisper特征梯度、
全部MoE专家/路由和Longformer `[1,32768,32]` 噪声输入梯度均检查。

TF9沿用原每阶段全场参考最大量级：base两设备0.002/0.01、CPU扰动/训练0.002/0.01、
CUDA扰动/训练0.005/0.02，绝对或全阶段归一化任一通过。Longformer沿用原每数组/梯度块floor
归一化及严格小于规则，CPU0.002/0.01、CUDA0.005/0.02；不能用TF9的大输出量级掩盖其小梯度。

| 设备 | cases / 正式worker / 数组对 | TF9最大原全阶段归一化误差 | Longformer最大原数组/梯度块归一化误差 | 候选fallback |
| --- | ---: | ---: | ---: | --- |
| cpu | 10 / 80 / 3927 | 1.79564466e-06 | 9.84836796e-06 | error / 0 |
| cuda | 10 / 80 / 3927 | 0.011524663 | 2.73647853e-05 | error / 0 |

CUDA T5最差在 `perturb19` 的 `grad::decoder.block.0.layer.2.DenseReluDense.wi.weight`，
最大绝对误差0.383440316，阶段归一化0.011524663，
保留原2e-2该阶段预算，不声明逐元素相对精度或严格IEEE。CPU/CUDA Longformer最大绝对误差
分别为0.0166015625 /
0.0439453125。

两次公开维护nox均终态rc0，CPU/CUDA分别约33.58/
39.27分钟（包含各形状首次JIT；独立base预检另计），仅作诊断，不授L4。
独立collector和统一评分器：**20 accepted、0 blocked、0 rejected、7854数组对**。
另纯host核对双方特殊场景64份实际阶段/运行时case，以及20套完整CPU/CUDA初始权重共870个数组字节一致。
覆盖轴保持9/374模型族、2/46task mappings @固定上游，两个Llama场景只计一个模型族；来源补证未增加模型能力。

## 复现与证据

`TF_MAIN_STATE=$JITTOR_LAB_ROOT/_state/transformers-main-validation/09e9e8eb8`（未版本化）。
实际命令、完整环境入口`run-gate-v1.sh`和串行执行记录`complete-execution-v1.json`、
`formal-v1/{cpu,cuda}`、独立评分、源码差异和whole-Git快照、完整oracle/provider复核与最终
`evidence-manifest-v1.json`均位于该state。原始产物、安装包、编译缓存不进入仓库。

公开入口仍为 `nox -s ecosystem_transformers_source`；配置候选/真Torch解释器、固定TF checkout/wheel，
先CPU后显式单张CUDA及真实NVCC，传入新的外置 `--output` 和私有 `--cache-root`。
不得缩小case/phase，维护入口自动做数值判定，worker rc0不足以通过。
独立只读复算（无需GPU/JIT）使用state的`recheck-v1.sh <new-empty-audit-output>`，
基线必须仍为完整`09e9e8eb8f67f4b32e3b46e05174013c1216983f`，不得改为后续报告提交。

本次没有新增模型家族或adapter，不授Trainer、生成、checkpoint/resume、其他optimizer、BF16、
完整预训练尺寸、多rank、NPU/ROCm或性能能力；不把普通生态pytest节点和完整nox门禁混为一谈，
不声明scheduled nightly已经选择该新session。

## 提交前检查

TF工具的12项host契约通过（1.185s）；仓库布局与diff检查通过。规定Torch模式结构检查使用独立
单测HOME/JITTOR_HOME/TMPDIR和全新CPU缓存，1506通过、6跳过、2失败（270.55s）。
两项失败分别是新报告尚未进入results toctree，以及MANIFEST.in未登记此文件；中央登记属于整合步骤，
不改计算预算或隐藏失败。6项跳过包括2项加速硬件、2项声明类别，以及2项source-only候选环境未安装
Jittor发行metadata的包解析检查；真实源码导入与CPU/CUDA计算由完整维护门禁分别证明。
原始日志为state的`unit-structure-v1.log`、`unit-execution-v1.json`及`host-tools-v1.log`。

## 中央整合复核

整合者在已发布 `0f2f9481cd59072cc02a1edff4bc16bc12d1b071` 上仅接入本报告与结果索引、
生成的 MANIFEST，未改测量代码。独立重读1106份原始/辅助、17197个外部文件、922个实际
编译二进制，以及2865个原测量 Git 文件，哈希全部匹配；原封存SHA256为
`125c2ffecefacb94680981424908d433a3ab5397f698d4ddb4deaf264d26ff06`。
源码symlink按Git保存的目标文本核验，编译二进制未复制或重标。

登记后主树完整 `tests/structure + compat/tests/structure` 为1622通过、6跳过，211.12秒，
布局与空白检查通过。原独立树两项登记失败保留原始日志，不追改成通过；上面的全部
数值仍对应09e9，不能外推至本次报告提交。中央复核记录在未版本化的
`$JITTOR_LAB_ROOT/_state/transformers-main-source-doc-integration-0f2/`。

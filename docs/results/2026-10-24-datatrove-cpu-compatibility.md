# DataTrove CPU 数据集与本地管线兼容验证

- Status: 三个固定 Linux CPU 场景的功能对拍通过
- Date: 2026-10-24
- Baseline commit: `ca56aa802c2acf9a22bca32d1bee3276381ed203` 加本 PR 的测试与文档
- Owner: yzhouli
- Review when: DataTrove dataset/executor、Torch shim 发布或 Jittor 构建入口改变

## 提交范围

DataTrove `v0.10.0`，上游 `https://github.com/huggingface/datatrove.git`，提交
`7024aecca2f9ffb7b7cf0d02c0c823b8b24cf664`。
本 PR 接入 CPU 数据集、本地 JSONL 管线和本地多 worker 执行器的回归测试与复现说明。
普通管线直接运行，dataset 使用现有 Torch 兼容层；没有新增专属 adapter、核心或后端改动。

## 环境与身份校验

本次通过 Slurm 在 Linux x86_64 计算节点执行，作业 `119913`，仅使用 CPU。
Python 3.11.13、GCC 11.5.0、NumPy 2.4.6、DataTrove 0.10.0、multiprocess 0.70.19。
独立 oracle 为 PyTorch `2.8.0+cu128`，核验了实际二进制 `torch._C`，没有 Jittor shim
安装标记。Jittor 侧检查独立 Torch frontend 已激活、`flags.use_cuda == 0`。

两个运行时使用相同安装的 DataTrove，全部 **126 个 Python 源文件**与固定上游提交
的源码比对无差异。环境位于作业独立目录；共享现有依赖文件时不执行旧 editable `.pth`。

## 固定输入与实测结果

| 场景 | 输入规格与断言 | Jittor | 独立 PyTorch |
| --- | --- | --- | --- |
| `DatatroveFileDataset` | uint16 `[10,11,2,20,21,30,2,40,41,42]`，seq_len=4，EOS=2；两个窗口的 token、position 精确相等，输出均为 int64 CPU Tensor | 通过 | 通过 |
| JSONL reader/filter/writer | 5 条文档，保留偶数 ID；输出恰为 `0,2,4` | 通过 | 通过 |
| LocalPipelineExecutor | 8 条文档，tasks=4，workers=2，forkserver；输出 `0,2,4,6`，total=8、forwarded=4、dropped=4，4 个 completion 文件 | 通过 | 通过 |
| 合计 | 同一测试文件与固定预期值 | **3/3** | **3/3** |

两个 `input_ids` 为 `[[10,11,2,20,21],[30,2,40,41,42]]`，positions 为
`[[0,1,2,0,1],[0,1,0,1,2]]`，均逐元素精确相等。
JSONL 路径自身不调用 torch，因此同时验证 token dataset 的真实 Tensor 输出。
这些是数据处理场景，模型参数、梯度和训练级别不用于替代其验收断言。

测试文件 SHA256：
`969f0cd671035f9b0891e9d40a4a359c403759b29f36d670d4a9dff6b794498a`。
Jittor 一次完整运行 328.13 s，含全新 CPU 核心与算子编译；独立 PyTorch 一次运行 4.11 s。
这些是功能测试墙钟，不作为吞吐或稳态性能对比。

## 门禁与复现

本轮仓库布局、Markdown 链接治理与发行清单检查通过。后续作业 `119936` 的
缺依赖负向门禁 **9/9 通过**，相关结构检查 **43/43 通过**。

全结构首轮为 1513 passed、6 skipped、1 failed；唯一失败为验证环境缺少 ASV。
补齐仓库固定的 `asv==0.6.6` 后，上述 43 项复验包含全部 ASV 入口、门禁范围、
私有路径与文档结构检查。首轮失败记录保留，没有把它改写成一次全量通过。

测试入口为
[`test_datatrove_compat.py`](../../compat/tests/torch/test_datatrove_compat.py)，
双解释器命令见
[`datatrove-torch-compat`](../../agent/skills/datatrove-torch-compat/SKILL.md)。
原始日志、环境身份、安装源码校验和 Slurm 结果位于
`$JITTOR_LAB_ROOT/_state/datatrove/publish-20261024/`，未版本化。

| 未版本化产物 | SHA256 |
| --- | --- |
| `results/119913/04-oracle.log` | `a2ea7f9e9409379061c7281a4018c0664402fb3c5e04535121d46ff39b1619e9` |
| `results/119913/05-jittor.log` | `803a51696952f217f9f8c95072f287eaef84274925443387343092b873206eb2` |
| `results/119936/report.json` | `e33a9c88226564f2a6277b4027821cead0c8957206a78a01d93a7b80f44822ff` |

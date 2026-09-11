# ms-swift 的两处本地改动

这两个补丁来自 `/home/zy/projects/jittor-lab/worktrees/` 下的两个 ms-swift
工作树，各有一条建立在上游之上的本地提交，从未推送到任何地方。机器重装前
导出到这里，只为不丢。

- `*.ms-swift-gkd.patch` — Initialize native rollout engine for GKD，基于上游 `42b17dd`
- `*.ms-swift-stage4.patch` — Fix PPO reward and value model task types，基于上游 `d17f031`

上游是 `https://github.com/modelscope/ms-swift.git`，不是我们的仓库，所以这里
只存补丁而不是推一整条外部历史。要用就 `git am` 到对应的上游基线上。

两处改动都**没有经过本仓库的验证流程**：没有复现记录、没有牙齿检查、没有逐
nodeid 归因。当作待验证的起点，不是结论。

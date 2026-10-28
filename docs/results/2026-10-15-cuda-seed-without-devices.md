# 无可见 CUDA 设备时的播种接口

- Status: 限定 CPU 与实际 CUDA seed 契约已验证；不授予完整下游训练支持。
- Date: 2026-10-15。
- Baseline: 同步 `7a1e66b3858279b40eaec0c4895c6059ce8eb8db`；原测量使用同一冻结文件、保留当时 dirty 状态，提交 `4a941dbe28fd93895c40590d4048094b333e9946` 映射这两个文件。整合基线 `6f8e160a4cc7807f89e2d24b4c41349405df7720`，整合提交 `97d35128f`。
- Owner: Torch CUDA frontend maintainers。
- Review when: CUDA 设备发现、延迟初始化、seed 转换或 Transformers/Swift 初始化路径变化。

真实 Swift Trainer 的 CPU 初始化通过 Transformers 调用 `torch.cuda.manual_seed_all`。
原实现无条件初始化 cuRAND，CPU-only 环境因此失败。独立 binary Torch 2.11 的
20 行 API 探针确认：无可见设备时仍先执行整数转换，但不初始化 CUDA，也不更改 CPU RNG；
超出 64 位的可转换整数此时不会触发设备 seed 范围错误。原候选只有 4/20 行一致。

修复先转换输入，再查询可见设备数；无设备直接返回。有真实 CUDA 时继续执行原有
完整位宽范围校验、同步与 native cuRAND 恢复。没有改变随机内核或跨框架对齐样本流。

## 原始实际验证

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| CPU 独立 Torch / 候选 API 探针 | 修复后 20/20 一致 | 参数转换、无设备、无初始化与 CPU 状态保护 |
| CPU 单测 | 20 passed / 16 skipped | 硬件节点不计执行 |
| 真实 CUDA 定向回归 | 36 passed / 0 skipped | 新契约、播种回绕与已有状态恢复 |
| 独立 Torch CUDA | 8 行通过 | 完整位宽、负 seed、重新播种与 CPU 流隔离 |
| CPU core | 322 passed / 42 skipped / 1 existing xfail | native 与 Torch 两种 mode |
| 完整结构 | 1508 passed / 6 skipped | 原 agent 的源码基线，不重标为整合测量 |

实际 CUDA 使用分配的真实 GPU；拒绝 fallback，读取真实设备样本并逐项比较重新播种后的
完整数组。初次 oracle 配置误暴露 cu122 库导致 undefined-symbol 安装错误，已保留原日志；
正式 oracle 使用独立 cu128 环境。unit cache relocation 的退出与正常重启也分别保留。

## 证据和边界

未版本化证据位于 `$JITTOR_LAB_ROOT/_state/downstream-training-evidence/20261015/`。
`seed-api-sealed-manifest-v1.json` 绑定 31 个原始文件、两份源码哈希及原始运行身份；
SHA256 为 `14630026c81b40a80cacfde7f0e680a3a575bf8e69ab4df2af73083507910b0e`。
原始失败不算最终通过，不用新版本号或新依赖重标旧结果。

整合后的工具、CPU/CUDA 定向、core 与结构回归另存
`$JITTOR_LAB_ROOT/_state/native-runner-seed-integration-6f8/integration-manifest.json`，
以实际冻结文件和提交映射记录，保留上述历史测量身份。
本项不意味着 Swift 完整训练、CPU jtrun、MPI 或真实多机已验收。

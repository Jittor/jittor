# 仓库结构现代化阶段 2：工具链、交付与性能基建

## 结果

阶段 2 已把仓库的开发与交付入口收敛到固定版本工具：Ruff 0.15.22、Mypy
1.8.0、Nox 2026.7.11、pre-commit 4.6.0、build 1.3.0、setuptools 83.0.0
和 wheel 0.45.1。`noxfile.py` 统一提供 lint、format、typing、structure、真实
Python 3.7 compile、CPU、CUDA、NPU 与 benchmark 会话，所有缓存、临时构建和 JIT
状态都写到 `${JITTOR_LAB_ROOT}/_state` 或会话临时目录。

旧 GitHub 单机 job 与 `.gitlab-ci.yml` 已删除，替换为 structure、CPU、CUDA 12.2 /
RTX 4090、CANN 9 / Ascend 910B 四层工作流。硬件 job 只在受控 push 或手动触发，
按精确 runner label 和能力预检 fail closed，不从 fork PR 或 `pull_request_target`
接触 self-hosted runner。

## 发布与容器

- release tag 只构建一份 canonical sdist 和 `py3-none-any` wheel，严格检查 tag、元数据、
  wheel 内容和跨平台安装后再通过 PyPI Trusted Publishing 与 GitHub CLI 发布同一产物。
- Jittor 是运行时编译的纯 Python wheel，不能伪造成平台 wheel。三平台 job 使用
  cibuildwheel 4.1.0 的 pure-wheel 拒绝作为架构断言，再安装 canonical wheel；不会上传
  改标签的重复产物。
- 发布、离线打包和 polish 路径已去掉 `setup.py sdist`、twine token、rsync/ssh 与服务端
  发布副作用。离线工具只在本地用 `python -m build --sdist` 产出归档。
- 一个参数化根 Dockerfile 同时覆盖 Ubuntu 24.04 CPU 和 NVIDIA CUDA 12.2.2 / cuDNN 8
  vendor 镜像；旧 `Dockerfile_cuda11` 与 bionic 源已从活动容器路径删除。tag workflow
  才拥有 GHCR `packages: write` 权限。

本机 Docker client 为 29.1.3，但当前用户无 `/var/run/docker.sock` 权限，因此本阶段只完成
Dockerfile/工作流静态验证；真正的两种 image build 是 `containers.yml` 的必过门禁，未把
本地未执行描述成成功构建。

## 性能回归

`asv.conf.json` 使用 existing environment 与 `--python=same`，避免为历史 commit 重复
编译整个 JIT runtime。`benchmarks/` 首批覆盖：

- Jittor/可选真实 PyTorch 的 matmul、softmax、LayerNorm、GELU 时间与 working set；
- 固定 2 层、hidden 256、FFN 768、8 head/4 KV head、batch 2、sequence 128 的 Tiny
  Llama 前向和前反向；
- 32/128/512 个张量下 SGD、AdamW step 的扩展性。

benchmark 在导入 Jittor 前强制检查独立 `JITTOR_HOME` 和 `cache_name=asv-*`。缺 CUDA
或真实 PyTorch会明确 skip，Jittor CPU 基准失败则直接失败。ASV 0.6.6 发现并检查 6 个
benchmark；强制 Jittor CPU GELU 实跑通过，最终 working set 为 272,338,944 bytes。

## 制品与验证

最终 index 制品位于（未版本化）：

`/home/zy/projects/jittor-lab/_state/verify/repository-modernization/stage2-build.sHPzrJ`

| 产物 | SHA-256 |
| --- | --- |
| direct wheel | `d7e8441e5926048496e41719b801507459d6fc84815fcccbb39ffc9adcc754d0` |
| sdist wheel | `d7e8441e5926048496e41719b801507459d6fc84815fcccbb39ffc9adcc754d0` |
| sdist | `77c213e51d85c1b47eabe73a23b02100ae6660699670bd9e767cb0562fdd6121` |

相对阶段 1，wheel 为 0 项新增、6 项批准内容变化、5 项批准删除；相对原始 1,053 项
基线的累计转换为 6 项新增、13 项内容变化、5 项删除，最终 1,054 项。显式阶段清单可
覆盖默认上一阶段哈希，同一显式阶段内冲突仍失败；标准库门禁测试 9/9 覆盖该顺序。
direct wheel 与 sdist 重建 wheel 的 1,054 个成员路径和内容逐字节相同，整个 wheel
SHA-256 也一致。

| 验证 | 结果 |
| --- | --- |
| Ruff / format / Mypy | Nox 三会话通过；Mypy ratchet 5 文件无问题 |
| Python 3.7 | 528 个 Python 文件按 3.7 grammar 全过；本机无真实 3.7，CI 使用真实解释器 |
| 文件系统结构测试 | 38/38 |
| ASV | 6 个 benchmark 检查通过；强制 Jittor CPU case 实跑通过 |
| staged archive CPU | autograd engine 9/9；silent-wrong regression 11/11 |
| wheel gate | 1,053 -> 1,054 精确转换通过；direct/sdist member diff 0 |
| 隔离 wheel CPU | 冷启动来源、metadata、深层资源和数值 smoke 通过 |
| 隔离 deploy | 7 项递归部署、`--check`、torch/flash-attn 导入通过 |
| 隔离 wheel CUDA | RTX 4090 / JTCUDA 12.2 / cuDNN 8；loss `14.0`，grad `[2,4,6]` |
| workflow / container | 6 份 YAML、权限、Bash 和静态契约通过；本机 Docker daemon 不可用 |

本机无 CANN/Ascend 设备，因此 NPU 只完成了 fail-closed 工作流与 nox 会话契约，未声明
数值通过。Python 3.7、CUDA、NPU 的 CI job 都会在能力缺失时失败而非降级或假绿。

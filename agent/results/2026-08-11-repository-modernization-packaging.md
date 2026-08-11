# 仓库结构现代化阶段 1：打包、部署与 wheel 基线

## 状态

✅ PEP 621 打包、完整包发现、显式资源清单、部署复制修复和 wheel 内容基线已完成。

## 改动

- 新增 `pyproject.toml`，由 PEP 621 统一声明名称、动态版本、Python 版本、依赖、
  CUDA extra、URL 和 setuptools build backend。
- `setup.py` 收敛为兼容旧工具的 `setup()` shim，不再复制元数据。
- setuptools 使用 `find_packages(where="python", namespaces=False)`。为四条断裂的
  regular-package 父链补充无副作用 `__init__.py`，最终文件系统与发现结果为 35/35。
- `MANIFEST.in` 显式递归包含 `python/jittor` 和 `python/jittor_utils`，并排除 bytecode、
  cache 和运行态目录；旧六层 `package_data` 通配已删除。
- `.gitignore` 不再忽略全部 Markdown 或裸 `test.py`，并正式放行可复用 Python 工具
  和 wheel 基线。
- `torch_shim.deploy` 递归复制每个 stub 包的全部 Python 文件，部署清单从 6 项增至
  7 项，`flash_attn/flash_attn_interface.py` 现在真实进入目标目录。
- deploy 先完整预检所有目标，再写文件；`--check` 校验缺失、内容 SHA-256 和不安全
  路径，同时保留目标包存在额外文件的兼容行为。安装器生成的一级 `__pycache__`
  会被明确忽略，其他缺少 `__init__.py` 的一级 stub 目录仍会响亮失败。

## Wheel 内容基线

新增：

- `agent/baselines/wheel-contents-1.3.11.0.txt`：来自已验收旧 wheel 的 1,053 条
  `成员 SHA-256 + 精确 archive 路径`。
- `agent/baselines/wheel-additions-stage1.txt` 与
  `wheel-content-changes-stage1.txt`：锁定阶段 1 的 6 项新增和 9 项内容变化；批准项
  必须全部被候选消费，因而旧 wheel 回退也不能通过当前门禁。
- `agent/scripts/check_wheel_contents.py`：生成 manifest、比较旧 wheel/哈希基线，默认
  拒绝未批准的新增、内容变化和删除，同时拒绝路径穿越、重复成员、cache/build/
  实验污染及关键资源缺失。三类显式 allowlist 同样要求精确消费。
- `agent/scripts/test_check_wheel_contents.py`：纯标准库覆盖篡改 `UnpackRaw.cuh`、未知
  `secret.env`、重复/穿越/污染成员、缺失资源、错误哈希和未消费许可。

用法：

```bash
python agent/scripts/check_wheel_contents.py compare dist/jittor.whl
python agent/scripts/check_wheel_contents.py compare new.whl --old-wheel old.whl
python agent/scripts/check_wheel_contents.py manifest old.whl --output baseline.txt
```

计划原文认为 `UnpackRaw.cuh` 已掉出 wheel，但旧 1,053 条基线证明它本来就在。阶段 1
做的是把该文件变成强制检查项，消除依赖偶然深层通配的风险。

最终 index wheel 位于（未版本化）：

`/home/zy/projects/jittor-lab/_state/verify/repository-modernization/stage1-final-setuptools83.h9crdb`

| 产物 | SHA-256 |
| --- | --- |
| direct wheel | `c723d6d1b714be930b6f7c1dd274397a585718d334db81ccaf75c22e68c0ba94` |
| sdist wheel | `8c64940837d0b2c7f32319563c1c941fd390c523dbac7f22cc770ffab8db31a0` |
| sdist | `b431a80dd049acec270445bf272e8107acf50bfed35e17ddebfb76d13cfea69d` |

制品通过 `python -m build` 和 setuptools 83.0.0 / wheel 0.45.1 构建。direct wheel
与 sdist 重建 wheel 均为 1,059 项，1,059 个成员内容逐字节一致；相对 1,053 项基线
为 6 项批准新增、9 项批准内容变化、0 项删除。`UnpackRaw.cuh`、
`flash_attn_interface.py`、`utils/data.gz`、`other/code_softmax.py` 和 LICENSE 均存在，
无 pyc/cache/build/实验污染；旧 wheel 直接使用默认门禁会因缺少批准转换而失败。

## 验证

| 验证 | 结果 |
| --- | --- |
| package discovery | 35/35 |
| 新增 packaging 结构测试 | 4/4 |
| deploy 文件系统测试 | Python 3.11 12/12；Python 3.10 12/12 |
| wheel gate 标准库测试 | 8/8；Python 3.7 grammar 通过 |
| 合并源码结构矩阵 | staged archive 75/75 |
| 旧/新 wheel 精确转换 | 1,053 -> 1,059；added 6，changed 9，removed 0 |
| direct/sdist wheel | 路径 1,059/1,059；成员 byte diff 0 |
| 隔离 wheel CPU 冷启动 | 通过，来源为隔离安装目录，metadata version `1.3.11.0` |
| 隔离 deploy | pip bytecode cache 存在时仍通过；7 项；部署后 `--check` 通过 |
| 隔离 wheel CUDA | RTX 4090 / JTCUDA 12.2 / cuDNN 8；loss `14.0`，grad `[2,4,6]` |
| repository layout / diff check | 通过 |

CUDA 首次按系统 `/usr/local/cuda` 探测时因缺少开发版 `cudnn.h` 响亮失败；按项目约定
切换 `/home/zy/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux` 后通过。这是环境选择诊断，
不是 wheel 内容回归。

## 已知边界

- 本机没有 Python 3.7 解释器；阶段 1 新增/修改的 Python 文件通过 Python 3.7 grammar
  检查，但全仓静态审计发现既存 `einops/experimental/indexing.py` 含 Python 3.8 语法
  与 3.7 运行时不兼容注解。阶段 2 必须先修复，再把真实 3.7 编译/构建纳入门禁。
- 本阶段未执行 NPU。没有把 CUDA 结果扩写为 ACL/NPU 支持证据。
- 测试、demo、notebook 和脚本仍在 wheel 中，以满足阶段 1 的严格不删基线；阶段 5/6
  按审核后的 removal allowlist 调整发行边界。
- 两棵 runtime tree 的 `recursive-include` 是完整性优先的过渡清单；领域包和发行边界
  稳定后再收窄到按职责声明的资源集合。

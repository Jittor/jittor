# Torch shim

Torch shim 让一个面向 torch 编写的 Python 应用把 Jittor 当作运行时。可复用的 Torch
API、部署、扩展构建、导入补丁和外部后端机制，全部归**独立的 `jittor-torch` 发行物**
所有；核心 `jittor` 发行物**不包含**任何这些兼容文件。项目专属的运行时策略由可选的
适配器发行物提供。

## 组成

| 路径 | 作用 |
| --- | --- |
| `compat/shim/resources/torch/__init__.py` | 安装与部署后的 `torch/__init__.py` 入口（很薄） |
| `resources/stubs/` | 随包提供的兼容包，如 `flash_attn`、`torchvision`、`torchaudio`、`torchdata` |
| `cpp_extension/` | 由 Jittor 支撑的 `torch.utils.cpp_extension` 接口，用于构建源码扩展 |
| `runtime.py` | 隔离运行时的建立与原生扩展发现 |
| `bootstrap.py` | 上者稳定而精简的公开门面 |
| `deploy.py` | 把完整 shim 树安装到目标 site-packages |

规范的 torch 兼容 API 位于 `jittor.compat.torch`。shim 把该 API 以 `torch` 模块名
导出，并接好第三方库期望的子模块路径。

`jittor.torch_shim` 仍是 `jittor.compat.shim` 的导入兼容别名，两个名字解析到**同一批
模块对象**。

## 引导

开发时从 checkout 根目录同时安装两个项目：

```bash
python -m pip install -e . -e ./compat
```

可选项目把顶层 `compat/` 源码树映射到已安装的 `jittor.compat` 包。**只设
`PYTHONPATH=python` 仅选中核心**；使用兼容 API 或跑它们的测试时，要在同一个解释器
里安装这个可选的 editable 项目。测试第二个 checkout 时，也要装那个 checkout 的
`compat` 项目。nox 的运行时门禁就是这么显式做的。

wheel 部署时，在同一个环境里安装配套的核心 wheel 与 `jittor-torch` wheel。分别用
`python -m build .` 和 `python -m build ./compat` 构建，后者声明了对核心版本的依赖。
**`torch` 本身由这个独立 wheel 提供。** 下面的部署助手在应用需要时会额外装上随包的
第三方 stub。

需要项目局部运行时的应用可以显式启用：

```python
from jittor.compat.shim import activate, activation_status

activate(project_root=__file__)
assert activation_status().active
import torch
```

激活与部署后的 `import torch` 发布**独立的** Tensor、Parameter 和 Module 类型：
**`torch is not jittor`**。原生 Var 方法保留自己的契约。**只有这一种形态**：历史上
把 Torch API 装到原生 `jittor` 模块上的别名模式已经移除，`independent_namespace=False`
与 `install(jittor)` 直接报迁移错误（`compat/shim/preflight.py` 的
`require_independent_frontend()`）。

设 `JITTOR_TORCH_SHIM=1` **就是开启开关**：`import jittor` 时会组合出同一个独立
`TorchNamespace`，不需要 PYTHONPATH、也不需要预先部署。它给到的与 `activate()`
相同——`torch is not jittor`、`torch.Tensor is not Var`。部署入口
（`compat/shim/resources/torch/__init__.py`）只设 `JITTOR_TORCH_SHIM=1`；它还会在
`JITTOR_TORCH_INDEPENDENT` 被设成假值时**直接抛错**，因为那个值点名的是已移除的前端。
这个变量今天不需要设。

有一处顺序要求：环境变量能让 `torch` 可导入，但**不能把模块放上 `sys.path`**。
所以 `import torch` 作为进程里第一个 import 时，仍然需要一份已部署的 `torch` 包
（或把 `compat/shim/resources` 放进 PYTHONPATH）；先 `import jittor` 则不需要。
契约由 `compat/tests/torch/test_env_activation_contract.py` 钉住。

`activate()` 是**进程级且幂等**的：重复调用返回最初的激活结果，不会重新扫描扩展或
重新打补丁。除非设了 `JITTOR_TORCH_RUNTIME_ROOT`，它会在
`${XDG_CACHE_HOME:-~/.cache}/jittor/torch-shim/` 下建立运行时，把 Jittor、torch 扩展、
CUDA、Triton、pip 和临时缓存都放在该运行时之下，并把 shim 部署进它本地的 site-packages。

本地源码扩展从 `setup.py`、`pyproject.toml` 和 `CMakeLists.txt` 的信号中发现。缺失或
过期的 setuptools 扩展会通过 Jittor 支撑的 cpp-extension API 重建。设
`JITTOR_TORCH_SKIP_EXT_BUILD=1` 可跳过暖启动的构建检查，或在自动发现不合适时显式传
`extension_dirs`。

**为了数值一致性**，除非设了 `JITTOR_TORCH_KEEP_FAST_MATH=1`，引导过程会为 Jittor 的
JIT kernel 关闭 CUDA fast-math contraction。项目自己的扩展保留其构建定义所请求的选项。

## 可选适配器

适配器使用两个公开的 entry-point 组：

- **`jittor.module_patches`** —— 通过 `jittor.compat.module_patcher` 注册精确的模块
  路径回调；
- **`jittor.external_backends`** —— 通过 `jittor.compat.external_backend` 注册扩展
  发现策略。

本仓库维护的适配器是一个独立发行物：`jittor-torch-adapters`（源码在
[`adapters/`](https://github.com/Jittor/jittor/blob/master/adapters/README.md)，
Python 包 `jittor_adapters`，依赖 `jittor-torch` 同版本）。它注册三个
`jittor.module_patches` entry point：

| entry point | 版本 | 作用 |
| --- | --- | --- |
| `jittor_transformers` | Transformers 4.56.2、5.5.3 | 让 Transformers 自带的 `torch_npu` 探测返回假 |
| `jittor_torchmetrics` | TorchMetrics 1.7.4 | 私有的 bincount / 拼接 / safe-divide 助手保持既有 Jittor 行为 |
| `jittor_vllm` | 见 `adapters/README.md` | vLLM 的编译扩展面与层补丁 |

安装它即可让 entry point 被发现；未安装时报告为 `unavailable`，不影响前端。
版本是显式的：未识别的版本在应用补丁时抛 `UnsupportedAdapterVersion`。
**Jittor 自身不导入这些第三方库、不检查它们的目录结构、也不安装长期存在的项目专属
导入 finder。** 其它项目（TRELLIS、Gaussian Splatting 之类）的粘合代码按同样的
entry point 形状做成各自的可选发行物，**不在本仓库，也不随本发布线发布**。

## 部署

`jittor-torch-shim` 是 `jittor-torch` 发行物提供的命令（`compat/pyproject.toml`
的 `[project.scripts]`，实现在 `compat/shim/deploy.py`）：

```bash
jittor-torch-shim                            # 部署进当前环境的 site-packages
jittor-torch-shim --target /path/to/site-packages
jittor-torch-shim --check                    # 校验已部署的文件与源文件一致
```

它拷贝四样东西：`resources/torch/__init__.py` 到 `<site-packages>/torch/__init__.py`、
`resources/stubs/<包>/**` 到各自的顶层包（`torchvision`、`torchaudio`、`torchdata`、
`flash_attn`）、以及 `torch-<版本>.dist-info` 与 `flash_attn-<版本>.dist-info` 两份
发行元数据（让按版本号做门控的库能查到）。命令是幂等的，改完随包资源后可以重跑。
**不要只复制 `resources/torch/__init__.py`**——嵌套的 stub 模块与接口是运行时契约的
一部分，`--check` 会把缺失或内容不一致的文件逐条列出来。

部署完成后它会打印一条校验命令，读的是 `torch.__file__`、`torch.__torch_version__`
和 `torch.version.__version__`——即**Torch API 级别**，而不是同一个模块上
`torch.__version__` 暴露的 Jittor 版本。

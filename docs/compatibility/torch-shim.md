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

默认激活与部署后的 `import torch` 发布**独立的** Tensor、Parameter 和 Module 类型：
**`torch is not jittor`**。原生 Var 方法保留自己的契约。要用历史上的 Jittor 别名模式，
显式传 `independent_namespace=False`；**已经激活的进程不能再换模式**。部署入口在设置
激活标志的同时会设 `JITTOR_TORCH_INDEPENDENT=1`，好让子解释器选中同一模式。单独设 `JITTOR_TORCH_SHIM=1` **就是开启开关**：`import jittor` 时会组合出同一个
独立 `TorchNamespace`，不需要 PYTHONPATH、也不需要预先部署。它给到的与
`activate()` 相同——`torch is not jittor`、`torch.Tensor is not Var`。
（历史上这个变量选中的是把 Torch API 装到原生模块上的 import 期路径，那条路径已经移除，
`install(jittor)` 现在直接拒绝。）

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

维护中的适配器都是独立发行物：

| 发行物 | 用途 |
| --- | --- |
| `jittor-trellis` | TRELLIS.2 的运行时策略与 kernel |
| `jittor-gs` | graphdeco Gaussian Splatting 的运行时策略与启动工具 |
| `jittor-hf-compat` | 显式选择的 Transformers 版本适配 |

安装适配器即可让它的 entry point 被发现；应用也可以显式调用适配器的 `install()`。
**Jittor 自身不导入这些项目、不检查它们的目录结构、也不安装长期存在的项目专属导入
finder。**

## 部署

用维护的助手部署完整的 shim：

```bash
jittor-torch-shim --target /path/to/site-packages
```

目标目录会包含 torch 包、随包 stub 和发行元数据。**不要只复制
`resources/torch/__init__.py`**——嵌套的 stub 模块与接口是运行时契约的一部分。

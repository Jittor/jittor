---
name: torch-api-cohort-promotion
description: 把 compat 层某一族 torch API 从 install 期闭包提升为模块级一等对象并登记保真度（任务 7.03）的完整口径——先核 owner、三件套验收怎么写、`_axis_to_dim` 适配器为什么会让「模块级对象」与 Var 方法不是同一个对象、以及怎么让一个 cohort 真的在 CUDA 上跑一遍而不是只写「CPU N passed」。改 `python/jittor/compat/torch/installers/**` 里的 install 闭包、或要给某个 torch API 写 fidelity 元数据时读这一篇。
---

# 把一族 torch API 提升为模块级一等对象

适用对象：`python/jittor/compat/torch/installers/{tensor,nn,numerical,cuda,data}.py`
里那些定义在 `install(...)` / `_install_*(...)` 内部的闭包。目标形态是
**模块级 def + `register_fidelity` + install 只做绑定**。

## 1. 先核最终 owner（跳过这步会写出重复实现）

三种情况，处理方式不同：

```bash
# owner 探针：这个名字在原生 jittor 里存在吗？在 Var 上呢？
PYTHONPATH=<worktree>/python JITTOR_HOME=<...> TMPDIR=<...> \
JITTOR_TEST_DEVICES=cpu nvcc_path="" python -c "
import jittor as jt
for n in ['amax','logaddexp','cumsum']:
    print(n, 'mod=', hasattr(jt,n), 'Var=', hasattr(jt.Var,n))"
# 有原生 owner 的，再把定义读出来逐行比对
rg -n 'def amax|def amin' python/jittor/misc python/jittor/nn --glob '!compat/**'
```

- **原生 owner 存在且契约就是 torch 的**（例：`jittor/misc/reductions.py` 的
  `amax`/`amin`/`count_nonzero`）→ 模块级对象写成**薄转发**，捕获原生 owner
  （`_NATIVE_AMAX = jt.amax`，在模块导入期捕获，那时 install 还没覆写它），
  fidelity detail 里写 "re-exports Jittor's native ... owner"。**删掉 compat 里
  那份复制的实现**，不要保留两份。
- **原生 owner 存在但契约不同**（jittor 的 `argmax` 返回 `(idx, val)`，torch 只返回
  idx）→ compat 拥有薄包装，detail 写清差在哪。
- **没有原生 owner**（`logaddexp`、`sign`、`trunc`、`frac`、`nan_to_num`）→ compat
  是最终 owner，实现整个搬到模块级。

## 2. install 期闭包依赖的状态怎么搬

多数闭包捕获的只是 `jt.*` 原生算子，直接模块级捕获即可。少数捕获的是**只能在
install 期取到**的东西——典型是 `_orig_setitem = Var.__setitem__`（必须在打补丁前取），
`_write_index_parent` 这类 retained-view 传播器就建立在它上面。

做法照 `numerical.py` 的 `_vmap_runtime_impl`：模块级留一个 `None` 句柄，install
时用 `global` 交接。

```python
_index_parent_writer = None          # 模块级


def _assign_out(out, value):
    out.assign(value)
    if _index_parent_writer is not None:
        _index_parent_writer(out, out)
    return out


def _install_tensor_methods(g, Var, ...):
    ...
    global _index_parent_writer
    _index_parent_writer = _write_index_parent    # 在 _write_index_parent 定义之后
```

判据：模块级对象**单独 import 不炸**（`out=None` 的路径必须能跑），只有需要那份
install 期状态的分支才依赖句柄。

## 3. `_axis_to_dim` 会让身份断言失败——这是坑，不是你的 bug

`installers/tensor.py` 的 `install_methods` 末尾有一段：

```python
for _rn in ("max", "min", "argmax", "argmin", "amax", "amin", "cumsum",
            "norm", "std", "var"):
    setattr(Var, _rn, _axis_to_dim(getattr(Var, _rn)))
```

它给这十个 Var 方法**重新包一层**做 `axis=` → `dim=` 翻译。于是
`torch.amax is jittor.Var.amax` 会失败，报的是
`<function install_methods.<locals>._axis_to_dim.<locals>._w> is not <function amax>`。

**正确的解法不是放弃身份断言**，而是让稳定对象自己接 `axis=`，并让适配器跳过它：

```python
def amax(input, dim=None, keepdim=False, keepdims=None, axis=None):
    return _NATIVE_AMAX(input, dim if axis is None else axis,
                        keepdim=keepdim, keepdims=keepdims)
amax._torch_accepts_axis = True      # 适配器据此跳过

def _axis_to_dim(orig):
    if getattr(orig, "_torch_accepts_axis", False):
        return orig
    ...
```

这十个名字里 `max`/`min`/`argmax`/`argmin`/`cumsum`/`norm`/`std`/`var` 都还没迁，
后面每个 cohort 都会撞上同一条。

## 4. 三件套验收（每个 cohort 都要这三样）

1. **模块级身份**：`torch.foo is <owner module>.foo`，且
   `torch.Var.foo is <owner module>.foo`（如果它也是 Var 方法），
   `foo.__module__ == owner.__name__`、`foo.__name__ == "foo"`。
   这三条一起才能排除「install 期又包了一层」。
2. **metadata**：`fidelity_of("torch.foo")` 的 `implementation` 是同一个对象、
   `level` 是 `APPROXIMATE`（能证明 exact 才写 exact），detail 里逐项写清**不支持
   什么**（out / device / layout / dtype / interpolation / named-dimension）。
   测试用 `assertIn("device", record.detail)` 这类断言钉住，**detail 改了测试要跟着改，
   不要反过来删断言**（本仓库出现过 `cosine_similarity` 的 detail 与断言对不上、
   整个 fidelity 测试文件长期一条红）。
3. **数值对拍**：对 NumPy 的 CPU 定点对拍 + Var 方法委托一致。dtype 相关的用例写
   `jt.array(v, dtype="float64")`，`jt.array(np.ones(4,"float64"))` 会静默变 float32。

## 5. 让 cohort 真的过 CUDA（第 §0 条完成定义要求的那一层）

**不要**手写 `if jt.has_cuda: with flag_scope(use_cuda=1)`。用设备参数化引擎，
门禁的 `JITTOR_TEST_DEVICES` 就能同时驱动两侧：

```python
from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


class TestTorchCumulative(cu.JittorTestCase):
    def test_xxx(self, device):        # 引擎负责 flag_scope(use_cuda=...)
        ...


instantiate_device_type_tests(TestTorchCumulative, globals())
```

设备无关的身份/metadata 用例放在普通 `unittest.TestCase` 里，不要参数化（跑两遍没意义）。

跑两侧、并**确认 CUDA 那侧真的实例化出来了**（空类收集为 0 条会报成 pass）：

```bash
# CPU
JITTOR_TORCH_SHIM=1 JITTOR_HOME=<...> TMPDIR=<...> \
JITTOR_TEST_DEVICES=cpu nvcc_path="" taskset -c <核> python -m pytest <文件> -q
# CUDA
JITTOR_TORCH_SHIM=1 JITTOR_HOME=<...> TMPDIR=<...> \
JITTOR_TEST_DEVICES=cuda CUDA_VISIBLE_DEVICES=<卡> \
nvcc_path=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH \
taskset -c <核> python -m pytest <文件> -q
# 判据：生成的类名里必须有 ...CUDA，不能是 ...Unselected
... python -m pytest <文件> -q --collect-only 2>&1 | grep -oE 'Test[A-Za-z]+' | sort -u
```

在 `nvcc_path=""` 与 `nvcc_path=/usr/local/cuda/bin/nvcc` 之间来回切会换掉缓存配置，
第一条命令常拿到 `jit_utils was rebuilt ... rerun the same command`。**照做重跑，
不要清缓存**；脚本里用重试循环包一层：

```bash
for i in 1 2 3; do <命令> > run.log 2>&1
  grep -q "rerun the same command" run.log || break; done
```

## 6. 该挑哪个 cohort 上 CUDA，以及跑出不一致怎么办

挑**归约 / 累加顺序 / dtype 提升**那一类：`cumsum`/`cumprod`、`logsumexp`、
`logcumsumexp`、`sort`/`topk`/`median`（并列元素的 index 定序）、`argmax`/`argmin`。
纯 elementwise 的（`square`、`copysign`）两侧必然一致，跑了也说明不了什么。

跑出差异**先量再判**，不要直接改实现。量的方式是三路比对，不是两路：

| 比什么 | 为什么 |
| --- | --- |
| CPU vs float64 参考 | 顺序 backend 的误差基线 |
| CUDA vs float64 参考 | 并行 backend 的误差基线 |
| CPU vs CUDA | 两条并行路径之间的差 |

实测（sm_89，4096 个 float32、正负交替、部分和量级 ~2e3 的 `cumsum`）：
CPU 与 float64 差 3.7e-03、CUDA 与 float64 差 2.1e-03、两者互差 2.7e-03
（相对 1.4e-06）。**CUDA 更准**——顺序扫描的误差随 n 线性累积，树形前缀和是 log n。
小数组（12 个元素）两侧逐位相同；整数/bool 路径两侧永远逐位相同。

判据：

- 两侧都在 float64 参考的合理容差内、且差异随长度按 float32 舍入的规模增长
  → **不是 bug**，写进 fidelity detail（"summation order is the backend's…"），
  测试写成**有界不一致**（相对容差）加**整数路径逐位相等**，不要写成 `assert_array_equal`。
- 只有一侧偏离参考，或差异远超 float32 舍入规模 → 是 bug，按 verify-then-fix 走。

## 7. 收尾前必查

- `JITTOR_TORCH_SHIM=1 pytest tests/structure -q`——**注意它不是 3 秒、也不是全绿**：
  本机实测约 2 分 15 秒，HEAD 上就有 15 条红。判据是**与改前逐条同集合**，
  不是「全绿」。先在改前跑一次留基线。
- 每个测试目录单独一条 pytest 命令；`tests/structure` 与 `tests/compat/torch`
  合并会因 `conftest` 模块名被抢而报假错。
- 禁止 `git stash` 的前提下要拿改前基线：`cp <文件> $TMPDIR/<文件>.wip`
  → `git checkout -- <文件>` → 跑 → `cp` 回来。**不要用 `git apply`**，它会写索引，
  后续不带路径的 `git commit` 会把别的东西一起带走。

---
name: torch-checkpoint-stride-oracle
description: 验证 Jittor 读 PyTorch checkpoint（.pt/.pth）读得对不对。用于改 compat/torch/serialization.py、jittor_utils/load_pytorch.py，或排查「权重加载了但数值不对/有零」的场合。核心是 torch 存的是整块 storage 加 (offset,size,stride)，非连续视图会被读错；本 skill 给出真 torch 对拍口径与不依赖 torch 的手工构造 checkpoint 方法。
---

# 读 PyTorch checkpoint 读得对不对

## 1. 先知道 torch 存的是什么

`torch.save` **不**把张量展平。它把**整块 storage** 写一份，再对每个张量记
`(storage_offset, size, stride)`。所以任何**视图**——转置、切片、`[::2]`、
融合 QKV 里的一段、多头注意力里的一个头——存下来都是「一块更大 buffer 的非连续描述」。

读的一方只要把 stride 丢掉（`storage[offset:offset+prod(size)].reshape(size)`），
就会**读到另一组元素**：形状对、不报错、数值错。这类 bug 不会崩，只会让模型输出变差。

判断一个 reader 有没有这个毛病，看它对 `stride` 做了什么：

```bash
grep -n "stride" python/jittor/compat/torch/serialization.py python/jittor_utils/load_pytorch.py
```

两种典型错法：

- 完全无视 `stride`（切片后 reshape）；
- **先把 storage 截断到 `prod(size)` 再按原 stride reindex**——原 stride 索引的是完整
  storage，截断后大部分下标落在源外，jittor 的 `reindex` 把越界读**填 0**。症状就是
  「前面几行对，后面全是 0」。

## 2. 对拍口径：真 torch 生成，Jittor 读

真 PyTorch 在 `jt312b` 环境里（`jt311` 里的 `torch` 是 shim，不是 torch）。

```bash
# 1) 用真 torch 造样本，同时把期望值存成 npz
taskset -c <核段> /home/zy/miniconda3/envs/jt312b/bin/python - <<'PY'
import torch, numpy as np
base = torch.arange(24., dtype=torch.float32).reshape(4, 6)
obj = {
    "contig":    base.clone(),
    "transpose": base.t(),                       # stride (1,6)
    "colslice":  base[:, 2:5],                   # offset 2, stride (6,1)
    "rowstep":   base[::2],                      # stride (12,1)
    "narrow3d":  base.reshape(2,3,4)[:, 1:, ::2],
    "expanded":  base[0, :2].expand(3, 2),       # stride 0
    "scalar":    base[2, 3].clone(),             # 0 维
    "half":      base.to(torch.float16)[:, ::3],
    "long":      base.to(torch.int64).t(),
    "param":     torch.nn.Parameter(base[:, :2].clone()),
}
torch.save(obj, "oracle.pt")                                    # zip 格式
torch.save(obj, "legacy.pth", _use_new_zipfile_serialization=False)  # 旧格式
np.savez("expected.npz", **{k: v.detach().numpy() for k, v in obj.items()})
for k, v in obj.items():
    print(k, tuple(v.shape), v.stride(), v.storage_offset(), v.dtype)  # 这行是判据来源
PY
```

**这九个样本是最小覆盖集**：连续、转置、带 offset 的切片、跨行步长、三维带 offset+步长、
stride 0、0 维、非 float32、Parameter。少一个就会漏掉一条真实路径。

读回来对拍（注意三件套环境变量与 `PYTHONPATH`，见 `jittor-worktree-verification`）：

```python
exp = np.load("expected.npz")        # 必须在 import jittor 之前读路径，见下面的坑
import jittor as torch
got = torch.load("oracle.pt")
for k in exp.files:
    e, g = exp[k], np.asarray(got[k].numpy())
    assert g.shape == e.shape and np.array_equal(g.astype(e.dtype), e), k
```

### 两条会让对拍白做的坑

- **两个 reader，走哪个取决于是否处于 torch 模式。** 裸 `python x.py` 走
  `jittor_utils/load_pytorch.py`（原生 `jt.load`）；`JITTOR_TORCH_SHIM=1` 才走
  compat 的 `serialization.py`。**两条路都要跑**，它们是各自独立实现的，可以一个对一个错。
  确认走了哪条：在 rebuild 函数里临时 `print(..., file=sys.stderr)`，没打印就是没走。
- **torch 模式在 import 期改写 `TMPDIR`/`HOME`。** 脚本里 `os.environ["TMPDIR"]` 在
  `import jittor` 之后会变成 shim 的 runtime 目录，用它拼路径会 FileNotFoundError。
  **先把路径读进变量再 import**，或者用一个自己的环境变量名传路径。

已知且可接受的差异（不要当成 bug）：`int64` 读成 `int32`、`float64` 读成 `float32`
（`jt.array` 收窄），数值本身相同。

## 3. 不依赖真 torch 的回归测试：手工造 checkpoint

仓库测试不能依赖装了 PyTorch。zip 格式可以手工拼：`archive/data.pkl`（对象图，张量是
`torch._utils._rebuild_tensor_v2` 的 REDUCE，storage 是 persistent id）加
`archive/data/<key>`（原始字节）。

难点是 pickle 里要写 `GLOBAL 'torch FloatStorage'`——直接 pickle 一个假类会被
`save_global` 的 `obj2 is obj` 校验挡下。**用纯 Python pickler 覆写 `save`**：

```python
class _Global:                      # 只是个名字，可调用是因为 save_reduce 要求
    def __init__(self, module, name):
        self.module, self.name, self.__name__ = module, name, name
    def __call__(self, *a, **k): raise AssertionError

class _Pickler(pickle._Pickler):    # C 版没有可覆写的 save
    def persistent_id(self, obj):
        if isinstance(obj, _StorageRef):
            return ("storage", _Global("torch", "FloatStorage"), obj.key, "cpu", obj.array.size)
    def save(self, obj, save_persistent_id=True):
        if isinstance(obj, _Global):
            self.write(pickle.GLOBAL + (obj.module+"\n"+obj.name+"\n").encode("ascii"))
            self.memoize(obj); return
        return pickle._Pickler.save(self, obj, save_persistent_id)
```

张量本身用 `__reduce__` 返回 `(_Global("torch._utils","_rebuild_tensor_v2"), args)`。
完整可运行版本见 `tests/compat/torch/test_torch_compat_load_strided.py`。

**旧格式（`_use_new_zipfile_serialization=False`）不要手工拼**——它是「三个 pickle 加
storage 键表加裸字节」，逐字节复刻等于在测自己对代码的理解。改成**直接单测 rebuild 函数**：
`jittor_rebuild_direct(...)` 返回的 `ArrayWrapper` 加 `materialize_wrappers({...})`，
参考 `tests/core/test_load_pytorch_strides.py`。

## 4. 期望值怎么算（不要手算）

用 numpy 做同样的跨步读，这正是 torch 自己做的事：

```python
def reference(base, offset, size, stride):     # base 是一维 storage
    if not size:
        return base[offset:offset+1]
    return np.ascontiguousarray(np.lib.stride_tricks.as_strided(
        base[offset:], shape=size, strides=tuple(s*base.itemsize for s in stride)))
```

## 5. 判据

跑绿了不算数，还要能回答：

1. **两个 reader 都跑过了吗？**（原生 `jt.load` 与 `JITTOR_TORCH_SHIM=1` 的 `torch.load`）
2. 九个样本里，**非连续的那六个**是不是修前就失败？没失败说明根本没走到这段代码。
3. 描述越界（`size`/`stride`/`offset` 超出 storage）时是**报错**还是**读出 0**？
   读出 0 就是没修完——`reindex` 的越界填 0 正是这个 bug 最初的伪装。

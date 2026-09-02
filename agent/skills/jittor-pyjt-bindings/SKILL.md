---
name: jittor-pyjt-bindings
description: 改动 pyjt 绑定层（python/jittor/pyjt_compiler.py 代码生成器、src/pyjt/py_converter.h、任何带 @pyjt 注释的头文件）时的验证方法。用于确认「生成的 C++ 真的变了」、定位当前生效的 gen/ 目录、把会段错误的用例写成不会带走 pytest 的测试，以及避免在 worktree 里误测另一棵源码树。
---

# 改 pyjt 绑定层怎么验证

绑定层的共同失败模式是**生成器不报错、生成的 C++ 能编译、行为悄悄不对**。
所以「测试跑绿了」不算通过，还要确认你测的是自己那份生成结果。

## 1. 先确认导入的是自己这棵树

`pip install -e` 装的 `.pth` 指向**另一棵源码树**。pytest 靠 `pyproject.toml` 里的
`tests/conftest.py` 天然导入当前目录的代码，但下面这些**不会**：

- `python -c ...` / `python some_script.py`
- 测试里 `subprocess.run([sys.executable, ...])` 起的子进程
- 任何不经过 pytest 的复现脚本

所以直接跑 python 时必须显式带上：

```bash
export PYTHONPATH=$PWD/python
python -c 'import jittor; print(jittor.__file__)'   # 必须打印本目录的路径
```

测试里起子进程时，把父进程的包目录传下去（本仓库
`tests/core/test_pyjt_binding_protocol.py:run_in_subprocess` 就是这么写的）：

```python
env = dict(os.environ)
package_root = os.path.dirname(os.path.dirname(os.path.abspath(jt.__file__)))
env["PYTHONPATH"] = os.pathsep.join(
    [package_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
```

漏了这一步，子进程导入的是 site-packages 里那棵树，**测试恒绿且毫无意义**。

## 2. 找到当前生效的 gen/ 目录

`pyjt_compiler.compile()` 每次 import 都无条件重写 `gen/*.cc`，所以不用手工删缓存；
真正的坑是缓存里同时存在多个 `gen/`，很容易 grep 到过期的那个。路径形如：

```
$JITTOR_HOME/.cache/jittor/jt<ver>/<gcc>/<py>/<os>/<cpu>/<源码哈希>/<git 分支>/<cuda key>/gen
```

- **源码哈希**变了会换一整棵目录：改完源码后旧目录仍在，别再去看它。
- **git 分支**是路径的一段：换分支 = 换目录。
- **cuda key 只在带 nvcc 时才有**：`nvcc_path=""`（CPU-only）生成到 `<分支>/gen`，
  带 nvcc 生成到 `<分支>/<cuda key>/gen`。两者是不同的两份，都要各自确认。

取「最近写入的那个」最可靠：

```bash
find "$JITTOR_HOME/.cache/jittor" -name 'pyjt_*.cc' -printf '%T@ %p\n' \
  | sort -n | tail -1 | cut -d' ' -f2-
```

改完生成器后，**先 grep 生成结果、后跑测试**：

```bash
grep -n 'YOUR_NEW_MARKER' "$GEN/pyjt_jit_op_maker.cc"   # 没有就是在测旧绑定
```

`pyjt_jit_op_maker.cc` 是 `var_holder.h` + `jit_op_maker.h` 合并生成的，
`Var` 上的方法都在这里；类型对象（`tp_*` 槽位）也在这里。

## 3. 改生成器：先在沙箱里 diff 生成结果，再重编

`pyjt_compiler.py` 只依赖 `jittor_utils`，不依赖编译好的核心，所以生成器可以单独跑。
用本目录的 `render_bindings.py` 把两份源码（或同一份改动前后）各渲染一遍再 diff，
几秒钟就能看清改动对生成的 C++ 做了什么，不用等重编：

```bash
GENH=$(find "$JITTOR_HOME/.cache/jittor" -name jit_op_maker.h | head -1)
python render_bindings.py <改动前的 repo>/python /tmp/gen_old "$(dirname $GENH)"
python render_bindings.py <改动后的 repo>/python /tmp/gen_new "$(dirname $GENH)"
diff -u /tmp/gen_old/pyjt_nano_vector.cc /tmp/gen_new/pyjt_nano_vector.cc
```

（`jit_op_maker.h` 本身是生成出来的，只存在于缓存里；脚本会像 compiler.py 那样把
`var_holder.h` 拼在它前面，`Var` 的方法和类型对象才会出现在输出里。）

先在小文件（`pyjt_nano_vector.cc`、`pyjt_py_ring_buffer.cc`）上看结构对不对，再看
`pyjt_jit_op_maker.cc`（3 万多行，所有算子）。一个很便宜的整体检查是花括号配平：

```bash
for f in /tmp/gen_new/*.cc; do
  echo "$(basename $f) $(( $(tr -cd '{' < $f | wc -c) - $(tr -cd '}' < $f | wc -c) ))"
done   # 全是 0
```

## 4. 会段错误的用例：用子进程判返回码

绑定层的 bug 有相当一部分表现为段错误（CPython 协议用错、在未构造对象上跑析构、
读已释放内存）。直接写进 pytest 会把整个 session 带走，看起来像「测试崩了」而不是
「测试失败」。写法是子进程 + 返回码：

```python
proc = run_in_subprocess("""
    try:
        jittor_core.RingBuffer()
    except Exception:
        print("RAISED")
    print("SURVIVED")
""")
output = proc.stdout.decode("utf8", "replace")
self.assertEqual(proc.returncode, 0, output)   # 段错误时非 0
self.assertIn("SURVIVED", output)
```

判据是三条一起看：**返回码为 0**、**打印了 SURVIVED**、**抛的是 Python 异常**。
只判返回码不够——jittor 的段错误处理器会打印 backtrace 后走 `exit(1)`，
也有它自己 catch 住而进程正常退出的情况。

## 5. C++ 改动的重编成本

改 `python/jittor/src/**` 或 `pyjt_compiler.py` 之后，**每个新进程**都要重编一次
`jittor_core`（分钟级）。所以：把一批改动攒起来一次验证，不要改一行跑一次。
CPU 与 CUDA 是两份缓存，两边都要跑一次；CPU-only 用
`JITTOR_TEST_DEVICES=cpu nvcc_path=""`，快很多。

## 6. 选测试文件时别把整个进程翻成 torch 模式

`tests/conftest.py` 会看你在命令行上点了哪些文件：只要其中任何一个属于
`tests/_helpers/process_modes.py` 的 `TORCH_MODE_PATHS`（`tests/core/test_type_system.py`、
`tests/core/test_regression.py`、`tests/ops/test_ops.py`、`tests/compat/torch/…` 等），
它就给**整个 pytest 进程**设上 `JITTOR_TORCH_SHIM=1`。torch 模式改的是全局语义
（惰性执行、归约默认值、梯度语义、`finfo`/`iinfo` 的形状），于是同一条命令里的原生
用例会成片地假失败——症状是 `TypeError: all() got ...`、`'finfo' object has no attribute`、
`number_of_hold_vars` 对不上这类与你的改动毫无关系的错。

判据：**同一个文件单独跑通、和别的文件一起跑就挂**，先查是不是混进了 torch 模式路径。
把 torch 模式的文件单独起一条命令跑。

## 7. 类型对象改了布局时要额外确认的事

给生成的类型加字段（改 `tp_basicsize`）或改 `tp_flags` 之后，除了新测试还要跑：

- `tests/core/test_var.py`、`tests/core/test_array.py`（`VarHolder` 的创建与释放路径）
- `tests/compiler/test_ring_buffer.py`、`tests/compiler/test_ring_buffer2.py`
- `tests/core/test_pyjt_binding_protocol.py`（本层的协议用例都在这里）

`VarHolder` 的 PyObject 不只从 `tp_new` 来：`py_converter.h` 里的 `to_py_object`
用 `_PyObject_New` 直接建，**那块内存不清零**。任何依赖「新对象字段为 0」的设计
都必须同时改这条手写路径。

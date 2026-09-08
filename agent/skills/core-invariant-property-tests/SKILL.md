---
name: core-invariant-property-tests
description: 给核心 C++（图、liveness 记账、执行器计划）写属性测试而不是点用例：放哪里最便宜且门禁能跑到、怎么造合成图绕开 JIT 编译器、为什么绝对计数不能断言而要扫描增量、以及属性测试抓到真缺陷时怎么落地（strict xfail 加一条「不许变多」）。要给 tests/core 加测试、要测执行器/图不变量、或者手上有一条「lived_vars N != 0」这类查不动的失败时读这一篇。
---

# 核心不变量的属性测试

## 0. 先量，别按行数加测试

`gate-tier-budget` §2.1 立过「行数不是耗时」。这里是同一个错误的第二种形态：
**行数也不是覆盖**，而且这一种更贵——它让「多写 tests/core 的行」看起来像个答案。

实测（`tools/measure_core_test_balance.py`，2026-09-06）：`tests/core` 从审计记的 7355 行
涨到 **14374 行**，比核心 C++（44934 行）的比值从 0.20 涨到 0.32。听起来在变好。
按文件实际点名的 API 归类之后：

| 主题 | 文件 | 行 | 用例 |
| --- | --- | --- | --- |
| dtype 与数值 | 69 | 12918 | 684 |
| autograd | 41 | 8933 | 480 |
| 进程与构建 | 40 | 7214 | 375 |
| 绑定 | 21 | 5207 | 284 |
| **图与 liveness** | **10** | **2328** | **141** |
| **执行器** | **7** | **1397** | **68** |

翻倍长出来的七千行几乎全在数值面上。**「这个目录多大」和「这个目录测了什么」是两个
独立的问题，而前者是免费就能看到的那个。** 加测试之前先跑一次归类，否则你会往已经
最厚的那一层再加一层。

脚本还报一个下界：179 个核心头文件里有多少个在整个 `tests/` 加 `src/tests` 里
**没有被任何文件点到名**（当时 14 个）。反过来读不成立——点到名字不等于有断言，
所以那 14 个是「肯定没测」，剩下的不是「已测」。本波抓到的真缺陷正好落在这 14 个
里的 `src/ops/tape_op.h`，这不是巧合。

## 1. 最便宜的位置：`src/tests/*.cc` 里的 `JIT_TEST`

**这个位置存在，而且大多数人不知道。** `compiler.py` 的 `gen_jit_tests()` 扫
`src/**/*.cc` 里的 `JIT_TEST(name)`，生成 `// @pyjt(name)` 绑定，于是

- C++ 里写 `JIT_TEST(foo) { ... }`
- 自动变成 `jt.tests.foo`
- 自动变成 pytest 节点 `tests/codegen/test_jit_tests.py::TestJitTests::test_foo`
- `tests/compiler` 已在原生 CPU 门禁里（0.04 之后门禁可达全树），所以**不用改任何
  门禁配置**

成本实测：新增 9 个执行器属性 + 5 个 liveness 属性共 14 条，`--durations=0` 全部
落在 5 ms 以下，整文件 15 条 **2.49 s**，其中几乎全是解释器启动与 import。
**边际层内成本约 0.05 s。** 没有设备、没有分配、没有 kernel。

对比：同一批性质如果从 Python 走真实算子去测，要编 kernel、要缓存、要设备，
一条就是秒级，而且**排序 bug 会和算术 bug 长得一样**。

代价只有一个：加 `src/` 文件会重编核心（本波 201 → 202 个 TU，约 2 分钟一次）。
所以别为了一条断言新建文件，按主题合并。

### 1.1 合成图：不要用真实算子

要测 Planner/执行器，就得有图。用 `binary`/`reduce` 建图会把 JIT 编译器拖进来。
Planner 只问算子三件事：`type()`、边、`order()`。所以造一个只提供这三件事的探针算子：

```cpp
struct PlanProbeOp final : Op {
    PlanProbeOp(NanoVector shape, OpType type) {
        set_flag(OpFlags::_cpu);
        set_flag(OpFlags::_cuda);
        // 否则 Op::init() 会给每条边打 _needed_by_backward，
        // 把 liveness 机制拖进一个只关心顺序的测试
        set_flag(OpFlags::_manual_set_vnbb);
        set_type(type);
        create_output(shape, ns_float32);
    }
    const char* name() const override { return "plan_probe"; }
    void infer_shape() override {}   // 形状在构造时定好
};
```

接线照 `src/tests/test_grad_snapshot.cc` 的 `make_snapshot_grad_op`：
`create_output` → `set_inputs({op})` → `VarPtr(move(outputs_holder[0]))` →
`op->set_inputs(nodes)` → `op->init()`。

**一个坑**：`count_fuse` 对批内非边界 var 会 deref `var->input()`，所以
**批里每个 var 都必须有批内的生产者**。叶子 var 直接 `VarPtr a({4},"float32")`
建出来没有生产者，BFS 会把它收进来然后空指针。解法是让叶子也由一个零输入的探针算子产出。

### 1.2 断性质，不断快照

`ExecPlan` 是值类型，很容易想「录一份计划下来逐字段比」。**不要**——`count_fuse`
有权改主意什么该融合，录快照会把启发式冻住，于是一次正常的启发式调整变成一片红。

该断的是「计划自身一致且按它自己声明的顺序可执行」：

- 所有下标在 `ops`/`all_vars` 范围内；`range.size()==queue.size()`、
  `range.back()==fuse_ops.size()`、单调且无空段
- 批编号自洽：`ops[i]->batch_index_at(stamp)==i`
- 每个算子至少被排进一个段（**是「至少」不是「恰好」**：喂多个段的算子会被复制进
  每个段，而不是切出来）
- 段间顺序是段图的拓扑序；段内顺序是该段依赖的拓扑序
- 每个批内 var 恰好被一个批内算子产出；没有生产者的必须是 `var_fused==1` 的边界
- 建计划不执行任何东西（`mem_ptr` 仍为空、没有 var 变 finished、活算子数不变）
- 同一张图建两次给出相同的计划（`stamp` 除外）

**读 `batch_index_at` 必须在 `plan.epoch` 还活着的时候**，检查完再 `plan.epoch.reset()`。

## 2. 绝对计数不能断言，要扫增量

`tests/core` 里那几条「存活 Var 数等于 N」在并发下不是稳定量：3.01 实测同一份源码
两次并发跑一次 9 一次 7，串行四轮才 md5 一致。**任何对绝对计数的断言都会变成抖动源。**

改成对**增量**断言，前序状态自动抵消：

```python
jt.clean(); gc.collect(); jt.clean()
before = jt.liveness_info()["lived_vars"]
shape()                      # 建图、执行、丢掉
gc.collect(); jt.clean()
after = jt.liveness_info()["lived_vars"]
assert after == before
```

`gc.collect()` 不是装饰：Python 侧的环（traceback、`Function` 把输入存在 `self` 上）
也会拖住 Var，那不是泄漏。少这一步会得到一堆假泄漏。

### 2.1 扫描要在**一个子进程**里跑，理由有两条

第一条是常规的隔离。**第二条更重要，而且容易漏**：如果某个形状真的泄漏，
**你的测试文件自己就变成了下一个跨文件抖动源**——正是它要测的那件事。

所以：一个子进程跑完整个「形状 × 模式」矩阵，打一行 JSON 出来，父进程解析后断言。
一个矩阵一次解释器启动（实测 1.38 s），比一条用例一个子进程便宜一个数量级。

父进程必须先断言**矩阵真的跑了**（条目数下界、两个模式都在）。空矩阵会让下面
每条断言都空转通过——这和 `test_jit_tests.py` 的 `_install_jit_tests` 拒绝装零个方法
是同一条教训。

## 3. 用差分形状定位，别猜机制

手上一条「`lived_vars 2 != 0`」是查不动的：它只说有两个 var 没回收，不说是谁。
属性测试的真正价值在这里——**把同一条性质扫过一组只差一个因素的形状**，
不泄漏的那些就是对照组。

本波实测的定位过程（每步都只改一个因素）：

| 形状 | 结果 |
| --- | --- |
| 两输出 `Function`，**其中一个** `stop_grad()`，都 sync | **泄漏 2** |
| 两输出，**两个都** `stop_grad()` | 干净 |
| 两输出，**都不** `stop_grad()` | 干净 |
| **单输出** `Function` + `stop_grad()` | 干净 |
| 两个**普通算子**各一输出，其一 `stop_grad()` | 干净 |
| 两输出 `jt.code`（原生多输出算子），其一 `stop_grad()` | 干净 |
| 两输出，其一 `stop_grad()`，**只 sync 一个** | 干净 |
| 两输出，其一 `stop_grad()`，**不执行** | 干净 |

八行读出来的结论比六条点用例强得多：触发条件是**经 `Function` 建的多输出算子
（即 `Tapes`）、其输出被 `stop_grad()` 的「有一些但不是全部」、且整批真的执行了**，
与 `lazy_execution` 无关。原生多输出算子干净，说明问题在 taped 路径而不是多输出本身。

**报机制之前先找到那条日志。** 这个泄漏的机制在 stderr 上是明写的：

```
[f] node.h:263 Check failed: value_ > 0   backward liveness release without a matching owner
```

即 `LivenessCounter<backward>::release()` 比 `own()` 多调了一次；抛出的异常被
`var_holder.cc` 的 teardown 路径吞掉，于是两个 Var 永久留在注册表里，
各自 `f=0 b=1`——**即各自 `need_free()==true` 却还活着**。

推广一条：**「账不平」的最好表述是「有节点说自己该被释放，却还在注册表里」**，
不是「计数等于 N」。前者指向机制，后者只报一个数。

### 3.1 让悬垂扫描说出它扫了多少

`jt.graph_check()` 会把这两个 Var 直接点名（`ERROR dnode ... Var(...:0:1:0:i0:o0...)`）。
但注意 6.C21 那个坑：`check_graph=1` 曾经在 release 构建下扫一张空表然后报成功。
修法留下的接口是**返回值**——`graph_check()` 返回扫过的节点数。

于是属性测试必须**同时**断两件事，否则区分不了「校验通过」和「跳过了」：

- 标志打开前 `graph_check()` 返回 **0**（注册表跟着标志走，之前建的节点不在里面）
- 标志打开后新建节点，`graph_check()` 返回 **> 0**

还要记住这个修法的既有限度：`swept` **不是**进程存活节点数。实测 10 扫过对 17 存活——
早于标志建的节点永远看不到。**别把一次干净的扫描读成全进程保证**，把这条差距也钉一条。

**推广：任何「检查器」都要能报出它检查了多少。** 一个永远说 OK 的检查器和不存在等价，
而前者更贵。这和 `gate-tier-budget` §5 对判据工具的那条是同一条。

## 4. 属性测试抓到真缺陷之后：两条测试，不是一条

属性测试抓到真的不变量违反是最有价值的产出，但**不能就这么把门禁弄红**——
一片人人都学会忽略的红等于没有门禁。也不能把它删掉或放宽成「允许泄漏」。

落地方式是**一对**测试：

```python
KNOWN_LEAKING_SHAPES = {"two_outputs_first_stopped/lazy=0": 2, ...}

def test_dropping_a_graph_leaks_nothing_new(self):
    """护门禁的那半：只要不多出来就绿。"""
    offenders = {n: d for n, d in deltas.items()
                 if d != KNOWN_LEAKING_SHAPES.get(n, 0)}
    assert offenders == {}

@pytest.mark.xfail(strict=True, reason="2.10: ... 修好之后这条会 XPASS 变红，
                                        强制回来更新记账")
def test_dropping_a_graph_leaks_nothing_at_all(self):
    """性质本身，不挖任何例外。"""
    assert {n: d for n, d in deltas.items() if d} == {}
```

两条各管一件事：

- 第一条是**门禁的防退化**：新形状开始泄漏、或者泄漏变多，红。它是绿的，所以没人
  需要学会忽略它。
- 第二条是**性质的真身**，`strict=True`。现在是 xfail；**owner 修好那天它 XPASS，
  strict 让 XPASS 算失败**，于是记账和那几条点用例被强制一起更新。这比在看板上留
  一行「待修」可靠，因为它不依赖谁记得。

再加一条**钉住机制**的：泄漏量必须恰好是每次 2（一次未匹配的 backward release
连带它已传播到的那一条出边）。这个数一动，说明机制不是原来那个，
归因得重新推而不是照抄。

## 5. 加了测试之后，证明没挤掉旧结论

`tools/gate_conclusion_diff.py compare` 默认把新增 nodeid 也当差异报，于是
**加测试的人只能用眼睛扫差异列表**——正是这个工具本来要替掉的习惯。

用 `--expect-new`（本波加的）把它变回退出码：

```bash
python tools/gate_conclusion_diff.py record --out base.json -- \
    tests/core tests/codegen tests/build --ignore=tests/core/test_my_new_file.py -q
python tools/gate_conclusion_diff.py record --out cand.json -- \
    tests/core tests/codegen tests/build -q
python tools/gate_conclusion_diff.py compare base.json cand.json \
    --expect-new 'tests/core/test_my_new_file.py::TestX::test_y' ...
```

被点名的 nodeid 允许新出现，其它一切照旧检查；**没点名就出现的仍然红**，
**点了名却没出现的也红**（陈旧的 `--expect-new` 会静默放宽比较）。
反向用例在 `tests/structure/test_gate_conclusion_record.py::TestExpectNew`：
一轮里既加了测试又丢了旧结论，即使把新增点了名也必须非零退出。

把 baseline 取成「同一次构建、同一条命令、只 `--ignore` 掉自己的新文件」而不是
「上一个提交」：后者会把冷缓存的差混进来（`gate-tier-budget` §1.2、§1.2bis）。
C++ 侧的新用例没法用 `--ignore`（它们和既有用例同在 `test_jit_tests.py`），
用 `--deselect` 逐条点名。

### 5.1 顺带修掉的：`--deselect` 曾经会被记成「丢了结论」

第一次这样做的时候，`compare` 报了 14 条 `COLLECTED BUT NO CONCLUSION`，
正好是那 14 个 `--deselect` 的 C++ 用例（`collected=1245 concluded=1231`）。
原因：插件用 `pytest_collection_modifyitems` 记 `collected`，注释写着 "After deselection"
**但它不是**——插件在 `pytest_configure` 里注册，因此**先于** pytest 自己对该钩子的实现跑，
而 `--deselect`/`-k`/`-m` 正是在那个实现里剔项的。

**这比记错数严重。** 那条 `collected - conclusions` 的检查存在的目的是抓「答案丢了」，
而一个每次 `--deselect` 都叫一遍的信号，会被它的读者学会跳过——和「永远说 IDENTICAL」
是同一类失效，只是方向相反。它还让串行记录和 xdist 记录对「collected」不是同一个意思
（xdist 那条钩子一直报 deselect 之后的 ids）。

已改成 `pytest_collection_finish` 读 `session.items`（跑在整条 `modifyitems` 链之后；
**xdist 下 controller 到这里 `items` 是空的，所以不能无条件覆盖**）。
回归 `tests/structure/test_gate_conclusion_record.py::TestDeselectionIsNotCollection`。

**推广一条**：任何「在 collection 钩子里记下要跑什么」的插件都要问一句
**我这个钩子跑在剔项之前还是之后**。想验证就跑一次带 `--deselect` 的会话，
看被剔掉的那条有没有出现在你的记录里。
（另一条：那个「deselect 与真丢结论仍可区分」的反向用例，**第一版真杀了一个 xdist worker，
xdist 会等它，花了 300 s 才超时**。要造「丢了一个结论」的记录，直接改 JSON，别杀进程。）

## 6. 第一次把一层测试接进门禁时，先假定它有红

把 `src/tests/*.cc` 接进 CPU 门禁，第一次跑就有 4 条红，全是别的分区新写的
native provider registry 用例。这不是意外，是**「零执行」这个审计结论的必然推论**：
一批测试从没被任何门禁跑过，就没有任何机制保证它们现在是绿的。所以接线之前
先把这件事排进计划，别把它当成「我的改动搞坏了什么」。

**第一步永远是排除自己。** 别人的红和自己造成的污染长得完全一样，尤其当你
新增的用例（比如注册了合成算子的 `PlanProbeOp`）按字母序排在人家前面、
跑在同一个进程里。两个机械判据，都便宜：

- **只跑那几条**（`-k`，把其余全部 deselect）。还红就与执行顺序无关。
- **`git diff <上游> HEAD -- <相关路径>` 为空**，证明你的提交没碰它们。

**第二步：隔离用 `xfail(strict=True)`，不要用 `skip`。** 差别不在这一刻，
在对方修好的那一天：`skip` 下你这边毫无反应，隔离条目就永久留着，
而且下一个人会以为那几条一直在跑；`strict` 下修好即 XPASS 变红，
逼着删掉隔离条目。**这和第 4 节对自己抓到的缺陷用同一个模式**——
唯一的区别是缺陷归属，处置方式不该因为「是别人的」而变松。

**隔离也要两个方向都验一次**，否则你只知道它没报红、不知道它还能不能报红：
把一条**本来就绿**的用例临时写进隔离表，它必须报 FAILED（XPASS strict）。
一次性的，验完删掉。

**再加一条防止隔离条目空指向的守卫**：隔离表里的名字必须都还存在于注册表里。
用例被改名或删掉之后，条目会变成一个谁也不标记的字符串，于是这张表
一边声称某处有缺陷、一边没有任何地方能复现它。

**隔离条目要写清失败的那一句，不是「坏了」。** 每条记下断言位置与失败的表达式
（例如「`test_op_register.cc:267`，观察者收到的事件过不了自己的 `event.valid()`」），
并注明为什么不在本波修（比如那个文件属别人的工作集）。修的人拿到的
是一个坐标，而不是一次重新调查。

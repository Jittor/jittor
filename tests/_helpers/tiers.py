"""Two tiers, one tree: what a pull request runs and what the nightly adds.

``gate_scope`` answers *which files a gate may run* (all of them). This answers
*which of them a pull request waits for*. The two are deliberately separate: a
test dropped from the fast tier is still gated, just later, and a test dropped
from ``gate_scope`` is not gated at all. Only the second is a hole.

The inversion is the same one 0.04 used, and for the same reason. The fast tier
is "the tree minus stated exceptions", so a new test file is in it the moment it
is written, and a slow one has to say -- here, in one place -- how long it takes
and why that is worth an entry. A hand-maintained include list drifts; this
cannot, because the default is inclusion.

Why seconds are written down
----------------------------
The tier's promise is a number ("a pull request gate under five minutes") and a
promise about a number needs a check. The obvious check -- assert the wall clock
of the run -- is the one thing this repository has learned not to do: an upper
bound on elapsed time fails on a loaded machine for reasons that have nothing to
do with the change, in exactly the way a real regression fails (see the
``load_sensitive`` marker). So the budget is checked *arithmetically* instead,
against measured costs recorded here, by ``tests/structure/test_gate_tiers.py``.
That check is a fact about the selection, not about the machine, and it is what
tells you the tier has drifted before the tier tells you by being slow.

The seconds are measurements, not estimates: ``--durations=0`` over a whole-tree
run on an idle machine, recorded with the run that produced them. They will
drift; the structure test asserts the arithmetic, not the accuracy, and a
re-measurement is a normal commit.
"""


#: ``(path, seconds, reason)`` -- a test file the fast tier does not run.
#:
#: ``path`` is a file, not a nodeid: node ids churn with every parametrisation
#: and a stale one silently selects nothing, which is the failure mode this list
#: exists to prevent. ``seconds`` is the file's measured total. ``reason`` says
#: what makes it slow, because "it is slow" is not reviewable -- a file that is
#: slow because it compiles two hundred kernels is a different decision from one
#: that is slow because it sleeps.
SLOW_FILES = (
    ("tests/ops/test_ops.py",
     2319,
     "OpInfo 电池组：227 个算子 x 每个算子的样本、dtype 与反向，每个组合一次 JIT 编译。它一个文件就是 torch 半边的 59%"),
    ("tests/data/test_dataset.py",
     412,
     "两条 worker 监管用例各自跑满子进程超时（300s + 90s），其余用例合计约 22 秒。缺陷是 6.P15 与 6.C31 叠加出来的，绑定分区在修；修好之后这一条应当删掉"),
    ("compat/tests/torch/test_torch_compat.py",
     319,
     "遗留的聚合兼容检查脚本，整套在一个子进程里跑，切不开也并行不了"),
    ("compat/tests/torch/test_einops.py",
     252,
     "einops 的全部重排/归约模式逐个对拍，每个模式一次 JIT 编译"),
    ("compat/tests/torch/test_torchmetrics_compat.py",
     134,
     "torchmetrics 的指标矩阵，逐指标建图"),
    ("tests/models/test_mmdet_ops.py",
     81,
     "mmdetection 的聚合兼容检查脚本，同样整套在一个子进程里跑"),
    ("tests/bindings/test_signal_and_teardown.py",
     64,
     "每条用例起一个会被信号杀死或中途退出的子进程；两条各约 31 秒，代价是子进程里的一次 jittor import"),
    ("tests/runtime/test_flag_env_and_setter.py",
     63,
     "逐个 flag 起一个新解释器验证环境变量解析——契约要求的就是新进程；两条各约 30 秒"),
    ("compat/tests/torch/test_torch_hf_alias.py",
     63,
     "HuggingFace 侧的别名表逐条走一遍真实调用"),
    ("tests/build/test_cold_start_runtime.py",
     59,
     "按定义要冷启动：每条用例在新进程里重编 jit_utils 与 core"),
    ("compat/tests/torch/test_torch_compat_conv_pool.py",
     53,
     "卷积与池化的形状/步长/padding 矩阵，每个组合一次 JIT 编译"),
    ("tests/ops/test_reduce_op.py",
     50,
     "归约算子的形状与 dtype 全组合，每个组合一次 JIT 编译"),
    ("tests/backends/parity/test_dtype_coverage.py",
     47,
     "整个 dtype 点阵上逐算子取值对拍；仅 test_binary_integer_widths 一条就 38 秒"),
    ("compat/tests/torch/test_torch_compat_loss.py",
     44,
     "全部损失函数 x reduction x 权重形状"),
    ("compat/tests/torch/test_torch_compat_reduce_shape.py",
     40,
     "归约的 dim/keepdim/空张量形状矩阵"),
    ("tests/codegen/test_merge_loop_var_pass.py",
     39,
     "MergeLoopVarPass 的多重 range 组合；单条 test_many_ranges_still_compute_the_right_values 就 32 秒"),
    ("tests/codegen/test_conv_tuner.py",
     38,
     "卷积调优器必须把候选实现逐个编出来才能比"),
    ("compat/tests/torch/test_torch_compat_fft_einsum.py",
     38,
     "FFT 与 einsum 的表达式矩阵"),
    ("compat/tests/torch/test_torch_compat_ops.py",
     37,
     "兼容层算子面的宽表"),
    ("compat/tests/torch/test_torch_compat_autograd.py",
     37,
     "自动微分语义矩阵，每条都要建反向图"),
    ("tests/nn/test_norm_unification.py",
     36,
     "归一化模块与函数式两条路径逐组合对拍；单条 test_module_and_functional_agree 就 24 秒"),
    ("compat/tests/torch/test_torch_compat_attention.py",
     34,
     "注意力的 mask/dtype/后端组合"),
    ("tests/optim/test_optimizer_save_load.py",
     34,
     "逐优化器存取一轮完整训练状态"),
    ("tests/bindings/test_pyjt_binding_protocol.py",
     33,
     "三条用例各起一个子进程验证绑定协议"),
    ("tests/nn/test_norm.py",
     31,
     "归一化的前向与反向数值稳定性，多形状多 dtype"),
    ("compat/tests/torch/test_torch_compat_sort_create.py",
     31,
     "排序与张量构造的宽表"),
    ("compat/tests/torch/test_torch_compat_optim.py",
     30,
     "优化器逐个跑若干步对拍"),
    ("tests/nn/test_nn_capabilities.py",
     30,
     "注意力/嵌入/稀疏的能力矩阵"),
    ("compat/tests/torch/test_torch_compat_math.py",
     30,
     "逐元素数学函数的宽表"),
    ("compat/tests/torch/test_torch_compat_indexing.py",
     29,
     "索引/切片/高级索引的组合矩阵"),
    ("compat/tests/torch/test_torch_compat_nn.py",
     29,
     "nn 模块面的宽表"),
    ("compat/tests/torch/test_torch_compat_scatter.py",
     28,
     "scatter/gather 的 reduce 模式与 dtype 组合"),
    ("compat/tests/torch/test_torch_compat_distributions.py",
     26,
     "分布对象逐个采样与对数概率对拍"),
    ("compat/tests/torch/test_torch_compat_pad.py",
     24,
     "padding 模式 x 维度组合"),
    ("tests/optim/test_optim_core.py",
     24,
     "优化器核心语义的全组合，18 条用例"),
    ("compat/tests/torch/test_torch_compat_norm.py",
     23,
     "范数与归一化的宽表"),
    ("compat/tests/torch/test_peft.py",
     20,
     "PEFT 适配器逐类型建图"),
)

# Every xdist worker inherits these pools. Keep the complete list beside the
# budget policy so nox and the standalone runner cannot silently diverge.
THREAD_POOL_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


#: Wall-clock budget for the fast tier, in seconds.
#:
#: Covers *both* process modes, because that is what a pull request waits for:
#: Torch compatibility mode is process-global, so the fast tier is two pytest
#: invocations one after the other and the budget has to buy both.
#:
#: **0.15 asked for "smoke < 5 minutes" and this is 8, so the gap is written
#: down rather than papered over.** Measured, the tier is 390 s (native 150 +
#: torch 240) and this arithmetic predicts 446 s. Five minutes is reachable
#: only by deferring about seven more files -- and those files are not slow.
#: ``test_torch_shim_aliases.py`` costs 12.3 s measured on its own and 174 s
#: measured inside the tier; what is over budget is not its work but its share
#: of eight cores. Deferring it would buy the number and lose the coverage for
#: a reason that is not about the file, which is exactly the dilution the
#: assertions below exist to prevent.
#:
#: The reason the tier cannot simply be given more workers is the same fact
#: 0.16 landed on from the other end: **this suite is already parallel inside
#: each test.** Jittor's OpenMP uses every core the process is allowed, so N
#: pytest workers divide cores that were already busy rather than adding any.
#: Measured speedup from 4 workers is 2.7x on the native half and 1.4x on the
#: torch half, not 4x. Closing the gap needs fewer or cheaper comparisons, or
#: more machines -- not a bigger ``-n``.
SMOKE_BUDGET_SECONDS = 480.0

#: Workers the fast tier is sized for. ``noxfile.GATE_WORKERS`` is the same
#: number and ``tests/structure/test_gate_tiers.py`` checks the budget against
#: it, so the three have to agree. It describes the CI runner the promise is
#: made about, not the biggest machine anyone has.
SMOKE_WORKERS = 4

#: What one whole-tree run cost, per process mode. Measured, with the run named
#: below, so the budget arithmetic has real numbers under it.
#:
#: **Two of these come from a serial run and two from the tier's own run, and
#: mixing them up is the mistake this comment exists to stop.** A file's cost
#: is not a property of the file: it depends on how many cores the process
#: running it was given. Measured, the same fast-tier files cost 407 s summed
#: in a serial run with all eight cores and 560 s summed inside the tier, where
#: each of four workers gets two -- and on the torch half, 287 s against 903 s.
#: So a prediction about the tier has to be fed numbers measured *in the tier*.
#: Using the serial figures predicted 254 s for a tier that measures 390 s, and
#: the budget assertion passed while it did.
#:
#: * ``total`` -- every test in that mode, summed from ``--durations=0``, in a
#:   serial run. **Used only to rank files for ``SLOW_FILES``**, where relative
#:   order is what matters and is not sensitive to the core count.
#: * ``fast_work`` -- what the files the fast tier keeps actually cost summed
#:   inside the tier, at ``SMOKE_WORKERS`` workers. This is what the makespan
#:   is computed from. It is *larger* than ``total`` minus the deferred
#:   seconds, never smaller; ``test_gate_tiers.py`` asserts that, so pasting
#:   the serial subtraction in here fails instead of quietly under-predicting.
#: * ``longest_fast_file`` -- the longest single file the fast tier keeps, also
#:   measured inside the tier. ``--dist loadfile`` cannot split a file, so this
#:   is a floor on the tier's wall clock however many workers it gets. Dividing
#:   a total by the worker count without it predicts three minutes for a tier
#:   holding a nine-minute file.
#: * ``startup`` -- interpreter start, jittor import and collection, paid once
#:   per invocation and not divisible by workers. Deliberately generous: the
#:   measured in-pytest share is about 9 s (native) and 13 s (torch), and the
#:   rest is the harness getting there. It is why the prediction (446 s) sits
#:   above the measurement (390 s), which is the direction a budget guard
#:   should err in.
MEASURED = {
    "native": {"total": 1449.0, "fast_work": 560.0,
               "longest_fast_file": 118.8,   # tests/core/test_setitem.py
               "startup": 40.0},
    "torch": {"total": 3927.0, "fast_work": 903.0,
              "longest_fast_file": 174.0,    # test_torch_shim_aliases.py
              "startup": 40.0},
}

#: Where MEASURED and the seconds in SLOW_FILES come from. Named so a
#: re-measurement can say what changed rather than only that it changed.
#:
#: Two whole-tree runs, 2026-09-03, `--durations=0`, one pytest process per
#: process mode, **warm Jittor cache**, `use_parallel_op_compiler=16`, eight
#: cores. Warm on purpose: a cold run measures the distribution of *compile*
#: time, and the tier is a promise about what a pull request waits for, which is
#: only reachable with the cache restored (see .github/workflows/cpu.yml).
#: The machine was shared and busy at the time (load average 15-20), so the
#: absolute seconds are an over-estimate; the ranking, which is what selects the
#: list, is not sensitive to that.
#:
#: native 1289 passed / 968 skipped / 1 failed / 1 xfailed in 1467 s;
#: torch   1855 passed / 550 skipped / 7 failed.
#: `fast_work` and `longest_fast_file` come from a second pair of runs the same
#: day: the fast tier as the gate runs it, `-n 4 --dist loadfile` with
#: `OMP_NUM_THREADS` split (`_home/gates/runs/smoke_s2_native.log`,
#: `smoke_s3_torch.log`), warm cache -- zero kernels compiled in either, so
#: none of the inflation above is compilation.
MEASURED_FROM = ("whole-tree serial runs of 2026-09-03 for the ranking; "
                 "the 4-worker fast-tier runs of the same day for the budget")

#: A 2026-09-06 re-measurement, recorded rather than substituted. Read this
#: before changing anything above.
#:
#: Same command the gate runs (``gate_scope`` selection, ``-m "not slow"``,
#: ``-n 4 --dist loadgroup``, thread pools split), warm cache, but on a
#: **sixteen**-core partition where each of four workers gets four threads --
#: not the eight-core shape the numbers above describe. Measured:
#:
#:   native  wall 406.3 s   fast_work 1592.9 s   longest 238.4 s (test_setitem.py)
#:   torch   wall  91.8 s   fast_work  328.2 s   longest  22.4 s
#:   total   wall 498.1 s
#:
#: Both halves are work-bound (1592.9/4 = 398.2 against a 406.3 s wall;
#: 328.2/4 = 82.1 against 91.8 s), so the makespan model above is the right
#: shape and only its magnitudes have moved. Feeding these numbers to
#: ``budget_report`` predicts 560 s, over ``SMOKE_BUDGET_SECONDS``, which would
#: make ``_enforce_smoke_budget`` refuse to start the tier.
#:
#: **They are deliberately not pasted in, for the reason the comment above
#: gives about serial figures**: a file's cost depends on how many cores its
#: process was given, and replacing eight-core numbers with sixteen-core ones
#: is the same category error in a new direction. The constants have to
#: describe the runner the promise is made about. What this measurement does
#: establish, independent of the machine, is the *composition*:
#:
#: * The native half is 81.6% of the tier's wall clock. Its work is
#:   ``tests/ops`` 31.8%, ``tests/core`` 27.3% (``test_setitem.py`` alone
#:   15.0%), ``tests/distributed`` 16.2%, ``tests/compiler`` 12.1%,
#:   ``tests/nn`` 9.5%.
#: * ``tests/structure`` is **9.4%** of the tier -- about 47 s of 498 s. It
#:   runs entirely in the torch half (``process_modes.TORCH_MODE_PATHS``), and
#:   there is not one structure test in the native half. Deferring the whole
#:   directory would buy roughly 47 s and reach about 451 s, so "the structure
#:   suite eats the smoke budget" is not true and tiering it is not the lever.
#: * Neither half is held up by one long file, so deferring individual slow
#:   files buys little: reaching 300 s needs the native half's work to fall
#:   from 1592.9 s to about 550 s, a 65% cut. Ordering cannot do that.
#:
#: Cold, the same command costs native 1697.3 s / 1535.2 s in two runs against
#: 406.3 s warm -- 2.4x to 4.2x. Any figure here without "warm" beside it is
#: unusable, and a rebase that touches ``python/jittor/src/**`` makes the next
#: run cold (measured: a 37-line change to ``backend.cc``/``.h`` recompiled
#: about 900 kernels).
MEASURED_2026_09_06 = {
    "native": {"wall": 406.3, "fast_work": 1592.9, "longest_fast_file": 238.4},
    "torch": {"wall": 91.8, "fast_work": 328.2, "longest_fast_file": 22.4},
    "conditions": ("warm cache; 16-core partition, 4 workers x 4 threads; "
                   "load average 13-18; gate_scope selection; "
                   "_home/gates/runs/w0155/smoke_before_{native,torch}.log"),
}



# --------------------------------------------------------------------------
# The core tier
# --------------------------------------------------------------------------
#
# A third selection, and the only one in this file that is an *include* list.
# The argument above -- "a hand-maintained include list drifts" -- is about a
# tier that gates the tree, where drift means a hole. This one gates nothing:
# it is the answer to "did I just break something fundamental", asked between
# edits, and its failure mode is that it stops catching things early while the
# fast and full tiers still catch them. That is a cost in minutes, not in
# coverage, and it buys the thing a per-edit check has to have: an answer in
# about a minute rather than seven.
#
# What belongs here is one file per fundamental, not one file per subsystem:
# the graph and its execution, the array boundary, laziness, autograd, the
# elementwise and broadcast paths, and on the compat side the frontend's
# identity, its dtype and promotion rules, autograd through it, and where a
# tensor is placed. What does not belong is anything whose failure the tier
# above would report just as clearly ten minutes later.
#
# The seconds are measured the same way as the rest of this file --
# ``--durations=0``, summed per file, serial -- and serial is also how the tier
# is meant to run: it is small enough that workers buy nothing, and it stays
# runnable on a checkout with no pytest-xdist.

#: ``(path, seconds, why)``. Measured 2026-09-18, serial, warm cache, this box.
CORE_FILES = (
    # ---- native: the graph, and what every other test stands on -----------
    ("tests/core/test_core.py", 1.2,
     "融合、内存优化与 hold/lived 计数——执行器的核心不变量"),
    ("tests/core/test_array.py", 4.2,
     "numpy 与 Var 的边界：dtype、连续性、分配器"),
    ("tests/core/test_fused_op.py", 0.3,
     "融合算子自身的装载与输出分类"),
    ("tests/core/test_clone.py", 1.4,
     "clone 与 stop_grad 的图切分，以及它们对存活计数的影响"),
    ("tests/autograd/test_grad.py", 0.6,
     "反向的基本契约：梯度的形状、dtype 与断开时的零梯度"),
    ("tests/autograd/test_autograd_engine.py", 1.1,
     "反向引擎的图遍历与广播梯度"),
    ("tests/ops/test_binary_op.py", 9.3,
     "逐元素路径与 dtype 提升的全组合；本层最贵的一条,但它覆盖的是所有算子的公共底座"),
    ("tests/ops/test_broadcast_to_op.py", 1.0,
     "广播——7e83d6da 之后它同时是存储描述符,融合的边界条件都在这里"),
    ("tests/nn/test_linear.py", 0.1,
     "matmul + bias 这条最常走的组合路径"),
    ("tests/runtime/test_runtime_device_state.py", 0.1,
     "设备与流的绑定状态"),

    # ---- torch mode: the frontend's identity and its numeric contract -----
    ("compat/tests/torch/test_independent_frontend.py", 16.9,
     "独立 Tensor 类型的身份：它是什么、原生 Var 不是什么"),
    ("compat/tests/torch/test_torch_compat_nn.py", 7.4,
     "nn 模块族的前向与参数归属"),
    ("compat/tests/torch/test_torch_compat_autograd.py", 3.8,
     "经由 shim 的反向语义"),
    ("compat/tests/torch/test_install_context.py", 3.2,
     "安装上下文与事务：装了什么、能不能回滚"),
    ("compat/tests/torch/test_torch_compat_indexing.py", 12.4,
     "索引、切片与 setitem——前端最常走的一族，且全部可在 CPU 上执行"),
    ("compat/tests/torch/test_torch_compat_dtype.py", 1.4,
     "dtype 对象与 torch 的精确对应"),
    ("compat/tests/torch/test_torch_compat_promotion.py", 1.2,
     "提升点阵与 result_type/can_cast,对的是 c10 的文档规则"),
    ("compat/tests/torch/test_tensor_state.py", 0.1,
     "张量状态与 requires_grad 的读写"),

    # compat 有两种测试,两种都要在这一层里有一条。上面是行为
    # (`compat/tests/torch`,在 Torch 模式下真的跑张量);下面是结构契约
    # (`compat/tests/structure`,断言谁拥有哪个名字、发布到哪个命名空间)。
    # 结构那类不跑数值,却是唯一能在安装期就抓到「装错地方」的检查,而且便宜。
    ("compat/tests/structure/test_torch_compat_structure.py", 3.0,
     "命名空间归属与 sys.modules 发布的白名单:谁被允许写进 torch 这个名字"),
    ("compat/tests/structure/test_compat_layering.py", 0.5,
     "分层方向:compat 可以依赖 jittor,反过来不行"),
)

#: Wall-clock budget for the core tier, in seconds, covering both process
#: modes. Predicted arithmetically like the fast tier's, and checked by
#: ``tests/structure/test_gate_tiers.py``; see CORE_STARTUP for the constant
#: that is not divisible by anything.
CORE_BUDGET_SECONDS = 120.0

#: Interpreter start, jittor import and collection, per invocation. Generous on
#: purpose, for the same reason ``MEASURED[...]["startup"]`` is: the prediction
#: should sit above the measurement, not below it. Measured here: the native
#: half ran 20.9 s of work in 24.0 s wall.
CORE_STARTUP = 15.0


def _runnable(path):
    """The spelling pytest can be pointed at; see ``gate_scope.runnable``."""
    from _helpers.gate_scope import runnable
    return runnable(path)


def core_paths(session=None, runnable=True):
    """The core tier's files, for one process mode or both.

    ``runnable=False`` gives the canonical repository paths, which is what a
    structure check compares against the gate's own selection.
    """
    selected = tuple(path for path, _seconds, _why in CORE_FILES
                     if session is None or session_of(path) == session)
    return tuple(_runnable(path) for path in selected) if runnable else selected


def core_seconds(session=None):
    return sum(seconds for path, seconds, _why in CORE_FILES
               if session is None or session_of(path) == session)


def predicted_core_seconds():
    """Both sessions, serial, plus one startup each."""
    sessions = {session_of(path) for path, _s, _w in CORE_FILES}
    return core_seconds() + CORE_STARTUP * len(sessions)

def slow_paths():
    return tuple(path for path, _seconds, _reason in SLOW_FILES)


def is_slow(relative_path):
    """Whether ``relative_path`` (posix, repo-relative) is out of the fast tier."""
    return relative_path in slow_paths()


def slow_seconds():
    return sum(seconds for _path, seconds, _reason in SLOW_FILES)


def worker_thread_budget(workers, available=None):
    """``OMP_NUM_THREADS`` for one worker when the gate runs ``workers`` of them.

    Jittor already defaults OpenMP to one thread per *physical core it is
    allowed to use* -- it reads the affinity mask, so ``taskset -c 104-111``
    gives 8, not the machine's 64. What it cannot know is that three other
    pytest workers are doing the same thing on the same eight cores. Each
    worker then starts eight OpenMP threads, and the gate runs 4x
    oversubscribed: measured, that is not a wash but a large loss, because the
    cost of an OpenMP barrier grows with the thread count while the work per
    thread shrinks.

    So the parallelism has to be split once, at the top, rather than claimed
    twice. This is the same fact the ``0.14`` task states from the other end:
    a gate that does not say how many threads it wants is not reproducible.

    ``None`` means "leave the default alone" -- a single-process gate should not
    have its threads cut.
    """
    if not workers or workers <= 1:
        return None
    import os

    if available is None:
        available = effective_cpu_count()
    return max(1, available // workers)


def apply_worker_thread_budget(environment, workers, available=None):
    """Apply the shared worker budget to every supported thread pool.

    Both the nox sessions and ``tools/run_test_suite.py`` call this helper so
    an added pool cannot be constrained in one entry point and leaked in the
    other.  The mapping is updated in place and returned for nox's copy-on-
    write environment convention.
    """
    budget = (worker_thread_budget(workers) if available is None
              else worker_thread_budget(workers, available=available))
    if budget is None:
        return environment
    for name in THREAD_POOL_ENV_NAMES:
        environment[name] = str(budget)
    return environment


def runtime_workers(configured_workers=None, available=None):
    """Return the xdist workers a gate can actually start.

    Keep this policy in the shared tier module so the nox session and the
    standalone budget report cannot disagree about the prediction.  ``None``
    uses the checked-in gate size; ``available`` is injectable for tests.
    """
    if configured_workers is None:
        configured_workers = SMOKE_WORKERS
    if (isinstance(configured_workers, bool)
            or not isinstance(configured_workers, int)
            or configured_workers < 1):
        raise ValueError("configured_workers must be a positive integer")
    if available is None:
        available = effective_cpu_count()
    if (isinstance(available, bool) or not isinstance(available, int)
            or available < 1):
        raise ValueError("available must be a positive integer")
    return max(1, min(configured_workers, available))


def effective_cpu_count():
    """Return CPUs the gate can actually consume, including cgroup quota.

    ``sched_getaffinity`` describes the cpuset, but a container may impose a
    smaller CFS quota on that set.  Ignoring the quota starts one OpenMP team
    per xdist worker with too many threads and inflates the worker-work term.
    A missing or malformed cgroup file is deliberately ignored: affinity is
    the portable baseline and remains the behavior on ordinary hosts.
    """
    import math
    import os

    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover - non-Linux
        affinity = os.cpu_count() or 1
    quota_paths = (
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    )
    period_paths = (
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    )
    try:
        with open(quota_paths[0], encoding="ascii") as stream:
            fields = stream.read().split()
        if fields and fields[0] != "max":
            quota, period = float(fields[0]), float(fields[1])
            if quota > 0 and period > 0:
                return max(1, min(affinity, int(math.ceil(quota / period))))
    except (OSError, ValueError, IndexError):
        try:
            with open(quota_paths[1], encoding="ascii") as quota_stream:
                quota = float(quota_stream.read().strip())
            with open(period_paths[1], encoding="ascii") as period_stream:
                period = float(period_stream.read().strip())
            if quota > 0 and period > 0:
                return max(1, min(affinity, int(math.ceil(quota / period))))
        except (OSError, ValueError, IndexError):
            pass
    return max(1, affinity)


def session_of(path):
    """Which of the two process modes runs this file. Not a choice: Torch
    compatibility mode is process-global, so the path decides (``gate_scope``)."""
    from _helpers.process_modes import TORCH_MODE_PATHS

    return "torch" if path.startswith(TORCH_MODE_PATHS) else "native"


def _slow_seconds_in(session):
    return sum(seconds for path, seconds, _reason in SLOW_FILES
               if session_of(path) == session)


def predicted_session_seconds(session, workers=None):
    """What the fast tier should cost in one process mode.

    ``max(work / workers, longest single file)`` is the standard makespan bound
    for a list scheduler that cannot split a job, and ``--dist loadfile`` is
    exactly that. Plus the startup nobody parallelises away.
    """
    workers = workers or SMOKE_WORKERS
    measured = MEASURED[session]
    work = measured["fast_work"]
    return max(work / float(workers), measured["longest_fast_file"]) \
        + measured["startup"]


def predicted_smoke_seconds(workers=None):
    """Both modes, one after the other -- that is what a pull request waits."""
    return sum(predicted_session_seconds(session, workers) for session in MEASURED)


def budget_report(workers=None, configured_workers=None):
    """Return an actionable, serialisable breakdown of the smoke budget.

    ``workers`` is the count xdist will actually start after runtime cgroup
    capping.  ``configured_workers`` is kept separately for diagnostics: a
    one-CPU container running a four-worker gate must say ``4 configured, 1
    actual`` rather than silently relabelling the run as a one-worker gate.
    """
    if workers is None:
        workers = runtime_workers(configured_workers)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if configured_workers is not None and (
            isinstance(configured_workers, bool)
            or not isinstance(configured_workers, int)
            or configured_workers < 1):
        raise ValueError("configured_workers must be a positive integer")
    configured_workers = (workers if configured_workers is None
                          else configured_workers)
    effective_cpus = effective_cpu_count()
    threads_per_worker = worker_thread_budget(workers, effective_cpus)
    sessions = {}
    for name, measured in sorted(MEASURED.items()):
        work_bound = measured["fast_work"] / float(workers)
        floor = measured["longest_fast_file"]
        predicted = max(work_bound, floor) + measured["startup"]
        sessions[name] = {
            "fast_work_seconds": measured["fast_work"],
            "work_bound_seconds": work_bound,
            "longest_file_seconds": floor,
            "startup_seconds": measured["startup"],
            "predicted_seconds": predicted,
            "bottleneck": "longest_file" if floor >= work_bound else "worker_work",
        }
    predicted = sum(item["predicted_seconds"] for item in sessions.values())
    return {
        "configured_workers": configured_workers,
        "workers": workers,
        "effective_cpus": effective_cpus,
        "threads_per_worker": threads_per_worker or effective_cpus,
        "budget_seconds": SMOKE_BUDGET_SECONDS,
        "predicted_seconds": predicted,
        "headroom_seconds": SMOKE_BUDGET_SECONDS - predicted,
        "sessions": sessions,
    }

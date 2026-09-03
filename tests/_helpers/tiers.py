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
    ("tests/compat/torch/test_torch_compat.py",
     319,
     "遗留的聚合兼容检查脚本，整套在一个子进程里跑，切不开也并行不了"),
    ("tests/compat/torch/test_einops.py",
     252,
     "einops 的全部重排/归约模式逐个对拍，每个模式一次 JIT 编译"),
    ("tests/compat/torch/test_torchmetrics_compat.py",
     134,
     "torchmetrics 的指标矩阵，逐指标建图"),
    ("tests/models/test_mmdet_ops.py",
     81,
     "mmdetection 的聚合兼容检查脚本，同样整套在一个子进程里跑"),
    ("tests/core/test_signal_and_teardown.py",
     64,
     "每条用例起一个会被信号杀死或中途退出的子进程；两条各约 31 秒，代价是子进程里的一次 jittor import"),
    ("tests/core/test_flag_env_and_setter.py",
     63,
     "逐个 flag 起一个新解释器验证环境变量解析——契约要求的就是新进程；两条各约 30 秒"),
    ("tests/compat/torch/test_torch_hf_alias.py",
     63,
     "HuggingFace 侧的别名表逐条走一遍真实调用"),
    ("tests/compiler/test_cold_start_runtime.py",
     59,
     "按定义要冷启动：每条用例在新进程里重编 jit_utils 与 core"),
    ("tests/compat/torch/test_torch_compat_conv_pool.py",
     53,
     "卷积与池化的形状/步长/padding 矩阵，每个组合一次 JIT 编译"),
    ("tests/ops/test_reduce_op.py",
     50,
     "归约算子的形状与 dtype 全组合，每个组合一次 JIT 编译"),
    ("tests/backends/parity/test_dtype_coverage.py",
     47,
     "整个 dtype 点阵上逐算子取值对拍；仅 test_binary_integer_widths 一条就 38 秒"),
    ("tests/compat/torch/test_torch_compat_loss.py",
     44,
     "全部损失函数 x reduction x 权重形状"),
    ("tests/compat/torch/test_torch_compat_reduce_shape.py",
     40,
     "归约的 dim/keepdim/空张量形状矩阵"),
    ("tests/compiler/test_merge_loop_var_pass.py",
     39,
     "MergeLoopVarPass 的多重 range 组合；单条 test_many_ranges_still_compute_the_right_values 就 32 秒"),
    ("tests/compiler/test_conv_tuner.py",
     38,
     "卷积调优器必须把候选实现逐个编出来才能比"),
    ("tests/compat/torch/test_torch_compat_fft_einsum.py",
     38,
     "FFT 与 einsum 的表达式矩阵"),
    ("tests/compat/torch/test_torch_compat_ops.py",
     37,
     "兼容层算子面的宽表"),
    ("tests/compat/torch/test_torch_compat_autograd.py",
     37,
     "自动微分语义矩阵，每条都要建反向图"),
    ("tests/nn/test_norm_unification.py",
     36,
     "归一化模块与函数式两条路径逐组合对拍；单条 test_module_and_functional_agree 就 24 秒"),
    ("tests/compat/torch/test_torch_compat_attention.py",
     34,
     "注意力的 mask/dtype/后端组合"),
    ("tests/optim/test_optimizer_save_load.py",
     34,
     "逐优化器存取一轮完整训练状态"),
    ("tests/core/test_pyjt_binding_protocol.py",
     33,
     "三条用例各起一个子进程验证绑定协议"),
    ("tests/nn/test_norm.py",
     31,
     "归一化的前向与反向数值稳定性，多形状多 dtype"),
    ("tests/compat/torch/test_torch_compat_sort_create.py",
     31,
     "排序与张量构造的宽表"),
    ("tests/compat/torch/test_torch_compat_optim.py",
     30,
     "优化器逐个跑若干步对拍"),
    ("tests/nn/test_nn_capabilities.py",
     30,
     "注意力/嵌入/稀疏的能力矩阵"),
    ("tests/compat/torch/test_torch_compat_math.py",
     30,
     "逐元素数学函数的宽表"),
    ("tests/compat/torch/test_torch_compat_indexing.py",
     29,
     "索引/切片/高级索引的组合矩阵"),
    ("tests/compat/torch/test_torch_compat_nn.py",
     29,
     "nn 模块面的宽表"),
    ("tests/compat/torch/test_torch_compat_scatter.py",
     28,
     "scatter/gather 的 reduce 模式与 dtype 组合"),
    ("tests/compat/torch/test_torch_compat_distributions.py",
     26,
     "分布对象逐个采样与对数概率对拍"),
    ("tests/compat/torch/test_torch_compat_pad.py",
     24,
     "padding 模式 x 维度组合"),
    ("tests/optim/test_optim_core.py",
     24,
     "优化器核心语义的全组合，18 条用例"),
    ("tests/compat/torch/test_torch_compat_norm.py",
     23,
     "范数与归一化的宽表"),
    ("tests/compat/torch/test_peft.py",
     20,
     "PEFT 适配器逐类型建图"),
)


#: Wall-clock budget for the fast tier, in seconds (0.15: "smoke < 5 minutes").
#:
#: Covers *both* process modes, because that is what a pull request waits for:
#: Torch compatibility mode is process-global, so the fast tier is two pytest
#: invocations one after the other and the budget has to buy both.
SMOKE_BUDGET_SECONDS = 300.0

#: Workers the fast tier is sized for. ``noxfile.GATE_WORKERS`` is the same
#: number and ``tests/structure/test_gate_tiers.py`` checks the budget against
#: it, so the three have to agree. It describes the CI runner the promise is
#: made about, not the biggest machine anyone has.
SMOKE_WORKERS = 4

#: What one whole-tree run cost, per process mode. Measured, with the run named
#: below, so the budget arithmetic has real numbers under it.
#:
#: * ``total`` -- every test in that mode, summed from ``--durations=0``.
#: * ``longest_fast_file`` -- the longest single file the fast tier keeps.
#:   ``--dist loadfile`` cannot split a file, so this is a floor on the tier's
#:   wall clock however many workers it gets. Dividing a total by the worker
#:   count without it predicts three minutes for a tier holding a nine-minute
#:   file.
#: * ``startup`` -- interpreter start, jittor import and collection, paid once
#:   per invocation and not divisible by workers.
MEASURED = {
    "native": {"total": 1449.0, "longest_fast_file": 17.0, "startup": 40.0},
    "torch": {"total": 3927.0, "longest_fast_file": 18.0, "startup": 40.0},
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
MEASURED_FROM = "whole-tree runs of 2026-09-03, warm cache, 8 cores, load 15-20"


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
        try:
            available = len(os.sched_getaffinity(0))
        except AttributeError:  # pragma: no cover - non-Linux
            available = os.cpu_count() or workers
    return max(1, available // workers)


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
    work = measured["total"] - _slow_seconds_in(session)
    return max(work / float(workers), measured["longest_fast_file"]) \
        + measured["startup"]


def predicted_smoke_seconds(workers=None):
    """Both modes, one after the other -- that is what a pull request waits."""
    return sum(predicted_session_seconds(session, workers) for session in MEASURED)

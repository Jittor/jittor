#!/usr/bin/env python3
"""In-process timer prelude for the CUDA backend benchmark.

`tools/bench_examples.py` launches an official example with this file as the
training script.  The example then runs unchanged through ``runpy`` while we
record synchronization-aware optimizer-step boundaries, optional DataLoader
wait time and the framework identity.  A JSON record is written to
``$BENCH_OUT`` even when the example raises, so a failed run still names its
end-to-end wall time and the exception.
"""
import json
import math
import os
import runpy
import statistics
import sys
import time
import traceback
import warnings

RUN_START = time.perf_counter()


def enabled(name):
    return os.environ.get(name, '0').lower() in ('1', 'true', 'yes')


def now():
    return time.perf_counter()


def configure_torch(torch):
    """Pin the native float32 path so both backends run the same arithmetic."""
    try:
        torch._dynamo.config.suppress_errors = False
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for owner, attribute, value in (
                (getattr(torch.backends, 'cuda', None), 'matmul.allow_tf32', False),
                (getattr(torch.backends, 'cudnn', None), 'allow_tf32', False),
                (getattr(torch.backends, 'cuda', None), 'matmul.fp32_precision', 'ieee'),
                (getattr(torch.backends, 'cudnn', None), 'conv.fp32_precision', 'ieee')):
            if owner is None:
                continue
            head, _, tail = attribute.partition('.')
            target = getattr(owner, head, None)
            try:
                setattr(target if tail else owner, tail or head, value)
            except Exception:
                pass


def patch_torchvision_video():
    """Keep datasets 4.0.0 importable with open-source torchvision 0.26.

    ``datasets.formatting.np_formatter`` imports ``VideoReader`` to type-check
    video columns.  Open-source torchvision 0.26 no longer ships that class
    (only the fb-only build does), so every tensorized text batch would raise
    ImportError.  The placeholder is never instantiated: the benchmark data are
    text and images.  The Jittor compatibility layer already fabricates this
    name, so the patch is a no-op there.
    """
    try:
        import torchvision.io as vision_io
    except Exception:
        return
    if not hasattr(vision_io, 'VideoReader'):
        vision_io.VideoReader = type('VideoReader', (), {})


COMPILE = []
COMPILED = []
ATTENTION = []


def install_compile_probe(torch):
    """Record every torch.compile call and what it returned.

    Native torch returns an Inductor ``OptimizedModule``; the Jittor shim
    returns its own ``OptimizedModule`` only for CUDA-graph modes and the
    module unchanged otherwise.  Recording the result distinguishes a real
    compilation from a silent identity/fallback.
    """
    original = torch.compile
    if getattr(original, '_bench_compile', False):
        return

    def compile(*args, **kwargs):
        result = original(*args, **kwargs)
        COMPILE.append({'backend': str(kwargs.get('backend')),
                        'mode': str(kwargs.get('mode')),
                        'result_type': type(result).__name__})
        COMPILED.append(result)
        return result

    compile._bench_compile = True
    torch.compile = compile


def install_pippy_compile(torch):
    """Compile the pippy model; accelerate only compiles prepare()d models."""
    mode = os.environ.get('BENCH_COMPILE_MODE')
    if not mode or os.environ.get('BENCH_EXAMPLE') != 'pippy':
        return
    try:
        import accelerate
        import accelerate.inference
    except Exception:
        return
    original = accelerate.inference.prepare_pippy
    if getattr(original, '_bench_compiled', False):
        return

    def prepare_pippy(*args, **kwargs):
        model = original(*args, **kwargs)
        return torch.compile(model, backend='inductor', mode=mode)

    prepare_pippy._bench_compiled = True
    accelerate.inference.prepare_pippy = prepare_pippy
    accelerate.prepare_pippy = prepare_pippy


def compilation_state():
    state = {'requested_mode': os.environ.get('BENCH_COMPILE_MODE'), 'calls': COMPILE}
    results = []
    for compiled in COMPILED:
        entry = {'type': type(compiled).__name__}
        replay = getattr(compiled, '__dict__', {}).get('_replay') \
            if hasattr(compiled, '__dict__') else None
        if replay is not None:
            entry['replay_stats'] = dict(getattr(replay, 'stats', {}))
            entry['replay_refused'] = str(getattr(replay, '_refused', None))
        results.append(entry)
    if results:
        state['results'] = results
    try:
        import torch._dynamo.utils as dynamo_utils
        state['dynamo_counters'] = {
            name: {key: int(value) for key, value in values.items()}
            for name, values in dynamo_utils.counters.items()}
    except Exception:
        pass
    try:
        import torch._inductor.metrics as metrics
        state['inductor'] = {
            name: int(getattr(metrics, name)) for name in (
                'generated_kernel_count', 'generated_cpp_vec_kernel_count',
                'ir_nodes_pre_fusion') if hasattr(metrics, name)}
    except Exception:
        pass
    return state


def install_attention_probe():
    """Record, and optionally select, transformers' attention implementation.

    jittor-compat's SDPA wrapper reads the attention mask back to the host on
    every forward, which can prevent Jittor's graph replay from capturing the
    step.  Eager attention has the same math and is available on both
    backends, so it is a fair paired configuration for isolating that cost.
    """
    eager = enabled('BENCH_EAGER_ATTENTION')
    try:
        import transformers
    except Exception:
        return
    for name in ('AutoModelForSequenceClassification', 'AutoModelForMaskedLM',
                 'AutoModelForCausalLM'):
        model_class = getattr(transformers, name, None)
        original = getattr(model_class, 'from_pretrained', None)
        if original is None or getattr(original, '_bench_attention', False):
            continue

        def from_pretrained(*args, __original=original, **kwargs):
            if eager:
                kwargs.setdefault('attn_implementation', 'eager')
            model = __original(*args, **kwargs)
            try:
                ATTENTION.append(str(model.config._attn_implementation))
            except Exception:
                pass
            return model

        from_pretrained._bench_attention = True
        model_class.from_pretrained = staticmethod(from_pretrained)


def synchronize(torch):
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def apply_pippy_single_rank(torch):
    """Make the pippy examples valid for one process.

    Every pippy example builds its measured input with a first dimension equal
    to the number of pipeline ranks (hard-coded ``2``).  On a single GPU the
    native pipeline validates the stage input against the example batch (1)
    and rejects batch 2, while the compatibility shim silently accepts it.
    Capping that dimension at the world size keeps the same workflow and the
    same input in both backends.
    """
    if not enabled('BENCH_PIPPY_SINGLE_RANK'):
        return
    world = int(os.environ.get('WORLD_SIZE', '1'))
    original = torch.randint
    if getattr(original, '_bench_single_rank', False):
        return

    def randint(*args, **kwargs):
        size = kwargs.get('size')
        positional = size is None and len(args) >= 3
        if positional:
            size = args[2]
        if isinstance(size, (tuple, list)) and len(size) == 2 and int(size[0]) > world:
            size = (world, size[1])
            if positional:
                args = [*args]
                args[2] = size
            else:
                kwargs['size'] = size
        return original(*args, **kwargs)

    randint._bench_single_rank = True
    torch.randint = randint


def install_step_timer(torch, marks):
    """Time the interval between user-visible optimizer-step calls."""
    try:
        from accelerate.optimizer import AcceleratedOptimizer as target
    except Exception:
        target = torch.optim.Optimizer
    original = target.step
    if getattr(original, '_bench_timed', False):
        return
    first = {'seen': False}

    def step(self, *args, **kwargs):
        synchronize(torch)
        moment = now()
        if first['seen']:
            marks['intervals'].append(moment - marks['last'])
        else:
            first['seen'] = True
            marks['first_step_seconds'] = moment - RUN_START
        marks['last'] = moment
        marks['calls'] += 1
        return original(self, *args, **kwargs)

    step._bench_timed = True
    target.step = step


def apply_num_workers_override(torch):
    """Diagnostic: force the same DataLoader worker count in both backends."""
    if not enabled('BENCH_FORCE_NUM_WORKERS'):
        return
    workers = int(os.environ.get('BENCH_NUM_WORKERS', '0'))
    try:
        from torch.utils.data import DataLoader as target
    except Exception:
        return
    original = target.__init__
    if getattr(original, '_bench_workers', False):
        return

    def __init__(self, *args, **kwargs):
        if 'num_workers' in kwargs:
            kwargs['num_workers'] = workers
        elif len(args) >= 6:
            args = [*args]
            args[5] = workers
        else:
            kwargs['num_workers'] = workers
        return original(self, *args, **kwargs)

    __init__._bench_workers = True
    target.__init__ = __init__


def install_data_timer(torch, marks):
    """Record the wall time each ``next(dataloader)`` call spends fetching."""
    try:
        from torch.utils.data import DataLoader as target
    except Exception:
        return
    original_iter = target.__iter__
    if getattr(original_iter, '_bench_timed', False):
        return

    class TimedIterator:
        def __init__(self, inner):
            self.inner = inner

        def __iter__(self):
            return self

        def __next__(self):
            start = now()
            try:
                return next(self.inner)
            finally:
                marks['fetch'].append(now() - start)

        def __len__(self):
            return len(self.inner)

    def timed_iter(self):
        return TimedIterator(original_iter(self))

    timed_iter._bench_timed = True
    target.__iter__ = timed_iter


def module_identity(name):
    module = sys.modules.get(name)
    if module is None:
        return None
    version = getattr(module, '__version__', None)
    return {'version': str(version) if version is not None else None,
            'file': getattr(module, '__file__', None)}


def framework_state(torch):
    state = {
        'torch': module_identity('torch'),
        'accelerate': module_identity('accelerate'),
        'transformers': module_identity('transformers'),
        'timm': module_identity('timm'),
        'datasets': module_identity('datasets'),
        'torchvision': module_identity('torchvision'),
        'jittor': module_identity('jittor'),
        'flags': {},
    }
    try:
        state['flags']['tf32_matmul'] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        pass
    try:
        state['flags']['tf32_cudnn'] = bool(torch.backends.cudnn.allow_tf32)
    except Exception:
        pass
    try:
        state['flags']['cuda_available'] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        state['flags']['max_memory_allocated'] = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    jittor = sys.modules.get('jittor')
    if jittor is not None:
        try:
            state['flags']['jittor_use_cuda'] = int(jittor.flags.use_cuda)
        except Exception:
            pass
        try:
            state['flags']['fallback'] = int(jittor.core.backend_fallback_count())
        except Exception:
            pass
        for name in ('cuda_kernel_math', 'use_parallel_op_compiler', 'enable_tuner'):
            try:
                value = getattr(jittor.flags, name)
                state['flags'][name] = value if isinstance(value, (int, float, str, bool)) else str(value)
            except Exception:
                pass
    accelerate = sys.modules.get('accelerate')
    if accelerate is not None:
        try:
            from accelerate.state import AcceleratorState
            storage = getattr(AcceleratorState._shared_state, '_storage', AcceleratorState._shared_state)
            state['accelerator_device'] = str(storage.get('device'))
            dynamo = storage.get('dynamo_plugin')
            state['dynamo_backend'] = str(getattr(dynamo, 'backend', None))
            state['dynamo_mode'] = str(getattr(dynamo, 'mode', None))
        except Exception:
            pass
    return state


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize(values, batch_size, warmup):
    steady = values[warmup:] if len(values) > warmup else values[-1:]
    median = statistics.median(steady) if steady else None
    return {
        'count': len(steady),
        'median_ms': None if median is None else 1000.0 * median,
        'p90_ms': None if not steady else 1000.0 * percentile(steady, 0.9),
        'mean_ms': None if not steady else 1000.0 * statistics.fmean(steady),
        'min_ms': None if not steady else 1000.0 * min(steady),
        'max_ms': None if not steady else 1000.0 * max(steady),
        'samples_per_second': None if not median else batch_size / median,
    }


def write_record(payload):
    target = os.environ.get('BENCH_OUT')
    if not target:
        return
    with open(target, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')


def main():
    import torch

    configure_torch(torch)
    patch_torchvision_video()
    install_compile_probe(torch)
    install_pippy_compile(torch)
    install_attention_probe()
    apply_num_workers_override(torch)
    apply_pippy_single_rank(torch)
    marks = {'calls': 0, 'intervals': [], 'fetch': [], 'last': None,
             'first_step_seconds': None}
    install_step_timer(torch, marks)
    if enabled('BENCH_TIME_DATA'):
        install_data_timer(torch, marks)
    script = os.path.abspath(sys.argv[1])
    sys.argv = [script, *sys.argv[2:]]
    status, error = 'ok', None
    try:
        runpy.run_path(script, run_name='__main__')
    except BaseException as caught:
        status = 'error'
        error = ''.join(traceback.format_exception_only(type(caught), caught)).strip()
        raise
    finally:
        end = now()
        intervals = marks['intervals']
        warmup = int(os.environ.get('BENCH_WARMUP', '5'))
        batch_size = int(os.environ.get('BENCH_BATCH', '0') or 0)
        record = {
            'schema': 'jittor-accelerate-bench/1',
            'example': os.environ.get('BENCH_EXAMPLE'),
            'variant': os.environ.get('BENCH_VARIANT', ''),
            'backend': os.environ.get('BENCH_BACKEND'),
            'device': 'cuda',
            'script': script,
            'status': status,
            'error': error,
            'in_process_wall_seconds': end - RUN_START,
            'first_step_seconds': marks['first_step_seconds'],
            'step_calls': marks['calls'],
            'step_intervals_seconds': intervals,
            'warmup_steps': warmup,
            'warmup_intervals_seconds': intervals[:warmup],
            'steady': summarize(intervals, batch_size, warmup),
            'batch_size': batch_size,
            'data_wait_seconds': marks['fetch'],
            'data_wait': summarize(marks['fetch'], batch_size, warmup),
            'framework': framework_state(torch),
            'attention_implementations': ATTENTION,
            'compilation': compilation_state(),
        }
        write_record(record)


if __name__ == '__main__':
    main()

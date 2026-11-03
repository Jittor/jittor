"""Bounded public-mask replay and first-fault observer for tiny PPO generation.

This is a diagnostic, not an indexing implementation or a workaround. Metadata
mode adds native graph entries/temporary holders but never reads tensor values.
Values mode deliberately materializes small inputs and each selected gather:
it can localize a fault, but changes timing/fusion and cannot clear a race.
"""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import importlib.machinery
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_stress_options(args):
    require(not args.boundary_stress or args.mode == 'mask', 'Boundary stress requires mask mode')
    require(not args.require_async or (args.mode == 'mask' and args.runtime == 'jittor'
            and args.device == 'cuda' and args.observe == 'off'),
            'Async coverage requires unobserved Jittor CUDA masks')
    require(not args.require_async or args.boundary_stress, 'Async coverage requires boundary stress')


def check_async_coverage(before, after, required):
    require(not required or after > before, 'No actual async worker batch was measured')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def external_output(root, output):
    root, output = root.resolve(), output.resolve()
    require(root != output and root not in output.parents, 'Output must be outside the repository')
    require(not output.exists(), 'Output already exists; preserve earlier evidence')
    output.mkdir(parents=True)
    return output


def check_indices(shape, indices):
    """Validate ordinary two-axis integer indexing before a device-side trap."""
    import numpy as np
    require(len(shape) == len(indices) == 2, 'Expected two explicit index tensors')
    for axis, (size, index) in enumerate(zip(shape, indices)):
        require(index.dtype in (np.dtype('int32'), np.dtype('int64')), 'Expected int32/int64 indices')
        require(index.size <= 64, 'Index observation exceeds the bounded scope')
        require(bool(np.all((index >= -size) & (index < size))),
                'Index out of bounds before native getitem on axis %d' % axis)


def genuine_oracle(torch):
    require(not hasattr(torch, '_torch_compat_install_context'), 'Independent genuine Torch oracle required')
    binary = Path(getattr(getattr(torch, '_C', None), '__file__', ''))
    require(binary.is_file() and any(str(binary).endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES),
            'Oracle must expose an actual compiled Torch binary')
    return dict(path=str(binary.resolve()), sha256=sha(binary))


def read_mask_reference(path, transformers_version):
    import numpy as np
    metadata = json.loads((path/'result.json').read_text())
    require(metadata.get('runtime') == 'torch' and metadata.get('genuine_oracle') is True
            and metadata.get('status') == 'complete', 'Reference is not a completed genuine oracle')
    require(metadata['versions']['transformers'] == transformers_version, 'Reference dependency mismatch')
    require(sha(path/'arrays.npz') == metadata['arrays_sha256'], 'Reference arrays hash mismatch')
    with np.load(path/'arrays.npz', allow_pickle=False) as archive:
        reference = {key: archive[key].copy() for key in archive.files}
    require(set(reference) == {str(n)+'-'+str(p)+'/'+kind for n in (3, 4, 5, 6)
                              for p in (0, 1) for kind in ('bool', 'eager')}, 'Reference array inventory mismatch')
    for name, value in reference.items():
        count = int(name.split('-')[0])
        require(value.shape == (1, 1, count, count) and
                value.dtype == np.dtype('bool' if name.endswith('/bool') else 'float32'),
                'Reference array shape/dtype mismatch')
    return reference


class IndexObserver:
    """Wrap only small two-index gathers from a one-row bool attention mask."""
    def __init__(self, torch, output, mode, runtime):
        self.torch, self.output, self.mode, self.runtime = torch, output, mode, runtime
        self.sequence = 0
        self.selected = 0
        self.stream = None

    def record(self, kind, **values):
        self.sequence += 1
        self.stream.write(json.dumps(dict(sequence=self.sequence, time=time.time(), kind=kind, **values))+'\n')
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def metadata(self, value):
        result = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                      stride=list(value.stride()))
        if self.runtime == 'jittor':
            result.update(native_id=int(value.id), var_pointer=int(value.var_ptr),
                          storage_pointer_now=int(value._mem_ptr_now), location=value.location(),
                          device_id=int(value.device_id), producer=value._producer_op())
        return result

    def selected_indices(self, target, index):
        if (str(target.dtype).split('.')[-1] != 'bool' or len(target.shape) != 2
                or int(target.shape[0]) != 1 or int(target.shape[1]) > 64
                or not isinstance(index, tuple) or len(index) != 2):
            return None
        if not all(self.torch.is_tensor(value) and int(value.numel()) <= 64
                   and str(value.dtype).split('.')[-1] in ('int32', 'int64') for value in index):
            return None
        return index

    def observed(self, target, index):
        indices = self.selected_indices(target, index)
        if indices is None:
            return self.original(target, index)
        self.selected += 1
        call = self.selected
        self.record('before_getitem', call=call, target=self.metadata(target),
                    indices=[self.metadata(value) for value in indices],
                    callsite=[dict(file=frame.filename, line=frame.lineno, function=frame.name)
                              for frame in traceback.extract_stack(limit=12)[:-1]])
        phase = 'metadata_recorded'
        try:
            if self.mode == 'values':
                values = []
                for axis, value in enumerate(indices):
                    phase = 'index_readback_%d' % axis
                    values.append(value.detach().cpu().numpy().copy())
                phase = 'index_range_check'
                self.record('index_values', call=call, values=[value.tolist() for value in values])
                check_indices(tuple(target.shape), values)
                phase = 'target_readback'
                expected = target.detach().cpu().numpy().copy()[tuple(values)]
            phase = 'gather_construction'
            result = self.original(target, index)
            phase = 'gather_constructed'
            dependencies = []
            if self.runtime == 'jittor':
                # Do not retain input holders beyond this statement. An empty
                # list can mean an auto-flush already executed the producer.
                dependencies = [self.metadata(value) for value in result._inputs()]
            self.record('after_getitem', call=call, result=self.metadata(result),
                        native_inputs=dependencies)
            if self.mode == 'values':
                import numpy as np
                phase = 'gather_readback'
                actual = result.detach().cpu().numpy().copy()
                require(actual.dtype == expected.dtype and np.array_equal(actual, expected),
                        'Gather differs from its actual input/index snapshots')
                self.record('checked_getitem', call=call, output=actual.tolist())
            return result
        except BaseException as error:
            self.record('first_failure', call=call, phase=phase, exception=type(error).__name__, message=str(error))
            raise

    def __enter__(self):
        self.stream = (self.output/'index-events.jsonl').open('x')
        self.original = self.torch.Tensor.__getitem__
        observer = self

        def wrapped(target, index):
            return observer.observed(target, index)

        self.torch.Tensor.__getitem__ = wrapped
        self.record('installed', mode=self.mode, runtime=self.runtime,
                    explicit_value_readback=self.mode == 'values',
                    caveat='Observation changes timing and graph entries; success cannot clear an intermittent failure')
        return self

    def __exit__(self, *exception):
        self.torch.Tensor.__getitem__ = self.original
        self.stream.close()


def bootstrap(args):
    sys.path.insert(0, str(args.root/'compat/tests/torch'))
    import _ecosystem_runner as runner
    torch = runner._import_torch(args.runtime)
    if args.runtime == 'torch':
        genuine_oracle(torch)
    require(importlib.metadata.version('transformers') == '4.56.2', 'Transformers pin mismatch')
    return runner, torch


def source_binding(args):
    names = ['compat/tests/torch/_ecosystem_runner.py']
    if args.runtime == 'jittor':
        names += ['src/ops/composite/getitem_op.cc', 'src/core/var_slices.h',
                  'python/jittor/build/codegen.py', 'compat/torch/installers/numerical/batching.py',
                  'python/jittor/nn/functional/loss.py', 'src/core/executor.cc',
                  'src/core/exec_runner.cc', 'src/runtime/async_exec.cc',
                  'src/runtime/backend.cc', 'src/runtime/backend_streams.cc',
                  'src/runtime/backend_streams.h']
    if args.mode == 'ppo':
        names += ['agent/skills/torch-compat-repo-runbook/scripts/trl_mainstream_probe.py']
    return dict(tool=sha(__file__), owners={name: sha(args.root/name) for name in names})


def run_masks(args, runner, torch):
    import numpy as np
    from transformers import masking_utils as mu
    require(args.repetitions >= 8 and args.repetitions <= 512 and args.repetitions % 8 == 0,
            'Repetitions must be a multiple of eight in [8, 512]')
    require(args.batch_size > 0 and args.repetitions % args.batch_size == 0, 'Batch size must divide repetitions')
    versions = dict(transformers=importlib.metadata.version('transformers'), torch=str(torch.__version__))
    reference = None
    if args.runtime == 'jittor':
        require(args.reference is not None, 'Candidate masks require an independent oracle reference')
    if args.reference:
        reference = read_mask_reference(args.reference, versions['transformers'])
    jt = __import__('jittor') if args.runtime == 'jittor' else None
    before = jt.core.backend_fallback_count() if jt else 0
    def batches():
        return int(jt.core.async_exec_batches()) if jt else 0
    async_before = batches()
    arrays, baseline, rows = {}, {}, []
    torch.manual_seed(0)
    with torch.no_grad():
        for start in range(0, args.repetitions, args.batch_size):
            pending, churn = [], []
            for iteration in range(start, start+args.batch_size):
                # Vary allocation and automatic-submission boundaries before
                # the temporary arange/reshape index grids are constructed.
                # No flags, explicit submissions or value observations change.
                if args.boundary_stress:
                    for _ in range((iteration * 17) % 121):
                        noise = torch.randn((1, 4), device=args.device, dtype=torch.float32)
                        churn.append(noise.square()+.125)
                count, padding = (3, 4, 5, 6)[(iteration % 8)//2], iteration % 2
                source = np.ones((1, count), dtype='bool')
                if padding:
                    source[0, 0] = False
                attention = torch.from_numpy(source).to(args.device)
                positions = torch.ones(count, dtype=torch.int64, device=args.device).cumsum(0)-1
                boolean = mu.sdpa_mask(batch_size=1, cache_position=positions, kv_length=count,
                    attention_mask=attention, allow_is_causal_skip=False, allow_torch_fix=False)
                eager = mu.eager_mask(batch_size=1, cache_position=positions, kv_length=count,
                    attention_mask=attention, dtype=torch.float32)
                pending.append((iteration, str(count)+'-'+str(padding), boolean, eager))
                # Ordinary work while temporary public-vmap index holders are
                # out of scope. No item/readback/explicit flush until batch end.
                for _ in range(16):
                    noise = torch.randn((1, 4), device=args.device, dtype=torch.float32)
                    churn.append(noise.square()+.125)
                left = torch.randn((32, 32), device=args.device)
                right = torch.randn((32, 32), device=args.device)
                churn.append(left@right)
            constructed_batches = batches()
            for iteration, case, boolean, eager in pending:
                for kind, tensor in (('bool', boolean), ('eager', eager)):
                    value, name = tensor.detach().cpu().numpy().copy(), case+'/'+kind
                    baseline.setdefault(name, value.copy())
                    require(value.dtype == baseline[name].dtype and np.array_equal(value, baseline[name]),
                            'Mask repeated output changed: '+name)
                    if reference is not None:
                        require(value.dtype == reference[name].dtype and np.array_equal(value, reference[name]),
                                'Mask differs from genuine Torch: '+name)
                    arrays[name] = value
                rows.append(dict(iteration=iteration, case=case, async_batches_at_construction=constructed_batches))
            for tensor in churn:
                require(bool(np.isfinite(tensor.detach().cpu().numpy()).all()), 'Ordinary churn became nonfinite')
    np.savez(args.output/'arrays.npz', **arrays)
    async_end = batches()
    check_async_coverage(async_before, async_end, args.require_async)
    return dict(status='complete', runtime=args.runtime, device=args.device, versions=versions,
                genuine_oracle=args.runtime == 'torch', repetitions=len(rows), rows=rows,
                oracle_binary=genuine_oracle(torch) if args.runtime == 'torch' else None,
                async_batches_before=async_before, async_batches_end=async_end,
                boundary_stress=args.boundary_stress, required_async_coverage=args.require_async,
                fallback_count=jt.core.backend_fallback_count()-before if jt else 0,
                arrays_sha256=sha(args.output/'arrays.npz'), masking_utils_sha256=sha(mu.__file__),
                reference_sha256=sha(args.reference/'result.json') if args.reference else None,
                scope='Public SDPA/eager masks, eight cases; delayed readback is not proof async worker ran')


def run_ppo(args):
    require(args.runtime == 'jittor' and args.device == 'cuda', 'PPO observer scope is Jittor CUDA')
    require(args.reference is not None and args.upstream is not None, 'PPO requires upstream and genuine oracle trace')
    producer = args.root/'agent/skills/torch-compat-repo-runbook/scripts/trl_mainstream_probe.py'
    argv = [str(producer), '--root', str(args.root), '--upstream', str(args.upstream),
            '--output', str(args.output/'trace'), '--runtime', 'jittor', '--device', 'cuda',
            '--algorithm', 'ppo', '--steps', str(args.steps), '--reference', str(args.reference)]
    previous = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(producer), run_name='__main__')
    finally:
        sys.argv = previous
    manifest = json.loads((args.output/'trace/manifest.json').read_text())
    require(manifest.get('status') == 'complete' and manifest.get('numerical_passed') is True
            and manifest.get('fallback_count') == 0 and manifest.get('optimizer_steps') == args.steps,
            'Observed PPO failed its unchanged producer contracts')
    return dict(status='complete', runtime='jittor', device='cuda', optimizer_steps=args.steps,
                numerical_passed=True, fallback_count=0, manifest_sha256=sha(args.output/'trace/manifest.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('mask', 'ppo'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--runtime', choices=('torch', 'jittor'), required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--observe', choices=('off', 'metadata', 'values'), default='off')
    parser.add_argument('--repetitions', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--boundary-stress', action='store_true',
                        help='Vary bounded pre-mask allocations without changing runtime flags')
    parser.add_argument('--require-async', action='store_true',
                        help='Reject a boundary-stress run unless the actual async worker executes')
    parser.add_argument('--steps', type=int, default=3)
    args = parser.parse_args(argv)
    require(args.mode != 'ppo' or args.observe != 'off', 'PPO mode requires an explicit observation level')
    check_stress_options(args)
    args.root = args.root.resolve()
    args.output = external_output(args.root, args.output)
    runner, torch = bootstrap(args)
    binding = source_binding(args)
    with ExitStack() as stack:
        runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        runner._configure_tf32(torch, args.device)
        observer = stack.enter_context(IndexObserver(torch, args.output, args.observe, args.runtime)) if args.observe != 'off' else None
        result = run_masks(args, runner, torch) if args.mode == 'mask' else run_ppo(args)
        if observer:
            require(observer.selected > 0, 'Observer selected no matching bool-mask gathers')
            result.update(observed_gathers=observer.selected, observer_mode=args.observe)
    require(binding == source_binding(args), 'Diagnostic/runtime owners changed during measurement')
    require(result['fallback_count'] == 0, 'Backend fallback occurred')
    result['source_binding'] = binding
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'runtime', 'device', 'fallback_count')}))


if __name__ == '__main__':
    main()

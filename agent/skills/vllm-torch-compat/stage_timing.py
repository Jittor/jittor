"""Diagnose warm vLLM host stages in independent oracle and Jittor processes.

Matches latency_acceptance.py's prompts and eager engine options. After warmup,
run a lightly instrumented stage pass, then a SEPARATE cProfile pass. Both are
diagnostic: neither replaces the uninstrumented performance acceptance. Times
are host wall time, including any waits already performed by vLLM; they are NOT
CUDA kernel time. No extra synchronization is inserted inside engine.step.
"""
import argparse
import cProfile
from contextlib import contextmanager
import functools
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pstats
import threading
import time
import traceback

from latency_acceptance import distribution


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class StageRecorder:
    """Record disjoint child spans inside the synchronous engine.step call."""

    def __init__(self, engine, core, runner):
        self.engine, self.core, self.runner = engine, core, runner
        self.current = None
        self.rows = []

    @contextmanager
    def installed(self):
        originals = []
        targets = ((self.core.scheduler, 'schedule', 'scheduler'),
                   (self.runner, 'execute_model', 'execute_model'),
                   (self.runner, 'sample_tokens', 'sample_tokens'),
                   (self.core.scheduler, 'update_from_output', 'scheduler_update'))
        try:
            for target, attribute, label in targets:
                original = getattr(target, attribute)
                existed = attribute in vars(target)
                previous = vars(target).get(attribute)

                @functools.wraps(original)
                def wrapped(*args, _original=original, _label=label, **kwargs):
                    if self.current is None:
                        return _original(*args, **kwargs)
                    assert threading.get_ident() == self.current['thread'], (
                        'stage ran on another thread; synchronous timing is unsupported')
                    started = time.perf_counter()
                    try:
                        result = _original(*args, **kwargs)
                    finally:
                        ended = time.perf_counter()
                        self.current['spans'].append((_label, started, ended))
                    if _label == 'scheduler':
                        scheduled = result.num_scheduled_tokens
                        prefill, decode = 0, 0
                        for request_id, count in scheduled.items():
                            request = self.core.scheduler.requests[request_id]
                            # vLLM 0.24 advances this count inside schedule().
                            before = request.num_computed_tokens - count
                            assert before >= 0, 'unexpected scheduler count contract'
                            prompt_count = min(count, max(0, request.num_prompt_tokens - before))
                            prefill += prompt_count
                            decode += count - prompt_count
                        self.current['scheduled_prefill_tokens'] += prefill
                        self.current['scheduled_decode_tokens'] += decode
                    return result

                originals.append((target, attribute, existed, previous))
                setattr(target, attribute, wrapped)
            yield
        finally:
            for target, attribute, existed, previous in reversed(originals):
                if existed:
                    setattr(target, attribute, previous)
                else:
                    delattr(target, attribute)

    def step(self):
        row = dict(thread=threading.get_ident(), spans=[],
                   scheduled_prefill_tokens=0, scheduled_decode_tokens=0)
        assert self.current is None
        self.current = row
        started = time.perf_counter()
        try:
            outputs = self.engine.step()
        finally:
            ended = time.perf_counter()
            self.current = None
        spans = sorted(row.pop('spans'), key=lambda span: span[1])
        cursor = started
        stages = {}
        for label, begin, end in spans:
            assert cursor <= begin <= end <= ended, (
                'overlapping or asynchronous stages cannot be added as disjoint time')
            cursor = end
            stages[label] = stages.get(label, 0.) + end - begin
        row.pop('thread')
        row.update(step_seconds=ended - started, stage_seconds=stages,
                   other_host_seconds=ended - started - sum(stages.values()))
        prefill, decode = row['scheduled_prefill_tokens'], row['scheduled_decode_tokens']
        row['phase'] = ('mixed' if decode else 'prefill') if prefill else (
            'decode' if decode else 'idle')
        self.rows.append(row)
        return outputs


def write_profile(profile, prefix):
    profile.dump_stats(str(prefix.with_suffix('.pstats')))
    stats = pstats.Stats(profile)
    records = []
    for (file, line, name), (primitive, calls, self_time, cumulative, _) in stats.stats.items():
        records.append(dict(file=file, line=line, function=name, calls=calls,
                            primitive_calls=primitive, self_seconds=self_time,
                            cumulative_seconds=cumulative))
    prefix.with_suffix('.json').write_text(json.dumps(dict(
        warning='Cumulative times overlap; extension calls may include device waits.',
        functions=sorted(records, key=lambda row: row['self_seconds'], reverse=True)), indent=2) + '\n')
    for sort in ('cumulative', 'tottime'):
        stream = io.StringIO()
        pstats.Stats(profile, stream=stream).sort_stats(sort).print_stats(60)
        prefix.with_name(prefix.name + '-' + sort + '.txt').write_text(stream.getvalue())


def summarize_steps(rows):
    result = {}
    for phase in sorted({row['phase'] for row in rows}):
        selected = [row for row in rows if row['phase'] == phase]
        labels = sorted({label for row in selected for label in row['stage_seconds']})
        result[phase] = dict(
            steps=len(selected), step_seconds=distribution([r['step_seconds'] for r in selected]),
            other_host_seconds=distribution([r['other_host_seconds'] for r in selected]),
            stage_seconds={label: distribution([r['stage_seconds'].get(label, 0.)
                                                for r in selected]) for label in labels})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('oracle', 'jittor'), required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--baseline', help='Declared synchronized production revision')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 4])
    parser.add_argument('--input-tokens', type=int, default=128)
    parser.add_argument('--output-tokens', type=int, default=32)
    parser.add_argument('--dtype', choices=('float16', 'bfloat16', 'float32'), default='float16')
    parser.add_argument('--skip-cprofile', action='store_true')
    args = parser.parse_args()
    if min(args.batches + [args.repetitions, args.warmups,
                          args.input_tokens, args.output_tokens]) < 1:
        parser.error('batch sizes, repetitions, warmups and token counts must be positive')
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        parser.error('output directory is nonempty; preserve earlier evidence')
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', arguments=vars(args), backend=args.backend,
                  scope='diagnostic host wall time; not GPU kernel timing or formal benchmark',
                  script_sha256=sha256(__file__),
                  latency_source_sha256=sha256(Path(__file__).with_name('latency_acceptance.py')),
                  batches=[])

    def save():
        temporary = out / 'report.json.tmp'
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(out / 'report.json')

    try:
        save()
        os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING='0', VLLM_USE_FLASHINFER_SAMPLER='0',
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        assert os.environ.get('JITTOR_VLLM_HOST_ONLY') != '1'
        if args.backend == 'jittor':
            import jittor as jt
            jt.flags.use_parallel_op_compiler = 0
            jt.flags.use_cuda = 1
        import torch
        assert hasattr(torch, '_torch_compat_install_context') == (args.backend == 'jittor')
        assert torch.cuda.is_available()
        from vllm import LLM, SamplingParams
        report.update(default_device=str(torch.get_default_device()),
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      gpu_name=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
                      versions={k: importlib.metadata.version(k) for k in ('vllm', 'transformers')})
        options = dict(model=args.model, dtype=args.dtype, tensor_parallel_size=1,
                       enforce_eager=True, enable_prefix_caching=False, seed=0,
                       max_model_len=max(512, args.input_tokens + args.output_tokens),
                       max_num_seqs=max(args.batches), gpu_memory_utilization=.35,
                       attention_config={'backend': 'FLASH_ATTN'})
        report['options'] = options
        llm = LLM(**options)
        engine = llm.llm_engine
        core = engine.engine_core.engine_core
        runner = core.model_executor.driver_worker.worker.model_runner
        parameters, caches = list(runner.model.named_parameters()), list(runner.kv_caches)
        assert parameters and caches
        assert all(t.device.type == 'cuda' for _, t in parameters)
        assert all(t.device.type == 'cuda' for t in caches)
        report['gpu_placement'] = dict(parameters=len(parameters), kv_caches=len(caches))
        prompts = []
        for index in range(max(args.batches)):
            text = ('Example %d. ' % index) + 'Explain how science helps us understand the world. ' * args.input_tokens
            ids = llm.get_tokenizer().encode(text, add_special_tokens=False)[:args.input_tokens]
            assert len(ids) == args.input_tokens
            prompts.append({'prompt_token_ids': ids})
        report['prompts'] = prompts
        params = SamplingParams(temperature=0, max_tokens=args.output_tokens,
                                ignore_eos=True, detokenize=True)
        serial = 0

        def generate(batch, recorder=None):
            nonlocal serial
            assert not engine.has_unfinished_requests()
            torch.cuda.synchronize()
            pending, completed = {}, {}
            for index in range(batch):
                request_id = 'stage_%d' % serial
                serial += 1
                pending[request_id] = dict(index=index, tokens=[])
                engine.add_request(request_id, prompts[index], params)
            steps = 0
            while engine.has_unfinished_requests():
                outputs = recorder.step() if recorder is not None else engine.step()
                steps += 1
                assert steps <= batch * (args.input_tokens + args.output_tokens + 10), 'engine stalled'
                for output in outputs:
                    row = pending[output.request_id]
                    tokens = list(output.outputs[0].token_ids)
                    previous = len(row['tokens'])
                    assert tokens[:previous] == row['tokens'] and len(tokens) - previous <= 1
                    row['tokens'] = tokens
                    if output.finished:
                        assert len(tokens) == args.output_tokens
                        assert output.outputs[0].finish_reason == 'length'
                        completed[row['index']] = tokens
                        del pending[output.request_id]
            torch.cuda.synchronize()
            assert not pending and len(completed) == batch
            return [completed[index] for index in range(batch)]

        for batch in args.batches:
            reference = None
            for index in range(args.warmups):
                print('WARMUP', batch, index, flush=True)
                tokens = generate(batch)
                if reference is not None:
                    assert tokens == reference, 'warmup greedy output changed'
                reference = tokens
            row = dict(batch=batch, token_ids=reference, repetitions=args.repetitions)
            report['batches'].append(row)
            recorder = StageRecorder(engine, core, runner)
            with recorder.installed():
                for repeat in range(args.repetitions):
                    print('STAGES', batch, repeat, flush=True)
                    assert generate(batch, recorder) == reference, 'stage hooks changed tokens'
            row.update(steps=recorder.rows, stage_summary=summarize_steps(recorder.rows))
            save()
            if not args.skip_cprofile:
                print('CPROFILE', batch, flush=True)
                profile = cProfile.Profile()
                started = time.perf_counter()
                profile.enable()
                try:
                    for _ in range(args.repetitions):
                        assert generate(batch) == reference, 'cProfile pass changed tokens'
                finally:
                    profile.disable()
                    row['cprofile_wall_seconds'] = time.perf_counter() - started
                    write_profile(profile, out / ('batch%d-python' % batch))
                save()
        report['status'] = 'completed'
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        report.update(status='failed', exception=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == '__main__':
    main()

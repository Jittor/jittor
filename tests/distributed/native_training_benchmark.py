"""Repeatable native DDP training, correctness, resume and resource probe.

Run with ``jtrun --nproc-per-node=N --device-ids=...``. The script is an
integration harness rather than a pytest test because it needs real GPUs. It
uses a deterministic synthetic classification workload so it needs no network
or external checkpoint.
"""

import argparse
import faulthandler
from functools import lru_cache
import json
import os
import signal
import time

import numpy as np

import jittor as jt
from jittor import distributed as dist
from jittor import nn
from jittor.nn.parallel import DistributedDataParallel


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, classes):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, classes)

    def execute(self, x):
        return self.fc2(jt.nn.relu(self.fc1(x)))


@lru_cache(maxsize=None)
def _teacher_weights(input_dim, classes):
    rng = np.random.RandomState(7301)
    weights = rng.standard_normal((input_dim, classes)).astype(np.float32)
    return weights / np.sqrt(input_dim)


def _cuda_allocator_reserved_mib():
    # Jittor's MemInfo total_cuda_used sums used + unused blocks owned by its
    # allocator, so this is a reserved/pool measure rather than live tensor
    # bytes or process-wide nvidia-smi usage.
    info = jt.get_mem_info()
    return float(info.total_cuda_used) / (1024.0 * 1024.0)


def _allclose_rank(tensor, rtol=1e-5, atol=1e-6):
    values = dist.all_gather(tensor.reshape((-1,)).stop_grad()).numpy()
    values = np.asarray(values).reshape(dist.get_world_size(), -1)
    return bool(np.allclose(values, values[0:1], rtol=rtol, atol=atol)), values


def _batch(step, rank, local_batch, input_dim, classes):
    # Identical global sequence across configurations; each rank gets a
    # disjoint, deterministic slice. Labels are generated from a fixed linear
    # rule so loss reduction and optimizer updates are reproducible.
    global_batch = local_batch * dist.get_world_size()
    start = step * global_batch + rank * local_batch
    sample_ids = np.arange(start, start + local_batch, dtype=np.float32)[:, None] + 1
    feature_ids = np.arange(input_dim, dtype=np.float32)[None, :] + 1
    hashed = np.sin(sample_ids * 12.9898 + feature_ids * 78.233) * 43758.5453
    features = ((hashed - np.floor(hashed)) * 2.0 - 1.0).astype(np.float32)
    targets = np.argmax(features @ _teacher_weights(input_dim, classes), axis=1)
    return jt.array(features.astype(np.float32)).cuda(0).stop_grad(), \
        jt.array(targets).int32().cuda(0).stop_grad()


def _step(model, optimizer, step, args, rank):
    x, target = _batch(step, rank, args.batch_size, args.input_dim, args.classes)
    logits = model(x)
    loss = jt.nn.cross_entropy_loss(logits, target)
    optimizer.backward(loss)
    optimizer.step()
    return loss


def _evaluate_loss(model, args, rank):
    x, target = _batch(0, rank, args.batch_size, args.input_dim, args.classes)
    value = float(jt.nn.cross_entropy_loss(model(x), target).numpy().mean())
    if not np.isfinite(value):
        raise AssertionError('evaluation loss is not finite')
    return value


def main():
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128,
                        help='per-rank batch; global batch scales with world size')
    parser.add_argument('--input-dim', type=int, default=1024)
    parser.add_argument('--hidden-dim', type=int, default=2048)
    parser.add_argument('--classes', type=int, default=32)
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.02)
    parser.add_argument('--require-loss-drop', action='store_true',
                        help='fail unless the fixed rank-0 evaluation loss decreases')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps <= args.warmup or args.warmup < 1:
        parser.error('require batch-size > 0 and steps > warmup >= 1')

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    print('rank={} process_group_initialized'.format(rank), flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        jt.set_global_seed(7301)
        raw_model = MLP(args.input_dim, args.hidden_dim, args.classes).cuda(0)
        print('rank={} model_created'.format(rank), flush=True)
        model = DistributedDataParallel(raw_model)
        optimizer = jt.optim.Adam(model.parameters(), lr=args.learning_rate)

        initial_eval_loss = _evaluate_loss(model, args, rank)
        jt.sync_all(True)
        print('rank={} first_step_start'.format(rank), flush=True)
        first_start = time.perf_counter()
        first_loss = _step(model, optimizer, 0, args, rank)
        jt.sync_all(True)
        print('rank={} first_step_done'.format(rank), flush=True)
        first_step_seconds = time.perf_counter() - first_start
        first_loss_value = float(first_loss.numpy().mean())
        if not np.isfinite(first_loss_value):
            raise AssertionError('first training loss is not finite')
        first_reserved_mib = _cuda_allocator_reserved_mib()

        for step in range(1, args.warmup):
            _step(model, optimizer, step, args, rank)
        jt.sync_all(True)
        print('rank={} warmup_done'.format(rank), flush=True)
        steady_start = time.perf_counter()
        for step in range(args.warmup, args.steps):
            loss = _step(model, optimizer, step, args, rank)
        jt.sync_all(True)
        print('rank={} steady_done'.format(rank), flush=True)
        steady_seconds = time.perf_counter() - steady_start
        steady_loss_value = float(loss.numpy().mean())
        if not np.isfinite(steady_loss_value):
            raise AssertionError('steady training loss is not finite')
        final_eval_loss = _evaluate_loss(model, args, rank)
        if args.require_loss_drop and not final_eval_loss < initial_eval_loss:
            raise AssertionError(
                'rank {} evaluation loss did not decrease: {} -> {}'.format(
                    rank, initial_eval_loss, final_eval_loss))
        steady_reserved_mib = _cuda_allocator_reserved_mib()

        consistent, final_parameters = _allclose_rank(
            jt.concat([p.reshape((-1,)) for p in model.parameters()]))
        if not consistent:
            raise AssertionError('DDP replicas diverged after training')

        checkpoint = os.path.join(args.output_dir, 'native_ddp_checkpoint.pkl')
        optimizer_checkpoint = os.path.join(
            args.output_dir, 'native_ddp_optimizer.pkl')
        if rank == 0:
            model.module.save(checkpoint)
            jt.save(optimizer.state_dict(), optimizer_checkpoint)
        dist.barrier()
        print('rank={} checkpoint_saved'.format(rank), flush=True)

        # Advance the original model once as the continuation reference. The
        # fresh model below must match it after loading both checkpoints,
        # including Adam's moment buffers and per-group step counters.
        control_loss = _step(model, optimizer, args.steps, args, rank)
        jt.sync_all(True)
        control_loss_value = float(control_loss.numpy().mean())
        control_parameters = jt.concat(
            [p.reshape((-1,)) for p in model.parameters()]).numpy()

        resume = MLP(args.input_dim, args.hidden_dim, args.classes).cuda(0)
        resume.load(checkpoint)
        print('rank={} checkpoint_loaded'.format(rank), flush=True)
        resumed = DistributedDataParallel(resume)
        resumed_optimizer = jt.optim.Adam(resumed.parameters(),
                                          lr=args.learning_rate)
        resumed_optimizer.load_state_dict(jt.load(optimizer_checkpoint))
        resume_loss = _step(resumed, resumed_optimizer, args.steps, args, rank)
        jt.sync_all(True)
        resume_loss_value = float(resume_loss.numpy().mean())
        if not np.isfinite(resume_loss_value):
            raise AssertionError('checkpoint resume loss is not finite')
        resumed_parameters = jt.concat(
            [p.reshape((-1,)) for p in resumed.parameters()]).numpy()
        optimizer_resume_matches = (
            np.allclose(resume_loss_value, control_loss_value, rtol=1e-5,
                        atol=1e-6) and
            np.allclose(resumed_parameters, control_parameters, rtol=1e-5,
                        atol=1e-6))
        if not optimizer_resume_matches:
            raise AssertionError('optimizer-state checkpoint diverged from continuation')
        resume_consistent, _ = _allclose_rank(
            jt.concat([p.reshape((-1,)) for p in resumed.parameters()]))
        if not resume_consistent:
            raise AssertionError('resumed DDP replicas diverged')

        if rank == 0:
            result = {
                'backend': 'nccl',
                'world_size': dist.get_world_size(),
                'global_batch_size': args.batch_size * dist.get_world_size(),
                'per_rank_batch_size': args.batch_size,
                'model': {'input_dim': args.input_dim,
                          'hidden_dim': args.hidden_dim,
                          'classes': args.classes},
                'steps': args.steps,
                'warmup_steps': args.warmup,
                'first_step_seconds': first_step_seconds,
                'first_step_loss_rank0_local': first_loss_value,
                'initial_eval_loss_rank0_local': initial_eval_loss,
                'final_eval_loss_rank0_local': final_eval_loss,
                'eval_loss_drop_fraction_rank0_local':
                    (initial_eval_loss - final_eval_loss) / initial_eval_loss,
                'steady_optimizer_steps_per_second':
                    (args.steps - args.warmup) / steady_seconds,
                'steady_samples_per_second_global':
                    ((args.steps - args.warmup) * args.batch_size *
                     dist.get_world_size()) / steady_seconds,
                'steady_seconds_per_optimizer_step':
                    steady_seconds / (args.steps - args.warmup),
                'steady_loss_rank0_local_last': steady_loss_value,
                'first_step_jittor_allocator_reserved_mib': first_reserved_mib,
                'steady_jittor_allocator_reserved_mib': steady_reserved_mib,
                'checkpoint_path': checkpoint,
                'optimizer_checkpoint_path': optimizer_checkpoint,
                'resume_loss_rank0_local': resume_loss_value,
                'optimizer_resume_matches': optimizer_resume_matches,
                'replicas_equal': consistent and resume_consistent,
                'final_parameter_abs_max': float(np.max(np.abs(final_parameters))),
            }
            out = os.path.join(args.output_dir, 'result.json')
            with open(out, 'w', encoding='utf-8') as stream:
                json.dump(result, stream, indent=2, sort_keys=True)
                stream.write('\n')
            print('NATIVE_DDP_TRAINING_OK ' + json.dumps(result, sort_keys=True),
                  flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

"""Fail-closed probe of verl's real worker process-group bootstrap contract.

This does not start Ray, an engine, or training. RANK/WORLD_SIZE/LOCAL_RANK and
DIST_INIT_METHOD must describe a task-owned rendezvous. Run under a launcher
for multiple ranks; a successful single-rank no-op is not communication proof.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import numpy as np

from probe_training_workflow import initialize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", help="explicit verl backend override (e.g. hccl); omitted tests upstream default")
    args = parser.parse_args()
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "DIST_INIT_METHOD"):
        if name not in os.environ:
            parser.error("explicit " + name + " is required")
    with ExitStack() as stack:
        torch, jt, move, synchronize, assert_device = initialize(args, stack)
        before_fallback = int(jt.core.backend_fallback_count()) if jt is not None else None
        from verl.utils.distributed import initialize_global_process_group_ray
        from verl.utils.device import get_device_name, get_nccl_backend

        requested = args.backend or "cpu:gloo,{}:{}".format(get_device_name(), get_nccl_backend())
        if torch.distributed.is_initialized():
            raise RuntimeError("probe requires a fresh, uninitialized process group")
        result = {"framework": args.framework, "requested_backend": requested, "status": "running"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            initialize_global_process_group_ray(timeout_second=120, backend=args.backend)
            actual = str(torch.distributed.get_backend())
            result = {
                "scope": "verl worker bootstrap only; not collective or trainer validation",
                "framework": args.framework, "requested_backend": requested,
                "actual_backend": actual, "rank": torch.distributed.get_rank(),
                "world_size": torch.distributed.get_world_size(),
                "status": "bootstrap_pass" if actual == requested else "fail",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            if actual != requested:
                raise RuntimeError("worker backend contract mismatch: {} != {}".format(actual, requested))
            rank, size = result["rank"], result["world_size"]
            value = move(torch.tensor([float(rank + 1), float(2 * rank + 3)]))
            synchronize()
            assert_device(value)
            torch.distributed.all_reduce(value)
            synchronize()
            np.testing.assert_array_equal(value.detach().cpu().numpy(),
                                          [size * (size + 1) / 2, size * (size + 2)])
            received = [None] * size
            torch.distributed.all_gather_object(received, {"rank": rank, "payload": "x" * (rank + 1)})
            assert received == [{"rank": r, "payload": "x" * (r + 1)} for r in range(size)], received
            torch.distributed.barrier()
            for iteration in range(3):
                for operation, expected in ((torch.distributed.ReduceOp.MAX, size + iteration),
                                            (torch.distributed.ReduceOp.MIN, 1 + iteration)):
                    reduced = move(torch.tensor([float(rank + 1 + iteration)]))
                    torch.distributed.all_reduce(reduced, op=operation)
                    synchronize()
                    assert_device(reduced)
                    np.testing.assert_array_equal(reduced.detach().cpu().numpy(), [expected])
                message = [{"source": rank, "iteration": iteration}] if rank == size - 1 else [None]
                torch.distributed.broadcast_object_list(message, src=size - 1)
                assert message == [{"source": size - 1, "iteration": iteration}]
            if jt is not None:
                try:
                    torch.distributed.all_reduce(torch.tensor([1.0], device="cpu"))
                except RuntimeError as error:
                    assert "HCCL tensor collectives require NPU tensors" in str(error), str(error)
                else:
                    raise AssertionError("pure HCCL accepted an unsupported CPU tensor collective")
                fallback_count = int(jt.core.backend_fallback_count()) - before_fallback
                assert fallback_count == 0
            else:
                fallback_count = None
            result.update(scope="verl public bootstrap, device all-reduce, host object all-gather and barrier; not trainer validation",
                          collective_rank_count=size, collective_status="pass",
                          communication_proven=size > 1, status="pass", fallback_count=fallback_count,
                          extra_checks=["three repeated MAX/MIN", "nonzero-root object broadcast",
                                        "CPU tensor reject (candidate only)"])
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
        except Exception as error:
            result.update(status="fail", error=type(error).__name__ + ": " + str(error))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            raise
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

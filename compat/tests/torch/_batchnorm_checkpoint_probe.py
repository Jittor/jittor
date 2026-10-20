"""Small two-process BatchNorm counter/checkpoint reproduction.

Invoke ``--phase save`` and then ``--phase resume`` with the same checkpoint.
The save process runs five batches; the new process must restore five and then
reach seven after two further train forwards. No framework imports at import.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np

from _training_trajectory_runner import _array, _assert_device, _import_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--phase", choices=("save", "resume"), required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    jt, torch = _import_runtime(args.runtime)
    with ExitStack() as stack:
        if jt is not None:
            stack.enter_context(jt.runtime.scope(
                use_cuda=int(args.device == "cuda"), backend_fallback="error"))
            from jittor._runtime.fallback import forbid_backend_fallbacks

            stack.enter_context(forbid_backend_fallbacks())
        fallback_before = jt.core.backend_fallback_count() if jt is not None else 0
        bn = torch.nn.BatchNorm2d(3).to(args.device)
        bn.train()
        inputs = torch.from_numpy(np.arange(6 * 3 * 4 * 4, dtype=np.float32).reshape(
            6, 3, 4, 4) / np.float32(100)).to(args.device)
        _assert_device(inputs, args.device, "probe input")
        before = int(_array(bn.num_batches_tracked).reshape(-1)[0])
        if args.phase == "save":
            for _ in range(5):
                bn(inputs).sum().item()
            state = bn.state_dict()
            Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": state, "completed_forwards": 5}, args.checkpoint)
            restored = None
            expected = 5
        else:
            checkpoint = torch.load(args.checkpoint, weights_only=False)
            state = checkpoint["model"]
            bn.load_state_dict(state, strict=True)
            restored = int(_array(bn.num_batches_tracked).reshape(-1)[0])
            for _ in range(2):
                bn(inputs).sum().item()
            expected = checkpoint["completed_forwards"] + 2
        actual = int(_array(bn.num_batches_tracked).reshape(-1)[0])
        fallback_count = jt.core.backend_fallback_count() - fallback_before if jt is not None else 0
        result = {"runtime": args.runtime, "device": args.device, "phase": args.phase,
                  "counter_before": before, "counter_after_load": restored,
                  "counter_after_forwards": actual, "expected": expected,
                  "counter_dtype": str(bn.num_batches_tracked.dtype),
                  "counter_shape": list(bn.num_batches_tracked.shape),
                  "state_keys": sorted(state),
                  "state_contains_counter": "num_batches_tracked" in state,
                  "fallback_count": fallback_count}
        print(json.dumps(result, sort_keys=True), flush=True)
        if actual != expected:
            raise AssertionError("restored BatchNorm counter actual=%d expected=%d" % (
                actual, expected))
        if fallback_count:
            raise AssertionError("probe used backend fallback")


if __name__ == "__main__":
    main()

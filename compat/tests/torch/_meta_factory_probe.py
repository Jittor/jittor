"""Minimal metadata-only meta factory/loading probe; one real device per run."""

import argparse
from contextlib import ExitStack
import json

from _hf_trainer_e2e_runner import _runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    jt, torch = _runtime(args.runtime)
    with ExitStack() as stack:
        if jt is not None:
            stack.enter_context(jt.runtime.scope(use_cuda=int(args.device == "cuda"),
                                                backend_fallback="error"))
            from jittor._runtime.fallback import forbid_backend_fallbacks

            stack.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
        real = torch.tensor([1.0, 2.0], device=args.device)
        explicit = torch.empty((2, 3), dtype=torch.float64, device="meta", requires_grad=True)
        assert explicit.device.type == "meta" and explicit.is_meta
        assert tuple(explicit.shape) == (2, 3) and explicit.dtype == torch.float64
        assert explicit.requires_grad
        with torch.device("meta"):
            contextual = torch.empty((2, 3))
            detection = torch.tensor([])
            parameter = torch.nn.Parameter(torch.empty((2, 3)))
            linear = torch.nn.Linear(3, 2)
            override = torch.zeros(2, device=args.device)
            assert real.device.type == args.device
            assert override.device.type == args.device and not override.is_meta
            assert detection.is_meta and contextual.is_meta and parameter.is_meta
        assert contextual.is_meta and parameter.is_meta
        assert all(value.is_meta for value in linear.parameters())
        assert torch.empty_like(explicit).is_meta
        assert explicit.detach().is_meta and explicit.clone().is_meta
        materialized = torch.empty_like(explicit, device=args.device)
        assert not materialized.is_meta and materialized.device.type == args.device
        materialized.fill_(3)
        assert float(materialized.sum().item()) == 18
        assert explicit.is_meta
        linear.load_state_dict({"weight": torch.ones((2, 3), device=args.device),
                                "bias": torch.zeros(2, device=args.device)}, assign=True)
        assert all(not value.is_meta and value.device.type == args.device
                   for value in linear.parameters())
        prediction = linear(torch.ones((1, 3), device=args.device))
        assert float(prediction.sum().item()) == 6
        if args.device == "cuda":
            torch.cuda.synchronize()
        fallback = jt.core.backend_fallback_count() - before if jt is not None else 0
        assert fallback == 0
        print(json.dumps({"runtime": args.runtime, "device": args.device,
                          "meta_shape": list(explicit.shape),
                          "context_and_explicit_meta": True,
                          "materialized_device": str(materialized.device),
                          "fallback_count": fallback,
                          "storage_contract": "metadata-only; no no-storage claim"}))


if __name__ == "__main__":
    main()

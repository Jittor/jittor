#!/usr/bin/env python3
"""Manual default/private ACL RNG comparison; no automatic device execution."""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import traceback


def provenance(obj):
    path = Path(inspect.getsourcefile(obj)).resolve()
    return dict(module=obj.__module__, name=obj.__qualname__, file=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(ordinal):
    import jittor as jt
    from jittor.compat.shim import activate
    activate(strict=True, auto_scan_extensions=False, build_extensions=False,
             local_home=False)
    import torch
    from jittor.compat.torch import namespace_owner
    if namespace_owner(torch) is not jt:
        raise RuntimeError("A Jittor-owned Torch namespace is required")
    if jt.compiler.build_config.backend != "acl" or not jt.compiler.has_acl:
        raise RuntimeError("A real ACL build is required")
    if not jt.compiler.core_build_is_current():
        raise RuntimeError("The existing Core does not match the installed sources")
    report = dict(status="NOT_COMPLETED", ordinal=ordinal, draw_calls=0, completed_draws=[],
                  providers=dict(Generator=provenance(torch.Generator),
                                 exponential=provenance(torch.Tensor.exponential_)),
                  core_sha256=hashlib.sha256(Path(jt.core.__file__).read_bytes()).hexdigest())
    with jt.runtime.scope(use_cuda=1, device_id=ordinal, backend_fallback="error"):
        before = int(jt.core.backend_fallback_count())
        report["fallback_before"] = before
        if before:
            raise RuntimeError("The reproducer requires an initially clean fallback counter")

        def draw(name, generator=None):
            if report["draw_calls"] >= 6:
                raise RuntimeError("Small-draw budget exceeded")
            report["draw_calls"] += 1
            value = torch.empty((2, 16), dtype=torch.float32,
                                device="cuda:%d" % ordinal)
            result = (value.exponential_() if generator is None
                      else value.exponential_(generator=generator))
            if result is not value:
                raise RuntimeError("The in-place operation replaced its object")
            jt.sync_all(True)
            if (not isinstance(value, jt.Var) or int(value.placement_backend) != 2
                    or int(value.device_id) != ordinal or value.location() != "device"):
                raise RuntimeError("The result is not resident on the selected ACL device")
            values = value.tolist()  # Explicit result readback, not host RNG.
            if not all(math.isfinite(x) and x >= 0 for row in values for x in row):
                raise RuntimeError("Invalid exponential result")
            observed = dict(name=name, device=str(value.device),
                            backend=int(value.placement_backend), ordinal=int(value.device_id),
                            location=value.location(), values=values)
            report["completed_draws"].append(observed)
            return observed

        try:
            torch.cuda.manual_seed_all(12345)
            first = draw("default_first")
            torch.cuda.manual_seed_all(12345)
            replay = draw("default_seed_replay")
            if first["values"] != replay["values"]:
                raise RuntimeError("Default-stream seed replay failed")
            torch.cuda.manual_seed_all(67890)
            changed = draw("default_changed_seed")
            if first["values"] == changed["values"]:
                raise RuntimeError("Changing the default seed did not change this sample")
            snapshot = torch.cuda.get_rng_state(ordinal)
            following = draw("default_before_restore")
            torch.cuda.set_rng_state(snapshot, ordinal)
            restored = draw("default_after_restore")
            if following["values"] != restored["values"]:
                raise RuntimeError("Default-stream state restoration failed")
            report["default_stream"] = dict(status="PASS")
            generator = torch.Generator(device="cuda:%d" % ordinal)
            generator.manual_seed(12345)
            report["private_owner"] = dict(type=type(generator).__module__ + "." + type(generator).__name__,
                                           device=str(generator.device),
                                           manual_seed="RETURNED")
            report["offset_interfaces"] = {
                name: callable(getattr(generator, name, None))
                for name in ("get_offset", "set_offset")}
            report["private_draw"] = draw("explicit_private", generator)
            report["status"] = "EXPLICIT_DRAW_RETURNED_NOT_FULL_CONTRACT"
        except Exception as error:
            # This is a diagnostic report, never an exception-to-PASS adapter.
            report["status"] = "FAIL"
            report["error"] = dict(type=type(error).__name__, message=str(error),
                                   traceback=traceback.format_exc())
        finally:
            report["fallback_after"] = int(jt.core.backend_fallback_count())
    print(json.dumps(report, indent=2))
    return 0 if (report["status"] == "EXPLICIT_DRAW_RETURNED_NOT_FULL_CONTRACT"
                 and report["fallback_after"] == 0) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Run only after external resource admission; imports and loads one existing ACL Core")
    parser.add_argument("--ordinal", type=int, required=True, help="Visible runtime user ordinal, not physical card ID")
    args = parser.parse_args()
    if not args.run:
        parser.error("--run is required; no framework or device was imported")
    if args.ordinal < 0:
        parser.error("--ordinal must be nonnegative")
    if os.environ.get("JT_BACKEND") != "acl" or os.environ.get("JITTOR_NO_BUILD") != "1":
        parser.error("Set JT_BACKEND=acl and JITTOR_NO_BUILD=1 before running")
    if any(os.environ.get(k) for k in ("JT_HCCL_WORLD_SIZE", "JT_HCCL_RANK", "JT_HCCL_ROOTINFO_FILE")):
        parser.error("This single-worker reproducer must not initialize an HCCL WORLD")
    return run(args.ordinal)


if __name__ == "__main__":
    sys.exit(main())

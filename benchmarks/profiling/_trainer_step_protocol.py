"""Training-step contracts without Torch/Jittor imports. CPU qualification and NPU performance admission remain separate."""

import ctypes
from pathlib import Path
import math
import statistics
import time
import collections
from unittest import mock
import copy


class UpdateTimer:
    def __init__(
        self, synchronize, *, warmup=4, measured=10, required_shapes, clock=time.perf_counter
    ):
        if warmup < 4 or measured < 10:
            raise ValueError("require >=4 full-cycle warmup and >=10 measured optimizer updates")
        self.required_shapes = frozenset(required_shapes)
        if not self.required_shapes:
            raise ValueError("require the frozen full microbatch input-shape inventory")
        self.warmed_shapes = set()
        (self.sync, self.clock) = (synchronize, clock)
        (self.warmup, self.measured) = (warmup, measured)
        self.active = None
        self.completed = 0
        self.in_timed_update = False
        self.samples = []
        self.admitted = False
        self.invalid = False

    def _reject(self, message):
        self.invalid = True
        self.admitted = False
        raise RuntimeError(message)

    def begin(self, step):
        if self.invalid:
            self._reject("invalidated timing cohort cannot continue")
        if step > self.warmup and self.warmed_shapes != self.required_shapes:
            self._reject("warmup did not cover the full microbatch shape inventory")
        if self.active is not None or step != self.completed + 1:
            self._reject("optimizer update boundary is incomplete or out of order")
        if step > self.warmup + self.measured:
            self._reject("optimizer update exceeds the frozen budget")
        self.active = step
        self.sync()
        self.in_timed_update = step > self.warmup
        self.started = self.clock() if self.in_timed_update else None

    def end(self, step, *, optimizer_skipped=False):
        if self.active != step:
            self._reject("optimizer update end has no matching begin")
        self.sync()
        elapsed = self.clock() - self.started if self.in_timed_update else None
        self.in_timed_update = False
        self.active = None
        if optimizer_skipped:
            self._reject("skipped optimizer update cannot be admitted")
        if elapsed is not None:
            if not math.isfinite(elapsed) or elapsed <= 0:
                self._reject("invalid synchronized update duration")
            self.samples.append(elapsed)
        self.completed = step

    def microbatch(self, shape):
        if self.active is None or shape not in self.required_shapes:
            self._reject("microbatch is outside an update or outside the frozen shape inventory")
        if self.active <= self.warmup:
            self.warmed_shapes.add(shape)

    def outside(self, operation):
        if self.in_timed_update:
            self._reject(operation + " is forbidden inside the measured update")

    def finish(self, jit_before, jit_after):
        if self.invalid:
            raise RuntimeError("invalidated timing cohort cannot be admitted")
        if self.active is not None or len(self.samples) != self.measured:
            self._reject("timing does not cover the full admitted update budget")
        if jit_before is None or jit_before != jit_after:
            self._reject("compilation evidence changed or is unavailable")
        self.admitted = True
        return {
            "updates": len(self.samples),
            "warmup_updates": self.warmup,
            "warmup_shape_inventory": sorted(self.warmed_shapes),
            "seconds_per_update": list(self.samples),
            "mean_seconds": statistics.mean(self.samples),
            "median_seconds": statistics.median(self.samples),
        }


class ReadbackGuard:
    METHODS = ("cpu", "numpy", "item", "tolist", "__bool__", "__float__", "__int__")

    def __init__(self, timer, tensor_type):
        (self.timer, self.tensor_type) = (timer, tensor_type)
        self.inside_cpu = collections.Counter()
        self.inside_accelerator = collections.Counter()
        self.outside_counts = collections.Counter()

    def install(self, stack):
        for name in self.METHODS:
            original = getattr(self.tensor_type, name)

            def wrapped(value, *args, _name=name, _original=original, **kwargs):
                if self.timer.in_timed_update:
                    device_type = str(value.device.type)
                    if device_type != "cpu":
                        self.inside_accelerator[_name] += 1
                        self.timer.outside("accelerator readback " + device_type + "." + _name)
                        raise RuntimeError("readback rejection must not return")
                    self.inside_cpu[_name] += 1
                else:
                    self.outside_counts[_name] += 1
                return _original(value, *args, **kwargs)

            stack.enter_context(mock.patch.object(self.tensor_type, name, wrapped))


class GateWindow:
    in_timed_update = True

    def outside(self, operation):
        raise RuntimeError(operation + " is forbidden inside the measured update")


def cpu_sdpa_control(torch):
    """Actual CPU SDPA helper control; does not establish NPU validation."""
    from contextlib import ExitStack
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

    mask = torch.ones((2, 4), dtype=torch.int64, device="cpu")
    guard = ReadbackGuard(GateWindow(), torch.Tensor)
    with ExitStack() as stack:
        guard.install(stack)
        result = _prepare_4d_attention_mask_for_sdpa(mask, torch.float32)
    assert result is None and guard.inside_cpu["__bool__"] == 1 and (not guard.inside_accelerator)
    return {
        "status": "CPU_CONTROL_PASS",
        "inside_cpu": dict(guard.inside_cpu),
        "npu_evidence": False,
    }


def reject_actual_npu_sdpa_mask(torch, mask):
    """Run only under a separately admitted NPU lease, with a real NPU all-one mask.
    This helper creates no tensor, does not acquire a device, and is not called by CPU probes.
    """
    from contextlib import ExitStack
    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

    if str(mask.device.type) != "npu":
        raise ValueError("real NPU mask required")
    guard = ReadbackGuard(GateWindow(), torch.Tensor)
    with ExitStack() as stack:
        guard.install(stack)
        try:
            _prepare_4d_attention_mask_for_sdpa(mask, torch.float32)
        except RuntimeError as error:
            if "accelerator readback npu.__bool__ is forbidden" not in str(error):
                raise
        else:
            raise AssertionError("SDPA NPU tensor-bool readback was not rejected")
    assert guard.inside_accelerator == {"__bool__": 1}
    return {
        "status": "ACTUAL_NPU_SDPA_READBACK_REJECTED",
        "operation": "__bool__",
        "device": str(mask.device),
        "outside_original_method": True,
    }


class AdmissionSequence:
    def __init__(self, protocol):
        self.protocol = copy.deepcopy(protocol)
        self.order = ["torch", "jittor"] * 10
        self.admissions = []
        self.attempts = []

    def append(self, row):
        row = copy.deepcopy(row)
        attempt = {"cohort": row.get("cohort"), "runtime": row.get("runtime"), "status": "FAIL"}
        self.attempts.append(attempt)

        def require(condition, message):
            if not condition:
                attempt["reason"] = message
                raise ValueError(message)

        require(len(self.admissions) < 20, "all20 admitted slots are already complete")
        require(
            isinstance(row.get("cohort"), str)
            and row["cohort"]
            and (not any((x.get("cohort") == row["cohort"] for x in self.attempts[:-1]))),
            "cohort identity must be unique, including failed attempts",
        )
        require(
            row.get("runtime") == self.order[len(self.admissions)],
            "runtime does not match the alternating admission order",
        )
        require(
            row.get("protocol_sha256") == self.protocol["sha256"],
            "frozen shared source/config/weights/data/precision protocol changed",
        )
        require(
            row.get("runtime_source_sha256") == self.protocol["runtime_sources"][row["runtime"]],
            "runtime source/version fingerprint changed",
        )
        require(
            row.get("attention_implementation") == "eager"
            and row.get("functional_attention_implementation") == "eager",
            "separate explicit eager functional and timing configuration must match",
        )
        require(
            row.get("phase_label") == "synchronized training phases"
            and row.get("phase_barrier_position") == "postclip-preoptimizer",
            "common synchronized training phase protocol is required",
        )
        require(
            row.get("actual_npu_sdpa_rejection_status") == "ACTUAL_NPU_SDPA_READBACK_REJECTED"
            and len(row.get("actual_npu_sdpa_rejection_artifact_sha256", "")) == 64,
            "real NPU SDPA tensor-bool rejection evidence is required before speed admission",
        )
        require(
            row.get("functional_status") == "PASS"
            and row.get("functional_all_gradients_status") == "PASS"
            and (row.get("functional_complete_state_status") == "PASS")
            and (row.get("state_schema_complete") is True)
            and (len(row.get("functional_artifact_sha256", "")) == 64),
            "matching full-gradient/state functional gate is required",
        )
        require(
            row.get("lease_id") == self.protocol["lease_id"]
            and row.get("physical_mask") == self.protocol["physical_mask"],
            "external resource lease and physical mask must match",
        )
        require(
            row.get("logical_device") == 0
            and row.get("visible_device_count") == 1
            and (row.get("actual_pid_on_physical_device") is True),
            "logical placement and actual physical-device PID evidence are required",
        )
        require(
            row.get("placement_tensor_count", 0) > 0 and row.get("fallback_count") == 0,
            "complete actual device placement and fallback0 are required",
        )
        require(
            row.get("threads") == 2 and row.get("affinity") == self.protocol["affinity"],
            "actual thread/CPU-affinity resource contract changed",
        )
        warmup = row.get("warmup_updates", 0)
        measured = row.get("measured_updates", 0)
        require(
            isinstance(warmup, int)
            and warmup >= 4
            and isinstance(measured, int)
            and (measured >= 10),
            "require full-cycle >=4 warmup and >=10 measured updates",
        )
        require(
            row.get("phase_barrier_steps") == list(range(1, warmup + measured + 1)),
            "one common postclip-preoptimizer strongsync for every update is required",
        )
        require(
            sorted(row.get("warmed_shapes", [])) == sorted(self.protocol["shape_inventory"]),
            "full microbatch shape inventory was not warmed",
        )
        require(
            row.get("measured_update_steps") == list(range(warmup + 1, warmup + measured + 1)),
            "all measured optimizer updates must be complete and consecutive",
        )
        durations = row.get("seconds_per_update", [])
        require(
            len(durations) == measured
            and all(
                (isinstance(x, (int, float)) and math.isfinite(x) and (x > 0) for x in durations)
            ),
            "synchronized update durations are incomplete or invalid",
        )
        require(row.get("optimizer_skipped") == 0, "skipped optimizer attempts cannot be admitted")
        require(
            row.get("synchronized_begin_end") is True,
            "both boundaries must synchronize the requested runtime",
        )
        require(
            row.get("device_readbacks") == 0 and row.get("observation_save_eval_inside") == 0,
            "D2H/observation/save/eval entered a measured window",
        )
        require(
            row.get("jit_before") is not None
            and row.get("jit_before") == row.get("jit_after")
            and (row.get("vendor_compile_before") is not None)
            and (row.get("vendor_compile_before") == row.get("vendor_compile_after")),
            "native and vendor compilation evidence must be stable",
        )
        require(
            row.get("logging_nan_inf_filter") is False
            and row.get("logging_eval_save_disabled") is True
            and (row.get("inputs_preplaced") is True),
            "observation-free configuration and preplaced inputs are required",
        )
        require(
            row.get("sample_count", 0) > 0 and row.get("unmasked_token_count", 0) > 0,
            "complete sample/token counts are required",
        )
        attempt["status"] = "PASS"
        self.admissions.append(row)

    def finish(self):
        if len(self.admissions) != 20:
            raise ValueError(
                "all20 alternating independently admitted cohorts (10 pairs) are required"
            )
        summaries = {}
        for runtime in ("torch", "jittor"):
            rows = [r for r in self.admissions if r["runtime"] == runtime]
            values = [x for r in rows for x in r["seconds_per_update"]]
            seconds = sum(values)
            summaries[runtime] = {
                "cohorts": len(rows),
                "updates": len(values),
                "mean_seconds_per_update": statistics.mean(values),
                "median_seconds_per_update": statistics.median(values),
                "samples_per_second": sum((r["sample_count"] for r in rows)) / seconds,
                "unmasked_tokens_per_second": sum((r["unmasked_token_count"] for r in rows))
                / seconds,
            }
        return {
            "status": "CALLER_EVIDENCE_GATES_COMPLETE",
            "performance_claim": False,
            "scope": "ledger candidate requires externally verified artifacts and actual resource authorization; it never executes hardware",
            "alternating_order": self.order,
            "admitted_cohorts": 20,
            "admitted_pairs": 10,
            "runtime_summaries": summaries,
            "attempts": self.attempts,
        }


def loaded_blas_threads(target=None):
    paths = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line and any((key in line.lower() for key in ["openblas", "mkl_", "blis"]))
        }
    )
    results = []
    for path in paths:
        lib = ctypes.CDLL(path)
        row = {"path": path, "pools": []}
        for prefix, suffix in [
            ("openblas_", ""),
            ("openblas_", "64_"),
            ("scipy_openblas_", ""),
            ("scipy_openblas_", "64_"),
        ]:
            getter = prefix + "get_num_threads" + suffix
            setter = prefix + "set_num_threads" + suffix
            if not hasattr(lib, getter):
                continue
            query = getattr(lib, getter)
            query.argtypes = []
            query.restype = ctypes.c_int
            item = {"getter": getter, "before": int(query())}
            if target is not None:
                assert hasattr(lib, setter), (path, getter, "missing matched setter")
                change = getattr(lib, setter)
                change.argtypes = [ctypes.c_int]
                change.restype = None
                change(int(target))
            item["after"] = int(query())
            if target is not None:
                assert item["after"] == int(target), (path, item)
            row["pools"].append(item)
        if row["pools"]:
            results.append(row)
    return results


class PersistentStepControl:
    """One exclusive device grant,4 warm pairs+10 steady pairs,2 finalizers.

    Models/contexts remain resident; this cannot establish single-model HBM.
    The old AdmissionSequence20-cohort contract remains a separate protocol.
    """

    def __init__(self):
        self.order = [(runtime, step) for step in range(1, 15) for runtime in ("torch", "jittor")]
        self.ready_workers = {}
        self.rows = []
        self.active = None
        self.finalized = []

    def ready(self, runtime, row):
        if runtime not in ("torch", "jittor") or runtime in self.ready_workers:
            raise ValueError("invalid or duplicate persistent worker")
        if not row.get("shadow_complete") or not row.get("initial_reset_exact"):
            raise ValueError("complete functional shadow and exact initial reset required")
        if (
            row.get("phase_label") != "synchronized training phases"
            or row.get("attention") != "eager"
        ):
            raise ValueError("explicit eager synchronized phase protocol required")
        if len(row.get("shape_inventory", [])) != 8 or len(row.get("initial_sha256", "")) != 64:
            raise ValueError("full8 shape inventory and frozen initial weights required")
        if self.ready_workers:
            other = next(iter(self.ready_workers.values()))
            for key in ("shape_inventory", "initial_sha256"):
                if row[key] != other[key]:
                    raise ValueError("paired worker protocol differs: " + key)
        self.ready_workers[runtime] = row

    def grant(self, runtime, step):
        if len(self.ready_workers) != 2 or self.active is not None:
            raise ValueError("workers not ready or exclusive grant overlaps")
        if len(self.rows) >= len(self.order) or (runtime, step) != self.order[len(self.rows)]:
            raise ValueError("wrong runtime/step or duplicate grant")
        self.active = (runtime, step)

    def done(self, runtime, step, row):
        if self.active != (runtime, step) or not row.get("synchronized"):
            raise ValueError("step completion lacks matching synchronized grant")
        if len(row.get("shapes", [])) != 2 or any(
            (x not in self.ready_workers[runtime]["shape_inventory"] for x in row["shapes"])
        ):
            raise ValueError("microbatch shape inventory changed")
        duration = row.get("seconds")
        if step <= 4:
            if duration is not None:
                raise ValueError("warmup is not a steady sample")
        elif not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("invalid synchronized steady duration")
        peers = [x for x in self.rows if x["step"] == step]
        if peers and any((row.get(key) != peers[0].get(key) for key in ("shapes", "sample_ids"))):
            raise ValueError("paired update input signature/order differs")
        self.rows.append(dict(row, runtime=runtime, step=step))
        self.active = None
        if step == 4 and runtime == "jittor":
            for name in ("torch", "jittor"):
                warmed = {
                    shape
                    for item in self.rows
                    if item["runtime"] == name
                    for shape in item["shapes"]
                }
                if warmed != set(self.ready_workers[name]["shape_inventory"]):
                    raise ValueError("warmup did not cover full input inventory")

    def finalize(self, runtime):
        if self.active is not None or len(self.rows) != 28:
            raise ValueError("finalization/readback before both training trajectories finish")
        if runtime != ("torch", "jittor")[len(self.finalized)]:
            raise ValueError("finalizers must be serialized")
        self.finalized.append(runtime)

    def finish(self):
        if self.finalized != ["torch", "jittor"]:
            raise ValueError("incomplete persistent protocol")
        return {
            "protocol": "persistent-two-worker-v1",
            "warmup_pairs": 4,
            "steady_pairs": 10,
            "updates_per_runtime": 14,
            "steady_samples": 20,
            "rows": self.rows,
            "dual_resident_models": True,
            "single_model_hbm_claim": False,
            "performance_claim": False,
        }


def compare_arrays(
    left_path, right_path, *, same_runtime=False, gradients=False, atol=0.0002, rtol=0.002
):
    """Compare every archive key; raw schema differences are failures.

    Same-runtime state must be strict-exact, including RNG. Cross-runtime RNG
    byte encodings are explicitly reported, never treated as interoperable.
    Hyperparameters and non-floating metadata always use exact comparison.
    """
    import numpy as np

    (failures, checks, opaque_rng) = ([], [], [])
    with np.load(left_path, allow_pickle=False) as left, np.load(
        right_path, allow_pickle=False
    ) as right:
        left_only = sorted(set(left.files) - set(right.files))
        right_only = sorted(set(right.files) - set(left.files))
        for side, names in (("left-only", left_only), ("right-only", right_only)):
            failures.extend(({"name": name, "reason": side} for name in names))
        for name in sorted(set(left.files) & set(right.files)):
            (a, b) = (left[name], right[name])
            if not same_runtime and (
                name == "rng/torch"
                or (name.startswith("rng/npu/") and name.rsplit("/", 1)[-1].isdigit())
            ):
                if not (
                    a.dtype == b.dtype == np.uint8 and a.ndim == b.ndim == 1 and a.size and b.size
                ):
                    failures.append({"name": name, "reason": "invalid opaque RNG state"})
                opaque_rng.append(
                    {
                        "name": name,
                        "left_bytes": int(a.size),
                        "right_bytes": int(b.size),
                        "value_compared": False,
                    }
                )
                continue
            if a.dtype != b.dtype:
                failures.append(
                    {"name": name, "reason": "dtype", "left": str(a.dtype), "right": str(b.dtype)}
                )
                continue
            if a.shape != b.shape:
                failures.append(
                    {"name": name, "reason": "shape", "left": list(a.shape), "right": list(b.shape)}
                )
                continue
            numeric = a.dtype.kind in "fiu" and b.dtype.kind in "fiu"
            floating_math = numeric and (
                gradients
                or name.startswith(("model/", "optimizer/state/"))
                or name == "reported_train_loss"
            )
            tolerance = not same_runtime and floating_math and (not name.endswith("/step"))
            if numeric and (not (np.isfinite(a).all() and np.isfinite(b).all())):
                failures.append({"name": name, "reason": "nonfinite"})
                continue
            passed = (
                bool(np.allclose(a, b, atol=atol, rtol=rtol))
                if tolerance
                else bool(np.array_equal(a, b))
            )
            error = (
                float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                if numeric and a.size
                else None
            )
            row = {
                "name": name,
                "pass": passed,
                "max_abs": error,
                "strict_exact": bool(np.array_equal(a, b)),
            }
            checks.append(row)
            if not passed:
                failures.append(dict(row, reason="value"))
    return {
        "status": "FAIL" if failures else "PASS",
        "schema_complete": not (left_only or right_only),
        "left_only": left_only,
        "right_only": right_only,
        "compared_arrays": len(checks),
        "failures": failures,
        "checks": checks,
        "opaque_rng": opaque_rng,
        "same_runtime_strict": same_runtime,
        "atol": 0 if same_runtime else atol,
        "rtol": 0 if same_runtime else rtol,
        "performance_claim": False,
    }


def compare_training_outputs(left_dir, right_dir, *, atol=0.0002, rtol=0.002):
    """Check complete state schema, every expected gradient, and all losses."""
    import json
    import numpy as np
    from pathlib import Path

    (left, right) = (Path(left_dir), Path(right_dir))
    states = compare_arrays(
        left / "functional-state.npz", right / "functional-state.npz", atol=atol, rtol=rtol
    )
    gradients = compare_arrays(
        left / "functional-gradients.npz",
        right / "functional-gradients.npz",
        gradients=True,
        atol=atol,
        rtol=rtol,
    )
    (within, completeness) = ({}, [])
    for name, directory in (("left", left), ("right", right)):
        summary = json.loads((directory / "functional-summary.json").read_text())
        expected = {
            f"{phase}/{index:02d}/{parameter}"
            for (phase, count) in (
                ("micro", summary["microbatches"]),
                ("postclip", summary["global_step"]),
            )
            for index in range(count)
            for parameter in summary["trainable_names"]
        }
        with np.load(directory / "functional-gradients.npz") as archive:
            if any(
                (
                    not any(
                        (
                            np.any(archive[key] != 0)
                            for key in archive.files
                            if key.endswith("/" + parameter)
                        )
                    )
                    for parameter in summary["trainable_names"]
                )
            ):
                completeness.append({"side": name, "reason": "all-zero complete gradient history"})
            if not expected or set(archive.files) != expected:
                completeness.append({"side": name, "reason": "incomplete gradient inventory"})
        with np.load(directory / "functional-state.npz") as archive:
            if len(archive.files) != summary["state_array_count"]:
                completeness.append({"side": name, "reason": "incomplete final state inventory"})
        within[name] = compare_arrays(
            directory / "functional-state.npz", directory / "plain-state.npz", same_runtime=True
        )
        if (directory / "functional-initial-state.npz").exists():
            reset = compare_arrays(
                directory / "functional-initial-state.npz",
                directory / "plain-initial-state.npz",
                same_runtime=True,
            )
            if reset["status"] != "PASS":
                completeness.append({"side": name, "reason": "initial model/RNG reset differs"})
    a = np.load(left / "functional-losses.npy", allow_pickle=False)
    b = np.load(right / "functional-losses.npy", allow_pickle=False)
    losses_pass = bool(
        a.dtype == b.dtype
        and a.shape == b.shape
        and a.size
        and np.isfinite(a).all()
        and np.isfinite(b).all()
        and np.allclose(a, b, atol=atol, rtol=rtol)
    )
    if not losses_pass:
        completeness.append({"reason": "microbatch losses differ or are incomplete"})
    passed = not completeness and all(
        (item["status"] == "PASS" for item in (states, gradients, *within.values()))
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "full_state": states,
        "all_gradients": gradients,
        "within_runtime": within,
        "micro_losses_pass": losses_pass,
        "inventory_failures": completeness,
        "state_schema_complete": states["schema_complete"],
        "functional_complete_state_status": states["status"],
        "functional_all_gradients_status": gradients["status"] if not completeness else "FAIL",
        "performance_claim": False,
        "npu_admitted": False,
    }


def capture_process_identity(process):
    """Record an owned Linux session leader without trusting a reusable PID."""
    pid = process.pid
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return {"pid": pid, "start_ticks": None, "session": None, "process_group": None}
    result = {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "session": int(fields[3]),
        "process_group": int(fields[2]),
    }
    if result["session"] != pid or result["process_group"] != pid:
        raise ValueError("owned worker must be a new process session leader")
    return result


def _matches_process_identity(identity):
    try:
        fields = (
            (Path("/proc") / str(identity["pid"]) / "stat").read_text().rsplit(")", 1)[1].split()
        )
    except FileNotFoundError:
        return False
    return (
        identity["start_ticks"] == int(fields[19])
        and identity["session"] == int(fields[3])
        and identity["process_group"] == int(fields[2])
    )


def accept_worker_connection(listener, process, timeout):
    """Bound accept/authentication and detect a worker exiting before connection."""
    import queue
    import threading

    outcome = queue.Queue(maxsize=1)

    def accept():
        try:
            outcome.put((True, listener.accept()))
        except Exception as error:
            outcome.put((False, error))

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("owned worker did not connect/authenticate before deadline")
        try:
            success, value = outcome.get(timeout=min(remaining, 0.05))
        except queue.Empty:
            if process.poll() is not None:
                raise RuntimeError("owned worker exited before connecting: %s" % process.returncode)
            continue
        if not success:
            raise value
        return value


def cleanup_owned_workers(processes, identities, grace=10):
    """Signal only a still-matching owned session, then reap boundedly."""
    import os
    import signal

    records = []
    for name, process in processes.items():
        identity = identities[name]
        row = dict(runtime=name, identity=identity, signals=[])
        if process.poll() is None:
            if _matches_process_identity(identity):
                try:
                    os.killpg(identity["process_group"], signal.SIGTERM)
                    row["signals"].append("SIGTERM")
                except ProcessLookupError:
                    pass
            else:
                row["identity_mismatch_refused_signal"] = True
                records.append(row)
                continue
        try:
            row["returncode"] = process.wait(timeout=grace)
        except __import__("subprocess").TimeoutExpired:
            if _matches_process_identity(identity):
                os.killpg(identity["process_group"], signal.SIGKILL)
                row["signals"].append("SIGKILL")
                row["returncode"] = process.wait(timeout=grace)
            else:
                row["identity_mismatch_refused_signal"] = True
        records.append(row)
    return records


def checked_lab_path(path, lab_root, repo_root):
    from pathlib import Path

    (path, lab, repo) = (Path(value).resolve() for value in (path, lab_root, repo_root))
    if not path.is_relative_to(lab) or path.is_relative_to(repo):
        raise ValueError("artifact/cache path must be inside lab and outside checkout")
    return path


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lab-root", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()
    target = checked_lab_path(args.output, args.lab_root, args.repo_root)
    if target.exists():
        raise FileExistsError(target)
    report = compare_training_outputs(args.left, args.right)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for (key, value) in report.items()
                if key not in ("full_state", "all_gradients", "within_runtime")
            }
        )
    )
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()

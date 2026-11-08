"""Serialize matching benchmark calls from two persistent independent workers.

The caller owns device allocation, runtime isolation and source/environment pins.
This controller records scheduling evidence only; accept performance only after
both complete reports and full tensor archives pass the separate comparator.
"""
import argparse
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time

READY = "BENCHMARK_CONTROL_READY "


def _start_ticks(pid):
    try:
        return Path("/proc/%d/stat" % pid).read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


class Worker:
    def __init__(self, name, spec, output):
        if not isinstance(spec.get("argv"), list) or not spec["argv"]:
            raise ValueError("worker argv must be a nonempty list")
        self.name, self.spec = name, spec
        self.events = queue.Queue()
        self.log = (output / (name + ".log")).open("x", encoding="utf-8")
        env = os.environ.copy()
        env.update(spec.get("env", {}))
        for key in spec.get("unset_env", []):
            env.pop(key, None)
        self.started = time.time()
        self.process = subprocess.Popen(spec["argv"], cwd=spec["cwd"], env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            start_new_session=True)
        self.start_ticks = _start_ticks(self.process.pid)
        if self.start_ticks is None:
            raise RuntimeError("worker exited before process identity capture")
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                self.log.write(line)
                self.log.flush()
                if line.startswith(READY):
                    try:
                        event = json.loads(line[len(READY):])
                    except ValueError as error:
                        self.events.put(("error", str(error)))
                        continue
                    self.events.put(("ready", event))
            self.events.put(("exit", self.process.wait()))
        except BaseException as error:
            self.events.put(("error", repr(error)))
        finally:
            self.log.close()

    def next(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("paired benchmark deadline exhausted")
        try:
            kind, event = self.events.get(timeout=remaining)
        except queue.Empty as error:
            raise TimeoutError("worker %s did not reach its next boundary" % self.name) from error
        if kind == "error":
            raise RuntimeError("worker %s protocol: %s" % (self.name, event))
        if kind == "exit" and event != 0:
            raise RuntimeError("worker %s exited with %s" % (self.name, event))
        return kind, event

    def admit(self, call):
        self.process.stdin.write(json.dumps({"go": call}) + "\n")
        self.process.stdin.flush()

    def stop(self):
        # Only the original child and its own new process group may be signalled.
        if self.process.poll() is None and _start_ticks(self.process.pid) == self.start_ticks:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if _start_ticks(self.process.pid) == self.start_ticks:
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10)
        self.reader.join(timeout=5)

    def provenance(self):
        return {"pid": self.process.pid, "start_ticks": self.start_ticks,
                "started": self.started, "returncode": self.process.poll(),
                "argv": self.spec["argv"], "cwd": self.spec["cwd"]}


def _validate_event(event, expected_call):
    if set(event) != {"call", "phase", "operation", "slot", "use_cache"}:
        raise RuntimeError("unexpected admission event fields")
    if type(event["call"]) is not int or event["call"] != expected_call:
        raise RuntimeError("admission call sequence changed")
    if event["phase"] not in ("first_prefill", "prefill_warmup", "prefill",
            "first_generate", "generate_warmup", "generate", "validation",
            "decode_setup", "first_decode", "decode_warmup", "decode", "decode_validation"):
        raise RuntimeError("unknown benchmark phase")
    if event["operation"] not in ("prefill", "generate", "decode") or type(event["slot"]) is not int or event["slot"] < 0:
        raise RuntimeError("invalid operation or input slot")
    if type(event["use_cache"]) is not bool:
        raise RuntimeError("cache mode must be a boolean")


def _device_snapshot(raw, physical):
    lines = raw.splitlines()
    header = re.compile(r'^\|\s*%d\s+910B3\s*\|\s*OK\s*\|' % physical)
    matching = [i for i, line in enumerate(lines) if header.search(line)]
    if len(matching) != 1 or matching[0] + 1 >= len(lines):
        raise RuntimeError("allocated 910B3 health or HBM row is unavailable")
    hbm = re.search(r'(\d+)\s*/\s*(\d+)\s*\|$', lines[matching[0] + 1])
    if hbm is None:
        raise RuntimeError("allocated device HBM usage is unavailable")
    used, total = map(int, hbm.groups())
    if not 0 <= used < total:
        raise RuntimeError("allocated device HBM usage is invalid")
    pattern = r'^\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+)\|\s*(\d+)\s*\|'
    processes = {int(pid): int(memory) for npu, chip, pid, name, memory
                 in re.findall(pattern, raw, re.M) if int(npu) == physical}
    empty = bool(re.search(r'No running processes found in NPU\s+%d\b' % physical, raw))
    if not processes and not empty:
        raise RuntimeError("allocated device process table is incomplete")
    if processes and empty:
        raise RuntimeError("allocated device process table is contradictory")
    return {"physical_device": physical, "used_mib": used, "total_mib": total,
            "free_mib": total - used, "process_memory_mib": processes}


def _guard_device(spec, workers, output, stage, *, startup_headroom=True):
    physical = spec.get("physical_device")
    if physical is None:
        return None
    if type(physical) is not int or physical < 0:
        raise ValueError("physical_device must be a nonnegative integer")
    for item in spec["workers"].values():
        if (item.get("env", {}).get("ASCEND_RT_VISIBLE_DEVICES") != str(physical) or
                "ASCEND_RT_VISIBLE_DEVICES" in item.get("unset_env", [])):
            raise ValueError("both workers must expose exactly the allocated physical device")
    proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=60)
    (output / (stage + ".npu-smi.txt")).write_text(proc.stdout + proc.stderr)
    if proc.returncode:
        raise RuntimeError("npu-smi resource preflight failed")
    snapshot = _device_snapshot(proc.stdout, physical)
    allowed = {worker.process.pid for worker in workers.values()}
    actual = set(snapshot["process_memory_mib"])
    if actual != allowed:
        raise RuntimeError("allocated NPU processes differ from verified resident workers")
    for worker in workers.values():
        if _start_ticks(worker.process.pid) != worker.start_ticks:
            raise RuntimeError("resident worker process identity changed")
    if allowed and startup_headroom:
        # Conservative startup bound, not a prediction of full workload peaks.
        required = sum(snapshot["process_memory_mib"].values()) + 4096
        if snapshot["free_mib"] < required:
            raise RuntimeError("insufficient HBM for a second resident model plus 4 GiB headroom")
        snapshot["second_model_startup_bound_mib"] = required
    return snapshot


def run_pair(spec, output, timeout):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    deadline = time.monotonic() + timeout
    workers, boundaries, records = {}, {}, []
    snapshots = []
    result = {"passed": False, "performance_accepted": False,
        "scope": "serialized counterbalanced calls; full output comparator required separately",
        "memory_scope": "both models resident; device-wide free memory is not individual model HBM",
        "interrupted_jittor_cache_requires_audit_before_reuse": False}
    try:
        for name in ("torch", "jittor"):
            snapshots.append(_guard_device(spec, workers, output, "before-" + name))
            workers[name] = Worker(name, spec["workers"][name], output)
            boundaries[name] = workers[name].next(deadline)
            if boundaries[name][0] != "ready":
                raise RuntimeError("worker %s exited before its first call" % name)
        call = 1
        while True:
            if boundaries["torch"][0] == boundaries["jittor"][0] == "exit":
                break
            if any(kind != "ready" for kind, _ in boundaries.values()):
                raise RuntimeError("workers reached unequal terminal boundaries")
            event = boundaries["torch"][1]
            _validate_event(event, call)
            if event != boundaries["jittor"][1]:
                raise RuntimeError("worker admission events disagree")
            # These boundaries are outside both workers' measured calls.
            # Later foreign work invalidates every subsequent admission.
            snapshots.append(_guard_device(
                spec, workers, output, "before-pair-%d" % call,
                startup_headroom=False))
            order = ("torch", "jittor") if call % 2 else ("jittor", "torch")
            record = {"event": event, "order": list(order), "completed": []}
            for name in order:
                admitted = time.time()
                workers[name].admit(call)
                boundaries[name] = workers[name].next(deadline)
                record["completed"].append({"worker": name, "admitted": admitted,
                    "next_boundary": time.time(), "kind": boundaries[name][0]})
            live_workers = {name: worker for name, worker in workers.items()
                            if worker.process.poll() is None}
            snapshots.append(_guard_device(
                spec, live_workers, output, "after-pair-%d" % call,
                startup_headroom=False))
            records.append(record)
            with (output / "schedule.jsonl").open("a") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            call += 1
        result.update(passed=True, calls=len(records),
            steady_prefill_pairs=sum(x["event"]["phase"] == "prefill" for x in records),
            steady_generate_pairs=sum(x["event"]["phase"] == "generate" for x in records),
            steady_decode_pairs=sum(x["event"]["phase"] == "decode" for x in records))
        return result
    except BaseException as error:
        result["error"] = repr(error)
        result["interrupted_jittor_cache_requires_audit_before_reuse"] = (
            "jittor" in workers and workers["jittor"].process.poll() is None)
        raise
    finally:
        for worker in workers.values():
            worker.stop()
        result["resource_snapshots"] = snapshots
        result["workers"] = {name: worker.provenance() for name, worker in workers.items()}
        (output / "execution.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="JSON workers mapping for torch and jittor")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=5400)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    output = Path(args.output).expanduser().resolve()
    lab = Path(os.environ.get("JITTOR_LAB_ROOT", "")).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if not os.environ.get("JITTOR_LAB_ROOT") or lab not in output.parents or checkout in output.parents:
        parser.error("output must be under JITTOR_LAB_ROOT outside the checkout")
    spec = json.loads(Path(args.spec).read_text())
    result = run_pair(spec, output, args.timeout)
    print("PAIRED_CONTROL_RESULT " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

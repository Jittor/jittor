"""Persistent two-worker tiny eager Trainer protocol; CPU-qualified preparation only.

No checkpoint downloads, evaluation or save/checkpoint timing. External NPU
resource authorization is mandatory. Dual-resident models are not a single-model
HBM comparison. Complete gradients belong to functional shadows, not timed steps.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from multiprocessing.connection import Listener
from _trainer_step_protocol import PersistentStepControl, checked_lab_path, compare_training_outputs


from _trainer_step_protocol import (
    accept_worker_connection,
    capture_process_identity,
    cleanup_owned_workers,
)


def receive(connection, timeout):
    if not connection.poll(timeout):
        raise TimeoutError("persistent worker did not reach the next protocol boundary")
    return connection.recv()


def run_controller(args):
    (lab, repo) = (Path(args.lab_root).resolve(), Path(args.repo_root).resolve())
    run = checked_lab_path(args.run_dir, lab, repo)
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    initial = checked_lab_path(args.initial, lab, repo)
    if not initial.is_file():
        raise FileNotFoundError(initial)
    if args.device == "npu" and (
        not args.lease_id or not os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdigit()
    ):
        raise ValueError("an externally assigned single physical mask and lease ID are required")
    source = json.loads(Path(args.source_manifest).read_text())
    for path, digest in source["files"].items():
        if hashlib.sha256((repo / path).read_bytes()).hexdigest() != digest:
            raise ValueError("frozen runtime source differs: " + path)
    base_environments = (
        json.loads(Path(args.environment_json).read_text()) if args.environment_json else {}
    )
    token = os.urandom(32)
    listener = Listener(("127.0.0.1", 0), authkey=token)
    (processes, connections, handles) = ({}, {}, {})
    events = []
    identities = {}
    control = PersistentStepControl()

    def event(row):
        row = dict(row, epoch=time.time())
        events.append(row)
        with (run / "events.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    try:
        for runtime, interpreter in (("torch", args.torch_python), ("jittor", args.jittor_python)):
            env = os.environ.copy()
            env.update(base_environments.get(runtime, {}))
            env.update(
                PYTHONDONTWRITEBYTECODE="1",
                PYTHONNOUSERSITE="1",
                TORCH_DEVICE_BACKEND_AUTOLOAD="0",
                TOKENIZERS_PARALLELISM="false",
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="2",
                NUMEXPR_NUM_THREADS="2",
                JITTOR_SOURCE_ROOT=str(repo),
                JITTOR_PERF_CONTROL_TOKEN=token.hex(),
                CUDA_VISIBLE_DEVICES="",
                JT_BACKEND_FALLBACK="error",
                DISABLE_MULTIPROCESSING="1",
                JT_USE_PARALLEL_OP_COMPILER="0",
                JT_BUILD_USE_MPI="0",
            )
            for key in (
                "HOME",
                "XDG_CACHE_HOME",
                "HF_HOME",
                "TORCH_HOME",
                "ASCEND_CACHE_PATH",
                "ASCEND_WORK_PATH",
                "CCACHE_DIR",
            ):
                directory = run / ("env-" + runtime) / key.lower()
                directory.mkdir(parents=True)
                env[key] = str(directory)
            temporary = (
                lab
                / "_tmp"
                / ("ts-" + hashlib.sha256((str(run) + runtime).encode()).hexdigest()[:8])
            )
            temporary.mkdir(parents=True, exist_ok=False)
            if len(str(temporary).encode()) + 34 >= 108:
                raise ValueError("temporary IPC path is too long")
            env["TMPDIR"] = str(temporary)
            vendor_paths = []
            if args.device == "npu":
                if not args.cann_prefix:
                    raise ValueError("explicit CANN prefix required for native SDK isolation")
                prefix = Path(args.cann_prefix).resolve()
                vendor_paths = [
                    item
                    for item in env.get("PYTHONPATH", "").split(os.pathsep)
                    if item and Path(item).resolve().is_relative_to(prefix)
                ]
            env["PYTHONPATH"] = os.pathsep.join(vendor_paths)
            env["JITTOR_TORCH_SHIM"] = "1" if runtime == "jittor" else "0"
            if runtime == "jittor":
                env["PYTHONPATH"] = os.pathsep.join(
                    [str(repo / "python"), str(repo / "adapters")]
                    + ([str(Path(args.adapter_site).resolve())] if args.adapter_site else [])
                    + vendor_paths
                )
                home = checked_lab_path(args.jittor_home, lab, repo)
                env["JITTOR_HOME"] = str(home)
                env["cache_name"] = args.cache_name
                env["JT_BUILD_CACHE_NAME"] = args.cache_name
                env["JT_BACKEND"] = "cpu" if args.device == "cpu" else "acl"
                env["JT_USE_CUDA"] = "0" if args.device == "cpu" else "1"
            else:
                directory = run / ("env-" + runtime) / "unused-jittor-home"
                directory.mkdir()
                env["JITTOR_HOME"] = str(directory)
            if args.device == "cpu":
                env["ASCEND_RT_VISIBLE_DEVICES"] = ""
            command = [
                interpreter,
                "-B",
                str(Path(__file__).resolve()),
                "--worker-runtime",
                runtime,
                "--device",
                args.device,
                "--run-dir",
                str(run),
                "--repo-root",
                str(repo),
                "--lab-root",
                str(lab),
                "--initial",
                str(initial),
                "--source-manifest",
                args.source_manifest,
                "--control-port",
                str(listener.address[1]),
            ]
            handles[runtime] = (run / (runtime + ".log")).open("x")
            process = subprocess.Popen(
                command,
                cwd=repo,
                env=env,
                stdout=handles[runtime],
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes[runtime] = process
            identities[runtime] = capture_process_identity(process)
            event(
                {
                    "event": "START",
                    "runtime": runtime,
                    "pid": process.pid,
                    "process_identity": identities[runtime],
                    "command": command,
                    "source_manifest": args.source_manifest,
                }
            )
            connection = accept_worker_connection(listener, process, args.connect_timeout)
            connections[runtime] = connection
            shadow = receive(connection, args.timeout)
            if (
                shadow.get("event") != "SHADOW_DONE"
                or shadow.get("complete_gradient_arrays") != 252
            ):
                raise ValueError("complete functional shadow is missing")
            event(shadow)
            ready = receive(connection, args.timeout)
            if ready.get("event") != "READY":
                raise ValueError("worker did not finish its exact initial reset")
            control.ready(runtime, ready)
            event(ready)
        for step in range(1, 15):
            for runtime in ("torch", "jittor"):
                control.grant(runtime, step)
                event({"event": "GRANT", "runtime": runtime, "step": step})
                connections[runtime].send({"event": "GRANT", "step": step})
                row = receive(connections[runtime], args.timeout)
                if row.get("event") != "DONE":
                    raise ValueError("worker omitted synchronized update completion")
                control.done(runtime, step, row)
                event(row)
        for runtime in ("torch", "jittor"):
            control.finalize(runtime)
            connections[runtime].send({"event": "FINALIZE"})
            row = receive(connections[runtime], args.timeout)
            if row.get("event") != "FINISHED" or row.get("status") != "PASS":
                raise ValueError("same-runtime full-state strict0 shadow comparison failed")
            event(row)
            if processes[runtime].wait(timeout=60) != 0:
                raise RuntimeError("worker exited unsuccessfully")
        report = control.finish()
        comparison = compare_training_outputs(run / "torch", run / "jittor")
        report.update(
            status="PASS" if comparison["status"] == "PASS" else "FAIL",
            device=args.device,
            comparison=comparison,
            lease_id=args.lease_id,
            scope="tiny eager synchronized phases; complete gradients from shadows only",
            npu_performance_admitted=False,
            determinism_policies={
                row["runtime"]: row["determinism"] for row in events if row.get("event") == "READY"
            },
            common_determinism_policy_claim=False,
            vendor_compilation_gate="external evidence still required before performance claim",
        )
        for path, digest in source["files"].items():
            if hashlib.sha256((repo / path).read_bytes()).hexdigest() != digest:
                raise ValueError("runtime source changed during protocol: " + path)
        (run / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        raise SystemExit(0 if report["status"] == "PASS" else 1)
    except Exception as error:
        event({"event": "FAIL", "error": repr(error), "performance_claim": False})
        (run / "failure.json").write_text(json.dumps(events, indent=2) + "\n")
        raise
    finally:
        for connection in connections.values():
            connection.close()
        listener.close()
        cleanup = cleanup_owned_workers(processes, identities)
        (run / "cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")
        for handle in handles.values():
            handle.close()


def create_initial(args):
    import importlib.util
    import numpy as np

    if args.device != "cpu" or importlib.util.find_spec("torch_npu") is not None:
        raise ValueError("initial weights require the clean native CPU interpreter")
    import torch
    from transformers import BertConfig, BertForSequenceClassification
    from peft import LoraConfig, get_peft_model

    if not hasattr(torch, "_C") or "jittor" in sys.modules:
        raise ValueError("initial weights must come from real CPU Torch")
    torch.manual_seed(321)
    config = BertConfig(
        vocab_size=25,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_labels=2,
        pad_token_id=1,
        _attn_implementation="eager",
    )
    model = get_peft_model(
        BertForSequenceClassification(config),
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["query", "value"],
            modules_to_save=["classifier"],
            task_type="SEQ_CLS",
        ),
    )
    target = checked_lab_path(args.initial, args.lab_root, args.repo_root)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    np.savez(
        target, **{name: value.detach().clone().cpu().numpy().copy() for (name, value) in tensors}
    )
    print(
        json.dumps(
            {
                "status": "INITIAL_ONLY",
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "npu_executed": False,
                "tensor_count": len(tensors),
            }
        )
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "npu"), required=True)
    for name in ("repo-root", "lab-root", "initial"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--source-manifest")
    parser.add_argument("--create-initial", action="store_true")
    parser.add_argument("--torch-python")
    parser.add_argument("--jittor-python")
    parser.add_argument("--adapter-site")
    parser.add_argument("--jittor-home")
    parser.add_argument("--cache-name", default="trainer_step_owned")
    parser.add_argument("--environment-json")
    parser.add_argument("--cann-prefix")
    parser.add_argument("--lease-id")
    parser.add_argument("--timeout", type=float, default=5400)
    parser.add_argument("--connect-timeout", type=float, default=60)
    parser.add_argument("--worker-runtime", choices=("torch", "jittor"), help=argparse.SUPPRESS)
    parser.add_argument("--control-port", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout <= 0 or args.connect_timeout <= 0:
        parser.error("protocol deadlines must be positive")
    if (
        not args.worker_runtime
        and not args.create_initial
        and (not all((args.torch_python, args.jittor_python, args.jittor_home)))
    ):
        parser.error("controller requires two explicit interpreters and an owned Jittor cache")
    return args


def run_worker(args):
    """CPU functional equivalence only; this tool does not admit NPU performance."""
    import collections
    import hashlib
    import json
    import os
    import random
    import sys
    import time
    from contextlib import ExitStack
    from pathlib import Path

    RUNTIME = args.worker_runtime
    JITTOR_MODE = RUNTIME == "jittor"
    DEVICE = "cpu" if args.device == "cpu" else "npu:0"
    ROOT = Path(args.run_dir).resolve() / RUNTIME
    TREE = Path(args.repo_root).resolve()
    from multiprocessing.connection import Client

    connection = Client(
        ("127.0.0.1", args.control_port),
        authkey=bytes.fromhex(os.environ["JITTOR_PERF_CONTROL_TOKEN"]),
    )
    ROOT.mkdir(exist_ok=False)
    import numpy as np

    if args.device == "cpu":
        assert (
            os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "") == ""
            and os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
        )
        if not JITTOR_MODE:
            import importlib.util

            assert importlib.util.find_spec("torch_npu") is None, (
                "CPU reference requires a clean Torch-only environment"
            )
    else:
        assert (
            os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdigit()
            and os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
        )
    if JITTOR_MODE:
        import jittor as jt

        assert Path(jt.__file__).resolve().is_relative_to(TREE / "python")
        assert jt.compiler.has_acl == (args.device == "npu")
    import torch

    torch.use_deterministic_algorithms(True)
    DETERMINISM = {
        "requested": True,
        "actual_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "api_fidelity": "UNIMPLEMENTED setter/getter boundary"
        if JITTOR_MODE
        else "native Torch True-before-first",
    }

    if args.device == "npu":
        if not JITTOR_MODE:
            import torch_npu  # noqa: F401 -- registers the real NPU backend

            torch.npu.matmul.allow_hf32 = False
            torch.npu.conv.allow_hf32 = False
            assert not torch.npu.matmul.allow_hf32 and not torch.npu.conv.allow_hf32
        else:
            jt.acl_allow_hf32 = False
            assert not jt.acl_allow_hf32
        assert torch.npu.device_count() == 1 and torch.npu.is_available()
        torch.npu.set_device(0)
        sdk_input = torch.ones((2, 2), device=DEVICE, dtype=torch.float32).requires_grad_(True)
        (sdk_input @ sdk_input).sum().backward()
        torch.npu.synchronize()
        del sdk_input
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
        TrainerCallback,
        PreTrainedTokenizerFast,
    )
    from tokenizers import Tokenizer, models, pre_tokenizers
    from _trainer_step_protocol import (
        UpdateTimer,
        ReadbackGuard,
        cpu_sdpa_control,
        reject_actual_npu_sdpa_mask,
        loaded_blas_threads,
    )

    UPDATES = 14
    PHASES = {"preopt"}
    PHASE_COUNTS = collections.Counter()

    def phase_sync(name):
        if name in PHASES:
            if JITTOR_MODE:
                jt.sync_all(device_sync=True)
            elif hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.synchronize()
            elif torch.cuda.is_available():
                torch.cuda.synchronize()
            PHASE_COUNTS[name] += 1

    VOCAB = {
        "[UNK]": 0,
        "[PAD]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "bright": 5,
        "useful": 6,
        "film": 7,
        "slow": 8,
        "dull": 9,
        "story": 10,
        "warm": 11,
        "funny": 12,
        "cold": 13,
        "boring": 14,
        "kind": 15,
        "moving": 16,
        "ending": 17,
        "harsh": 18,
        "empty": 19,
        "excellent": 20,
        "joyful": 21,
        "acting": 22,
        "awful": 23,
        "tedious": 24,
    }

    def _attach_peft(torch, base):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            base,
            LoraConfig(
                r=2,
                lora_alpha=4,
                lora_dropout=0.0,
                target_modules=["query", "value"],
                modules_to_save=["classifier"],
                task_type="SEQ_CLS",
            ),
        )
        return model

    def _load_initial(torch, model, path):
        with np.load(path, allow_pickle=False) as saved:
            targets = dict(_named_tensors(model))
            if set(saved.files) != set(targets):
                raise AssertionError(
                    "initial model keys differ: missing=%s extra=%s"
                    % (
                        sorted(set(targets) - set(saved.files))[:8],
                        sorted(set(saved.files) - set(targets))[:8],
                    )
                )
            with torch.no_grad():
                for name, target in targets.items():
                    target.copy_(torch.from_numpy(saved[name].copy()))

    def _named_tensors(model):
        return list(model.named_parameters()) + list(model.named_buffers())

    assert hasattr(torch, "_torch_compat_install_context") if JITTOR_MODE else hasattr(torch, "_C")
    assert ("jittor" in sys.modules) == JITTOR_MODE
    if JITTOR_MODE or args.device == "cpu":
        assert "torch_npu" not in sys.modules
    if args.device == "cpu":
        assert torch.npu.device_count() == 0 if hasattr(torch, "npu") else True
    torch.set_num_threads(2)
    assert torch.get_num_threads() == 2
    torch.use_deterministic_algorithms(True)
    SEED = 321
    LENGTHS = [22, 25, 29, 32, 37, 41, 45, 47]
    SDPA_CPU_CONTROL = cpu_sdpa_control(torch)
    NPU_READBACK_GATE = None
    if args.device == "npu":
        mask = torch.ones((2, 4), dtype=torch.int64, device=DEVICE)
        NPU_READBACK_GATE = reject_actual_npu_sdpa_mask(torch, mask)
    for key in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[key] = "2"
    BLAS = loaded_blas_threads(2)
    raw = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", model_max_length=64
    )
    BATCHES = []
    for i, length in enumerate(LENGTHS):
        texts = [
            " ".join(["bright", "useful", "film"] * (length // 3) + ["bright"] * (length % 3)),
            " ".join(["slow"] * (length - 1)),
        ]
        batch = dict(
            tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        )
        batch["labels"] = torch.tensor([i % 2, 1 - i % 2], dtype=torch.int64)
        assert tuple(batch["input_ids"].shape) == (2, length)
        batch = {key: value.to(DEVICE) for (key, value) in batch.items()}
        assert all((value.device.type == args.device for value in batch.values()))
        BATCHES.append(batch)

    def signature(batch):
        return json.dumps(
            {
                key: [list(value.shape), str(value.dtype).removeprefix("torch.")]
                for (key, value) in sorted(batch.items())
            },
            sort_keys=True,
        )

    SHAPES = [signature(batch) for batch in BATCHES]
    assert len(set(SHAPES)) == 8

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 16

        def __getitem__(self, index):
            return index

    def collate(indices):
        assert len(indices) == 2 and indices[0] % 2 == 0 and (indices[1] == indices[0] + 1)
        result = dict(BATCHES[indices[0] // 2])
        result["_sample_ids"] = tuple(indices)
        return result

    def model():
        torch.manual_seed(SEED)
        base = BertForSequenceClassification(
            BertConfig(
                _attn_implementation="eager",
                vocab_size=len(VOCAB),
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=64,
                max_position_embeddings=64,
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                num_labels=2,
                pad_token_id=VOCAB["[PAD]"],
            )
        )
        assert (
            base.config._attn_implementation == "eager"
            and type(base.bert.encoder.layer[0].attention.self).__name__ == "BertSelfAttention"
        )
        return _attach_peft(torch, base)

    initial = Path(args.initial).resolve()
    assert initial.is_file()
    INITIAL_SHA = hashlib.sha256(initial.read_bytes()).hexdigest()

    def array(value):
        return value.detach().clone().cpu().numpy().copy()

    def flatten(value, prefix, result):
        if isinstance(value, torch.Tensor):
            result[prefix] = array(value)
        elif isinstance(value, np.ndarray):
            result[prefix] = value.copy()
        elif isinstance(value, dict):
            for key in sorted(value, key=lambda x: str(x)):
                flatten(value[key], prefix + "/" + str(key), result)
        elif isinstance(value, (list, tuple)):
            result[prefix + "/@kind"] = np.asarray(type(value).__name__)
            for index, item in enumerate(value):
                flatten(item, prefix + "/" + str(index), result)
        elif value is None:
            result[prefix] = np.asarray("<none>")
        elif isinstance(value, (int, float, bool, str, np.generic)):
            result[prefix] = np.asarray(value)
        else:
            raise TypeError((prefix, type(value)))

    def full_state(trainer):
        result = {}
        flatten(trainer.model.state_dict(), "model", result)
        flatten(trainer.optimizer.state_dict(), "optimizer", result)
        flatten(trainer.lr_scheduler.state_dict(), "scheduler", result)
        flatten(torch.get_rng_state(), "rng/torch", result)
        if args.device == "npu":
            flatten(torch.npu.get_rng_state_all(), "rng/npu", result)
        flatten(random.getstate(), "rng/python", result)
        flatten(np.random.get_state(), "rng/numpy", result)
        return result

    def assert_materialized_tensor(value):
        assert value.device.type == args.device
        if args.device == "npu":
            assert int(value.device.index) == 0
        if JITTOR_MODE:
            assert value.location() == ("cpu" if args.device == "cpu" else "device")
            assert tuple(jt.core.dispatch_context([value]))[0] == (
                "cpu" if args.device == "cpu" else "acl"
            )

    def assert_model_inputs_actual(model):
        device_sync()
        tensors = [value for (_, value) in _named_tensors(model)]
        tensors += [value for batch in BATCHES for value in batch.values()]
        for value in tensors:
            assert value.device.type == args.device
            if args.device == "npu":
                assert int(value.device.index) == 0
            if JITTOR_MODE:
                assert value.location() == ("cpu" if args.device == "cpu" else "device")
                assert jt.core.dispatch_context([value])[0] == (
                    "cpu" if args.device == "cpu" else "acl"
                )
        if args.device == "npu":
            import subprocess

            snapshot = subprocess.run(
                ["npu-smi", "info"], capture_output=True, text=True, timeout=60
            )
            assert snapshot.returncode == 0
            physical = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
            assert any(
                (
                    line.strip().startswith("| " + physical + " ")
                    and str(os.getpid()) in line.split()
                    for line in snapshot.stdout.splitlines()
                )
            )
            (ROOT / ("physical-pid-" + str(time.time_ns()) + ".txt")).write_text(
                snapshot.stdout + snapshot.stderr
            )
        return len(tensors)

    class FixtureTrainer(Trainer):
        def __init__(self, *args, **kw):
            self.plan = []
            self.pending = []
            self.timing = None
            super().__init__(*args, **kw)

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            phase_sync("backward")
            return loss

        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(
                train_dataset if train_dataset is not None else self.train_dataset
            )

        def get_batch_samples(self, epoch_iterator, num_batches, device):
            result = super().get_batch_samples(epoch_iterator, num_batches, device)
            if self.timing:
                self.timing.outside("batch fetch/input planning")
            self.pending = []
            for batch in result[0]:
                ids = batch.pop("_sample_ids")
                assert isinstance(ids, tuple) and all((isinstance(x, int) for x in ids))
                shape = signature(batch)
                assert shape in SHAPES
                self.plan.append({"ids": list(ids), "shape": shape})
                self.pending.append(shape)
                assert all((value.device.type == args.device for value in batch.values()))
            return result

    class FunctionalTrainer(FixtureTrainer):
        def __init__(self, *args, **kw):
            self.gradients = []
            self.losses = []
            super().__init__(*args, **kw)

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            device_sync()
            assert_materialized_tensor(loss)
            self.losses.append(array(loss))
            assert np.isfinite(self.losses[-1]).all()
            gradients = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None, name
                    assert_materialized_tensor(parameter.grad)
                    assert parameter.grad.dtype == parameter.dtype
                    gradients[name] = array(parameter.grad)
                    assert np.isfinite(gradients[name]).all(), name
            self.gradients.append(gradients)
            return loss

    def device_sync():
        if JITTOR_MODE:
            jt.sync_all(device_sync=True)
        elif hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    class PhaseObserver(TrainerCallback):
        def __init__(self, mode):
            self.mode = mode

        def on_step_begin(self, *args, **kw):
            if self.mode == "functional":
                device_sync()

        def on_step_end(self, *args, **kw):
            if self.mode == "functional":
                device_sync()

        def on_pre_optimizer_step(self, *args, **kw):
            phase_sync("preopt")
            for parameter in kw["model"].parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None
                    assert_materialized_tensor(parameter.grad)
                    assert parameter.grad.dtype == parameter.dtype

        def on_optimizer_step(self, *args, **kw):
            phase_sync("postopt")

    class FunctionalObserver(TrainerCallback):
        def __init__(self):
            self.trainer = None
            self.clipped = []

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            snapshot = {
                name: array(value.grad)
                for (name, value) in self.trainer.model.named_parameters()
                if value.requires_grad
            }
            assert all((np.isfinite(value).all() for value in snapshot.values()))
            self.clipped.append(snapshot)

    class TimedObserver(TrainerCallback):
        def __init__(self):
            self.trainer = None
            self.timer = UpdateTimer(device_sync, warmup=4, measured=10, required_shapes=SHAPES)
            self.jit_before = None
            self.jit_after = None
            self.events = []

        def counter(self):
            if JITTOR_MODE:
                cache = Path(jt.compiler.cache_path) / "jit"
                return json.dumps(
                    {
                        str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(cache.rglob("*"))
                        if p.is_file() and p.suffix in (".so", ".cc", ".o")
                    },
                    sort_keys=True,
                )
            from torch._dynamo.utils import counters

            return json.dumps(
                {key: dict(value) for (key, value) in counters.items()}, sort_keys=True
            )

        def on_step_begin(self, args, state, control, **kwargs):
            assert int(state.global_step) + 1 <= 14
            self.timer.begin(int(state.global_step) + 1)
            for shape in self.trainer.pending:
                self.timer.microbatch(shape)
            self.events.append(["begin", int(state.global_step)])

        def on_step_end(self, args, state, control, **kwargs):
            self.timer.end(
                int(state.global_step),
                optimizer_skipped=bool(self.trainer.accelerator.optimizer_step_was_skipped),
            )
            self.events.append(["end", int(state.global_step)])
            if state.global_step == 4:
                self.jit_before = self.counter()
            if state.global_step == 14:
                self.jit_after = self.counter()
            step = int(state.global_step)
            connection.send(
                {
                    "event": "DONE",
                    "runtime": RUNTIME,
                    "step": step,
                    "shapes": list(self.trainer.pending),
                    "sample_ids": [row["ids"] for row in self.trainer.plan[-2:]],
                    "seconds": self.timer.samples[-1] if step > 4 else None,
                    "synchronized": True,
                }
            )
            grant = connection.recv()
            assert grant == (
                {"event": "FINALIZE"} if step == 14 else {"event": "GRANT", "step": step + 1}
            )

        def on_save(self, *args, **kw):
            self.timer.outside("save")
            raise AssertionError("save disabled")

        def on_evaluate(self, *args, **kw):
            self.timer.outside("eval")
            raise AssertionError("eval disabled")

        def on_log(self, *args, **kw):
            self.timer.outside("log")

    def train(mode):
        PHASE_COUNTS.clear()
        net = model()
        _load_initial(torch, net, initial)
        training_args = TrainingArguments(
            output_dir=str(ROOT / mode),
            use_cpu=args.device == "cpu",
            seed=SEED,
            data_seed=SEED,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            max_steps=UPDATES,
            learning_rate=0.003,
            weight_decay=0.01,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            logging_nan_inf_filter=False,
            disable_tqdm=True,
            report_to=[],
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        callback = FunctionalObserver() if mode == "functional" else TimedObserver()
        cls = FunctionalTrainer if mode == "functional" else FixtureTrainer
        trainer = cls(
            model=net,
            args=training_args,
            train_dataset=Dataset(),
            data_collator=collate,
            processing_class=tokenizer,
            callbacks=[PhaseObserver(mode), callback],
        )
        callback.trainer = trainer
        assert trainer.model is net and trainer.args.device.type == args.device
        assert next(trainer.model.parameters()).device.type == args.device
        placement_count = assert_model_inputs_actual(trainer.model)
        assert trainer.optimizer is None and trainer.lr_scheduler is None
        initial_state = {}
        flatten(trainer.model.state_dict(), "model", initial_state)
        flatten(torch.get_rng_state(), "rng/torch", initial_state)
        flatten(random.getstate(), "rng/python", initial_state)
        flatten(np.random.get_state(), "rng/numpy", initial_state)
        if args.device == "npu":
            flatten(torch.npu.get_rng_state_all(), "rng/npu", initial_state)
        np.savez(ROOT / (mode + "-initial-state.npz"), **initial_state)
        if mode == "plain":
            with np.load(ROOT / "functional-initial-state.npz") as reference:
                assert set(reference.files) == set(initial_state)
                for name, value in initial_state.items():
                    assert value.dtype == reference[name].dtype
                    np.testing.assert_array_equal(value, reference[name])
        if mode == "plain":
            connection.send(
                {
                    "event": "READY",
                    "determinism": DETERMINISM,
                    "runtime": RUNTIME,
                    "placement_tensor_count": placement_count,
                    "initial_sha256": INITIAL_SHA,
                    "phase_label": "synchronized training phases",
                    "attention": "eager",
                    "shape_inventory": SHAPES,
                    "shadow_complete": True,
                    "initial_reset_exact": True,
                }
            )
            grant = connection.recv()
            assert grant == {"event": "GRANT", "step": 1}
        for key in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[key] = "2"
        torch.set_num_threads(2)
        assert torch.get_num_threads() == 2
        blas_pretrain = loaded_blas_threads(2)
        guard = None
        fallback_before = jt.core.backend_fallback_count() if JITTOR_MODE else 0
        with ExitStack() as stack:
            if JITTOR_MODE:
                from jittor._runtime.fallback import forbid_backend_fallbacks

                stack.enter_context(jt.runtime.scope(backend_fallback="error"))
                stack.enter_context(forbid_backend_fallbacks())
            if mode == "plain":
                trainer.timing = callback.timer
                guard = ReadbackGuard(callback.timer, torch.Tensor)
                guard.install(stack)
            result = trainer.train()
        assert (jt.core.backend_fallback_count() - fallback_before if JITTOR_MODE else 0) == 0
        assert trainer.state.global_step == UPDATES and len(trainer.plan) == UPDATES * 2
        expected = (list(range(16)) * 4)[: UPDATES * 4]
        actual = [index for row in trainer.plan for index in row["ids"]]
        assert actual == expected
        state = full_state(trainer)
        state["sample_order"] = np.asarray(actual, dtype=np.int64)
        state["reported_train_loss"] = np.asarray(result.training_loss)
        np.savez(ROOT / (mode + "-state.npz"), **state)
        info = {
            "global_step": trainer.state.global_step,
            "microbatches": len(trainer.plan),
            "sample_order": actual,
            "shape_plan": [row["shape"] for row in trainer.plan],
            "state_array_count": len(state),
            "phase_sync_counts": dict(PHASE_COUNTS),
            "blas_before_train": blas_pretrain,
            "trainable_names": [
                n for (n, p) in trainer.model.named_parameters() if p.requires_grad
            ],
            "attention_implementation": trainer.model.config._attn_implementation,
            "logging_nan_inf_filter": trainer.args.logging_nan_inf_filter,
            "phase_label": "synchronized training phases",
            "outer_update_sync": True,
            "sdpa_cpu_control": SDPA_CPU_CONTROL,
        }
        if mode == "functional":
            assert len(trainer.gradients) == UPDATES * 2 and len(callback.clipped) == UPDATES
            gradients = {}
            for index, snapshot in enumerate(trainer.gradients):
                for name, value in snapshot.items():
                    gradients["micro/%02d/%s" % (index, name)] = value
            for index, snapshot in enumerate(callback.clipped):
                for name, value in snapshot.items():
                    gradients["postclip/%02d/%s" % (index, name)] = value
            np.savez(ROOT / "functional-gradients.npz", **gradients)
            np.save(ROOT / "functional-losses.npy", np.asarray(trainer.losses))
            info["gradient_array_count"] = len(gradients)
        else:
            diagnostic = (
                callback.timer.finish(callback.jit_before, callback.jit_after)
                if UPDATES == 14
                else {"admitted": False, "scope": "PREFIX_ENDPOINT_DIAGNOSTIC_ONLY"}
            )
            assert not guard.inside_accelerator
            info.update(
                cpu_diagnostic_timing=diagnostic,
                inside_cpu_scalar_or_array_calls=dict(guard.inside_cpu),
                inside_accelerator_readbacks=dict(guard.inside_accelerator),
                outside_readback_calls=dict(guard.outside_counts),
                callback_events=callback.events,
                jit_scope=args.device.upper()
                + " Jittor owned JIT artifact .so/.cc/.o hashes only; vendor compilation evidence still external"
                if JITTOR_MODE
                else args.device.upper()
                + " Torch eager/Dynamo counters only; vendor compilation evidence still external",
            )
        (ROOT / (mode + "-summary.json")).write_text(json.dumps(info, indent=2) + "\n")
        return info

    functional = train("functional")
    import gc

    gc.collect()
    connection.send(
        {
            "event": "SHADOW_DONE",
            "runtime": RUNTIME,
            "complete_gradient_arrays": functional["gradient_array_count"],
        }
    )
    plain = train("plain")
    assert (
        functional["sample_order"] == plain["sample_order"]
        and functional["shape_plan"] == plain["shape_plan"]
    )
    with np.load(ROOT / "functional-state.npz", allow_pickle=False) as left, np.load(
        ROOT / "plain-state.npz", allow_pickle=False
    ) as right:
        assert set(left.files) == set(right.files)
        errors = {
            name: float(
                np.max(np.abs(left[name].astype(np.float64) - right[name].astype(np.float64)))
            )
            for name in left.files
            if left[name].dtype.kind in "fiu" and left[name].size
        }
        differences = [
            {
                "name": name,
                "max_abs": errors.get(name),
                "dtype": str(left[name].dtype),
                "shape": list(left[name].shape),
                "different_elements": int(np.count_nonzero(left[name] != right[name])),
            }
            for name in left.files
            if left[name].dtype != right[name].dtype or not np.array_equal(left[name], right[name])
        ]
        comparison = {
            "strict_exact": not differences,
            "array_count": len(left.files),
            "first_difference": differences[0] if differences else None,
            "differences": differences,
            "maximum_numeric_error": max(errors.values()),
            "updates": UPDATES,
            "barriers": sorted(PHASES),
            "phase_sync_counts_functional": functional["phase_sync_counts"],
            "phase_sync_counts_plain": plain["phase_sync_counts"],
            "initial_sha256": INITIAL_SHA,
            "performance_claim": False,
            "scope": "TINY_EAGER_SYNCHRONIZED_TRAINING_PHASES",
            "source_manifest": args.source_manifest,
        }
        (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        print(json.dumps(comparison), flush=True)
        assert not differences, ("strict_exact_failure", comparison["first_difference"])
    result = {
        "status": "PASS",
        "scope": "TINY_EAGER_SYNCHRONIZED_TRAINING_PHASES",
        "performance_claim": False,
        "npu_executed": args.device == "npu",
        "npu_admitted": False,
        "torch_version": torch.__version__,
        "torch_source": str(TREE / "python/jittor/compat/torch/namespace.py")
        if JITTOR_MODE
        else torch.__file__,
        "npu_extension_imported": "torch_npu" in sys.modules,
        "determinism": DETERMINISM,
        "jittor_imported": JITTOR_MODE,
        "jittor_source": jt.__file__ if JITTOR_MODE else None,
        "updates": UPDATES,
        "phase_barriers": sorted(PHASES),
        "warmup_updates": min(4, UPDATES),
        "measured_updates": max(0, UPDATES - 4),
        "complete_microbatch_inventory_count": 8,
        "dynamic_lengths": LENGTHS,
        "functional_gradient_arrays": functional["gradient_array_count"],
        "compared_final_state_arrays": functional["state_array_count"],
        "full_model_optimizer_scheduler_rng_loss_sample_order_exact": True,
        "maximum_numeric_error": max(errors.values()),
        "shape_plan_exact": True,
        "cpu_threads": torch.get_num_threads(),
        "inside_accelerator_readbacks": plain["inside_accelerator_readbacks"],
        "fixture": "local initial NPZ; tiny random BERT+PEFT, no real checkpoint",
    }
    assert ("torch_npu" in sys.modules) == (args.device == "npu" and not JITTOR_MODE)
    (ROOT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    assert all(
        (
            os.environ.get(key) == "2"
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        )
    )
    BLAS = loaded_blas_threads()
    assert all((pool["after"] == 2 for row in BLAS for pool in row["pools"]))
    connection.send(
        {
            "event": "FINISHED",
            "runtime": RUNTIME,
            "status": "PASS",
            "full_state_exact": True,
            "npu_readback_gate": NPU_READBACK_GATE,
            "blas": BLAS,
        }
    )
    connection.close()


if __name__ == "__main__":
    arguments = parse_args()
    create_initial(arguments) if arguments.create_initial else run_worker(
        arguments
    ) if arguments.worker_runtime else run_controller(arguments)

"""Actual verl checkpoint save and fresh-process restore, fixed two-rank NPU worker.

Run baseline and resume in SEPARATE launches, using the same --resume-dir.
Only model and optimizer are saved/restored through TrainingWorker's real APIs.
SGD has momentum zero: this proves empty optimizer state plus parameter-group
restoration, NOT momentum slots, RNG, scheduler, rollout or full trainer resume.
All NPZ arrays and checkpoint files must remain on the execution server.
No model, tensor, optimizer mathematics or upstream method is replaced.
"""
import argparse
from contextlib import ExitStack
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import traceback
import uuid

import numpy as np

from fsdp2_checkpoint_fixture import CONTENTS, SCRIPT_NAMES, SCOPE, checkpoint_files, protocol
from fsdp2_worker_fixture import INPUTS, TARGETS, aliases, parameter_shapes
from probe_fsdp2_worker import run_worker
from probe_training_worker import Recorder, checkpoint_manifest, runtime
from worker_buffer_contract import buffer_shapes, expected_inv_freq


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observe_worker_output(record, key, value):
    # TrainingWorker.infer_batch explicitly returns _postprocess_output(...).cpu().
    # This host-return contract is not evidence of model/gradient placement.
    assert value.device.type == "cpu", "upstream worker return must be CPU"
    assert tuple(value.shape) == (2, 4)
    record.report.setdefault("worker_output_devices", {})[key] = "cpu"
    record.tensor(key, value)


def training_files(directory):
    files = {str(p.relative_to(directory)): digest(p) for p in directory.rglob("*") if p.is_file()}
    if not checkpoint_files() <= set(files) or any("extra_state" in name for name in files):
        raise ValueError("real model+optimizer shards required; extra RNG/scheduler state forbidden")
    if json.loads((directory / "fsdp_config.json").read_text()) != {"FSDP_version": 2, "world_size": 2}:
        raise ValueError("checkpoint is not from the fixed two-rank FSDP2 protocol")
    return files


class CheckpointRecorder(Recorder):
    """Observe our harness's return values; configure checkpointing before init."""
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.worker = self.model_config = None

    def stage(self, name, fn):
        result = super().stage(name, fn)
        if name == "import":
            from verl.trainer.config import CheckpointConfig
            self.args.checkpoint_config = CheckpointConfig(
                save_contents=CONTENTS[:], load_contents=CONTENTS[:], async_save=False,
                strict=True, save_lora_only=False)
        elif name == "init":
            self.worker, self.model_config = result
        return result


def run_checkpoint(args, record, torch, jt):
    # This intentionally stops at reset. It is not reported as a three-step run.
    run_worker(args, record, torch, jt)
    worker, model = record.worker, record.model_config
    rank = int(os.environ["RANK"])
    manager = worker.engine.checkpoint_manager
    record.report["checkpoint_manager"] = type(manager).__module__ + "." + type(manager).__name__
    record.report["observed_checkpoint_config"] = {
        "save_contents": list(manager.checkpoint_save_contents),
        "load_contents": list(manager.checkpoint_load_contents),
        "async_save": manager.checkpoint_config.async_save,
        "strict": manager.checkpoint_config.strict,
        "save_lora_only": manager.should_save_lora_only,
    }
    assert record.report["observed_checkpoint_config"] == dict(
        save_contents=CONTENTS, load_contents=CONTENTS, async_save=False, strict=True, save_lora_only=False)
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding
    from verl.workers.config import ActorConfig, CriticConfig
    from verl.workers.utils.losses import ppo_loss, value_loss
    from worker_gradient_observer import ObservedSGD

    def synchronize():
        jt.sync_all(True) if jt is not None else torch.npu.synchronize()

    def device_assert(value):
        if jt is None:
            assert value.device.type == "npu" and value.device.index == int(os.environ["LOCAL_RANK"])
        else:
            native = getattr(value, "_var", value)
            native.sync()
            assert native.location() == "device" and native.device_id == int(os.environ["LOCAL_RANK"])

    def inventory(stage):
        params = dict(worker.engine.module.named_parameters())
        all_paths = dict(worker.engine.module.named_parameters(remove_duplicate=False))
        assert set(params) == set(parameter_shapes(args.role))
        assert set(all_paths) == {path for paths in aliases(args.role).values() for path in paths}
        assert all(all_paths[path] is params[name] for name, paths in aliases(args.role).items() for path in paths)
        record.report.setdefault("alias_checks", []).append(stage)
        return params

    def snapshot(stage, gradients=False):
        params, placement = inventory(stage), {}
        with torch.no_grad():
            for name in sorted(params):
                value = params[name].grad if gradients else params[name]
                assert value is not None and callable(getattr(value, "full_tensor", None))
                local = value.to_local()
                count = int(local.numel())
                if count:
                    device_assert(local)
                placement[name] = dict(local_numel=count,
                                       residency="verified_npu" if count else "empty_not_materialized")
                full = value.full_tensor()
                device_assert(full)
                assert tuple(full.shape) == parameter_shapes(args.role)[name] and full.dtype == torch.float32
                assert params[name].requires_grad
                record.tensor(stage + "/" + name, full)
        record.report.setdefault("local_shard_observations", {})[stage] = placement

    def batch(anchor=None):
        ids = torch.tensor(INPUTS[rank], dtype=torch.long, device="cpu")
        data = dict(input_ids=ids, prompts=ids[:, :4], responses=ids[:, 4:],
                    attention_mask=torch.ones_like(ids),
                    position_ids=torch.arange(8, dtype=torch.long, device="cpu").unsqueeze(0).repeat(2, 1),
                    response_mask=torch.ones((2, 4), dtype=torch.float32, device="cpu"))
        if anchor is not None:
            data["old_log_probs" if args.role == "actor" else "values"] = anchor.detach().cpu().clone()
            data["advantages" if args.role == "actor" else "returns"] = torch.tensor(
                TARGETS[args.role][rank], dtype=torch.float32, device="cpu")
        proto = DataProto.from_single_dict(data, meta_info=dict(
            temperature=1.0, global_token_num=[8, 8], compute_loss=anchor is not None, calculate_entropy=False))
        packed = left_right_2_no_padding(proto.to_tensordict())
        if anchor is not None:
            tu.assign_non_tensor(packed, global_batch_size=4, update_lr_scheduler=False)
        return packed

    def infer(key):
        packed = batch()
        output = worker.infer_batch(packed)
        synchronize()
        value = no_padding_2_padding(tu.get(output, "log_probs" if args.role == "actor" else "values"), packed)
        observe_worker_output(record, key, value)
        return value.detach().cpu().clone()

    root = args.resume_dir
    checkpoint = root / "training"
    anchor_path = root / ("anchor.rank%d.npz" % rank)
    receipt_path = root / ("receipt.rank%d.json" % rank)
    expected_receipt = {key: record.report[key] for key in
                        ("framework", "role", "rank", "checkpoint_sha256", "script_sha256", "training_config")}

    def fixture():
        if args.phase == "baseline":
            if rank == 0:
                root.mkdir(parents=True, exist_ok=False)
            torch.distributed.barrier()
            anchor = infer("infer/output")
            with anchor_path.open("xb") as stream:
                np.savez(stream, anchor=record.arrays["infer/output"],
                         input_ids=np.asarray(INPUTS[rank], dtype=np.int64),
                         targets=np.asarray(TARGETS[args.role][rank], dtype=np.float32))
            receipt = dict(expected_receipt, baseline_process=record.report["process"],
                           anchor_sha256=digest(anchor_path))
            with receipt_path.open("x") as stream:
                json.dump(receipt, stream, indent=2)
        else:
            receipt = json.loads(receipt_path.read_text())
            assert all(receipt.get(key) == value for key, value in expected_receipt.items())
            assert receipt["anchor_sha256"] == digest(anchor_path)
            assert receipt["baseline_process"]["token"] != record.report["process"]["token"]
            with np.load(anchor_path, allow_pickle=False) as saved:
                assert set(saved.files) == {"anchor", "input_ids", "targets"}
                assert saved["anchor"].shape == (2, 4) and saved["anchor"].dtype == np.float32
                assert np.isfinite(saved["anchor"]).all()
                np.testing.assert_array_equal(saved["input_ids"], INPUTS[rank])
                np.testing.assert_array_equal(saved["targets"], np.asarray(TARGETS[args.role][rank], dtype=np.float32))
                anchor = torch.tensor(saved["anchor"].copy(), dtype=torch.float32, device="cpu")
            # Observe fresh initialization separately; do not replace the persisted anchor.
            infer("infer/output")
        record.report["anchor_receipt"] = receipt
        record.tensor("fixture/anchor", anchor)
        record.arrays["fixture/input_ids"] = np.asarray(INPUTS[rank], dtype=np.int64)
        record.arrays["fixture/targets"] = np.asarray(TARGETS[args.role][rank], dtype=np.float32)
        return anchor

    anchor = record.stage("fixture", fixture)
    if args.role == "actor":
        config = ActorConfig(strategy="fsdp2", model_config=model, rollout_n=1,
                             ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=2,
                             use_torch_compile=False, use_dynamic_bsz=False, ppo_epochs=1,
                             clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0,
                             loss_agg_mode="token-mean", entropy_coeff=0.0, use_kl_loss=False)
    else:
        config = CriticConfig(strategy="fsdp2", model=model, rollout_n=1,
                              ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=2,
                              cliprange_value=0.5, loss_agg_mode="token-mean")
    worker.set_loss_fn(partial(ppo_loss if args.role == "actor" else value_loss, config=config))
    assert isinstance(worker.engine.optimizer, ObservedSGD)

    def optimizer_observation():
        optimizer = worker.engine.optimizer
        params = dict(worker.engine.module.named_parameters())
        active = [p for group in optimizer.param_groups for p in group["params"]]
        assert len(active) == len(params) and {id(p) for p in active} == {id(p) for p in params.values()}
        state = optimizer.state_dict()
        assert state["state"] == {}, "protocol does not support nonempty optimizer slots"
        groups = []
        for group in state["param_groups"]:
            assert group["lr"] == 0.01 and group.get("momentum", 0) == group.get("weight_decay", 0) == 0
            groups.append(dict(lr=group["lr"], momentum=group.get("momentum", 0),
                               weight_decay=group.get("weight_decay", 0), parameter_count=len(group["params"])))
        assert sum(g["parameter_count"] for g in groups) == len(params)
        return dict(state_empty=True, groups=groups, parameter_identity_verified=True)

    current, observed = [None], []
    def capture(optimizer):
        inventory(current[0] + "/observer")
        optimizer_observation()
        snapshot(current[0] + "/param_grad", gradients=True)
        observed.append(True)

    worker.engine.optimizer.set_gradient_observer(capture)
    record.report["training_steps"] = []

    def update(step):
        prefix = "step_%03d" % step
        current[0] = prefix
        observed.clear()
        snapshot(prefix + "/pre_weight")
        output = worker.train_batch(batch(anchor))
        synchronize()
        assert len(observed) == 1
        snapshot(prefix + "/post_weight")
        metrics = tu.get(output, "metrics")
        loss, norm = float(metrics["loss"]), float(metrics["grad_norm"])
        assert np.isfinite([loss, norm]).all() and norm > 0
        record.arrays[prefix + "/loss"] = np.asarray(loss)
        record.arrays[prefix + "/grad_norm"] = np.asarray(norm)
        infer(prefix + "/output")
        record.report["training_steps"].append(dict(step=step, loss=loss, grad_norm=norm,
                                                   captured_gradient_count=len(parameter_shapes(args.role))))

    def save():
        record.report["checkpoint_optimizer"] = optimizer_observation()
        worker.save_checkpoint(str(checkpoint), global_step=2, max_ckpt_to_keep=None)
        synchronize()
        torch.distributed.barrier()
        record.report["training_checkpoint_sha256"] = training_files(checkpoint)

    def load():
        before = {name: id(p) for name, p in worker.engine.module.named_parameters()}
        record.report["training_checkpoint_sha256"] = training_files(checkpoint)
        # Public optimizer configuration sentinel, with NO train/step before load.
        for group in worker.engine.optimizer.param_groups:
            group["lr"] = 0.02
        record.report["resume_lr_sentinel"] = {"before_load": [group["lr"] for group in worker.engine.optimizer.param_groups],
                                               "updates_before_load": 0}
        worker.load_checkpoint(str(checkpoint), del_local_after_load=False)
        synchronize()
        assert before == {name: id(p) for name, p in worker.engine.module.named_parameters()}
        record.report["checkpoint_optimizer"] = optimizer_observation()
        record.report["resume_lr_sentinel"]["after_load"] = [group["lr"] for group in worker.engine.optimizer.param_groups]
        assert record.report["training_checkpoint_sha256"] == training_files(checkpoint)

    def checkpoint_point():
        snapshot("checkpoint_point/weight")
        infer("checkpoint_point/output")
        buffers = dict(worker.engine.module.named_buffers())
        assert set(buffers) == set(buffer_shapes())
        for name, value in buffers.items():
            device_assert(value)
            actual = record.tensor("checkpoint_point/buffer/" + name, value)
            np.testing.assert_array_equal(actual, record.arrays["initial_buffer/" + name])
            np.testing.assert_allclose(actual, expected_inv_freq(worker.engine.module.config), atol=1e-10, rtol=1e-7)
        record.report["checkpoint_buffer_residency"] = "all verified NPU; reconstructed nonpersistent buffers"

    if args.phase == "baseline":
        for step in (0, 1):
            record.stage("step_%03d" % step, lambda step=step: update(step))
        record.stage("save_checkpoint", save)
    else:
        record.stage("load_checkpoint", load)
    record.stage("checkpoint_point", checkpoint_point)
    record.stage("step_002", lambda: update(2))
    assert training_files(checkpoint) == record.report["training_checkpoint_sha256"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--role", choices=("actor", "critic"), required=True)
    parser.add_argument("--phase", choices=("baseline", "resume"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.output.suffix != ".json" or args.timeout < 1 or not args.checkpoint.is_dir():
        parser.error("existing HF checkpoint, .json output and positive timeout required")
    if os.environ.get("WORLD_SIZE") != "2" or os.environ.get("RANK") not in ("0", "1"):
        parser.error("requires real two-rank launcher environment")
    if args.output.with_name(args.output.stem + ".rank" + os.environ["RANK"] + ".json").exists():
        parser.error("refusing to overwrite an existing phase artifact")
    args.output = args.output.with_name(args.output.stem + ".rank" + os.environ["RANK"] + ".json")
    args.backend, args.stop_after, args.steps = "hccl", "reset", 3
    args.capture_gradients, args.prepare_checkpoint = True, False
    return args


def main():
    args = parse_args()
    record = CheckpointRecorder(args)
    record.report.update(schema=1, artifact_type="verl_fsdp2_checkpoint", role=args.role, phase=args.phase,
                         scope=SCOPE, requested_stage="checkpoint_next_step", training_config=protocol(args.role),
                         process={"pid": os.getpid(), "token": uuid.uuid4().hex},
                         script_sha256={name: digest(Path(__file__).with_name(name)) for name in SCRIPT_NAMES},
                         limitations=["model and empty-state SGD only; no RNG/scheduler/momentum state resume",
                                      "same two-rank runtime only; no cross-framework checkpoint import",
                                      "nonpersistent buffers reconstructed and checked, not loaded from checkpoint",
                                      "fixed targets, no rollout/Ray/full trainer or performance acceptance",
                                      "NVTX marker binding only; oracle CPU fallbacks remain uninstrumented"])
    try:
        with ExitStack() as stack:
            torch, jt = record.stage("runtime", lambda: runtime(args, stack, record))
            record.report["checkpoint_sha256"] = checkpoint_manifest(args.checkpoint)
            run_checkpoint(args, record, torch, jt)
            if jt is not None:
                jt.sync_all(True)
                record.report["fallback_count"] = int(jt.core.backend_fallback_count()) - record.report["fallback_before"]
                assert record.report["fallback_count"] == 0
            record.report["status"] = "checkpoint_next_step_completed"
            record.flush()
    except Exception as error:
        record.report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        record.flush()
        raise


if __name__ == "__main__":
    main()

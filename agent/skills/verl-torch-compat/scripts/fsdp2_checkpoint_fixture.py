"""Fixed model+optimizer-only resume contract; no RNG/scheduler or momentum claim."""
from fsdp2_worker_fixture import SCRIPT_NAMES as WORKER_SCRIPTS, protocol as worker_protocol

SCOPE = "two-rank actual TrainingWorker model+empty-SGD-state checkpoint and fresh-process next update"
SCRIPT_NAMES = set(WORKER_SCRIPTS) | {
    "fsdp2_checkpoint_fixture.py", "probe_fsdp2_checkpoint.py", "compare_fsdp2_checkpoint.py",
    "run_fsdp2_checkpoint.sh",
    "compare_training_worker.py",
}
CONTENTS = ["model", "optimizer"]


def protocol(role):
    return dict(worker=worker_protocol(role), save_after_updates=2, resume_update=2,
                checkpoint_save_contents=CONTENTS[:], checkpoint_load_contents=CONTENTS[:],
                async_save=False, strict=True, optimizer_momentum=0.0,
                optimizer_state="empty SGD state, parameter groups are restored",
                resume_lr_sentinel=0.02, restored_lr=0.01, updates_before_load=0,
                anchor="original pre-update inference persisted per rank; never recomputed on resume",
                update_lr_scheduler=False, rng_restore=False, scheduler_restore=False,
                fixed_world_size=2, cross_framework_checkpoint_loading=False)


def steps(phase):
    if phase not in ("baseline", "resume"):
        raise ValueError("invalid checkpoint phase")
    return [0, 1, 2] if phase == "baseline" else [2]


def stages(phase):
    return ["runtime", "import", "distributed_init", "init", "reset", "fixture"] + (
        ["step_000", "step_001", "save_checkpoint", "checkpoint_point", "step_002"]
        if phase == "baseline" else ["load_checkpoint", "checkpoint_point", "step_002"])


def snapshots(phase):
    return ["initial", "checkpoint_point/weight"] + [
        "step_%03d/%s" % (step, kind) for step in steps(phase)
        for kind in ("pre_weight", "param_grad", "post_weight")]


def alias_checks(phase):
    checks = ["initial", "reset_verified"]
    for step in steps(phase):
        if step == 2:
            checks.append("checkpoint_point/weight")
        checks.extend("step_%03d/%s" % (step, kind) for kind in
                      ("pre_weight", "observer", "param_grad", "post_weight"))
    return checks


def checkpoint_files():
    return {"%s_world_size_2_rank_%d.pt" % (kind, rank)
            for kind in ("model", "optim") for rank in (0, 1)} | {"fsdp_config.json", "huggingface/config.json"}

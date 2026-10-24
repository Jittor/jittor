"""Real CUDA BF16 AdamW public-state offload/reload regression.

The run helper also supports authenticated CUDA wrappers. CPU tensors are only
moment storage copies; parameter/gradient/moment arithmetic executes on CUDA.
Scalar step bookkeeping retains the original Adam contract. This test proves
neither full PPO completion nor allocator/peak-memory savings.
"""

import weakref
import unittest
import pytest

pytestmark = pytest.mark.cuda


MOMENTS = ("exp_avg", "exp_avg_sq")
CASES = ("replace_moment_fields", "replace_whole_record")


def require(condition, label):
    if not condition:
        raise AssertionError(label)


def _execution_slot(optimizer, parameter):
    """Read-only test observation, NOT an adapter/public migration operation."""
    from jittor.optim.base import optimizer_execution_groups
    matches = [(group, index) for group in optimizer_execution_groups(optimizer)
               for index, candidate in enumerate(group["params"])
               if candidate is parameter]
    require(len(matches) == 1, "exact_parameter_execution_slot")
    return matches[0]


def _slot_identity(group, index, record):
    return (group["m"][index] is record["exp_avg"]
            and group["values"][index] is record["exp_avg_sq"])


def _old_slot_count(group, index, references):
    return sum(reference() is not None and group[key][index] is reference()
               for key, reference in zip(("m", "values"), references))


def _resident(jt, pointer_probe, tensor, device):
    # Materialize actual state storage, including CPU offload copies. This is
    # storage transfer/verification, not a CPU reference/model computation.
    tensor.sync()
    from jittor._core.dtypes import var_dtype_name
    require(var_dtype_name(tensor) == "bfloat16", "actual_BF16_storage_dtype")
    if device == "cuda":
        pointer_probe(jt, tensor)
        require((int(tensor.placement_backend), int(tensor.device_id)) == (1, 0)
                and str(tensor.device) == "cuda:0"
                and tensor.location() == "device", "actual_CUDA_BF16_storage")
    else:
        require(int(tensor.placement_backend) == 0
                and str(tensor.device) == "cpu" and tensor.location() == "cpu",
                "actual_CPU_offload_storage")


def _move_public_moments(optimizer, parameter, device, whole_record):
    # No local reference to the old inner dict survives this helper. Real dict
    # writes intentionally bypass compat's private execution-slot machinery.
    if whole_record:
        optimizer.state[parameter] = {
            key: value.to(device=device) if key in MOMENTS else value
            for key, value in optimizer.state[parameter].items()}
    else:
        for key in MOMENTS:
            optimizer.state[parameter][key] = optimizer.state[parameter][key].to(device=device)


def _counts_projection(counts):
    return [dict(label=key[0], exception=key[1], count=count)
            for key, count in sorted(counts.items())]


def run(torch, jt, pointer_probe, *, mode="baseline", reconcile=None, **wrapper_arguments):
    """Baseline: no bridge; future public_bridge mode requires explicit callable.

    Caller must pin this exact file and the future bridge's authentic source.
    Passing mode=public_bridge never fabricates admission or a successful result.
    Expected stale-reference observations are data, not intercepted exceptions;
    all unexpected framework errors propagate to caller's existing safe reporter.
    """
    from collections import defaultdict
    from jittor.compat.diagnostics import counts
    from jittor._runtime.fallback import forbid_backend_fallbacks
    require(mode in ("baseline", "public_bridge"), "explicit_probe_mode")
    require((mode == "baseline" and reconcile is None)
            or (mode == "public_bridge" and callable(reconcile)), "no_implicit_private_bridge")
    before_counts = counts()
    before_grad_mode = torch.is_grad_enabled()
    rows = []
    with forbid_backend_fallbacks(), jt.runtime.scope(
            use_cuda=1, device_id=0, backend_fallback="error"), torch.enable_grad(), torch.device("cuda:0"):
        for case in CASES:
            whole = case == "replace_whole_record"
            initial = torch.tensor([[.5, -.75], [1.25, -.25]], dtype=torch.bfloat16, device="cuda:0")
            p = torch.nn.Parameter(initial.detach().clone())
            reference_p = torch.nn.Parameter(initial.detach().clone())
            foreign_p = torch.nn.Parameter(initial.detach().clone())
            options = dict(lr=.03125, betas=(.5, .75), eps=.125,
                           weight_decay=.125, foreach=False, fused=False)
            optimizer = torch.optim.AdamW([p], **options)
            reference_optimizer = torch.optim.AdamW([reference_p], **options)
            # A second independent owner deliberately becomes the latest one.
            foreign_optimizer = torch.optim.AdamW([foreign_p], **options)
            gradient = torch.tensor([[.25, -.5], [.75, -.125]],
                                    dtype=torch.bfloat16, device="cuda:0")
            for parameter, current_optimizer in ((p, optimizer),
                    (reference_p, reference_optimizer), (foreign_p, foreign_optimizer)):
                current_optimizer.zero_grad(set_to_none=True)
                (parameter * gradient).sum().backward()
                _resident(jt, pointer_probe, parameter.grad, "cuda")
                require(bool(torch.equal(parameter.grad, gradient)), "real_CUDA_backward_gradient")
                current_optimizer.step()
                _resident(jt, pointer_probe, parameter, "cuda")
                for key in MOMENTS:
                    _resident(jt, pointer_probe, current_optimizer.state[parameter][key], "cuda")
            group, index = _execution_slot(optimizer, p)
            foreign_group, foreign_index = _execution_slot(foreign_optimizer, foreign_p)
            require(_slot_identity(group, index, optimizer.state[p]), "initial_canonical_moments")
            old = tuple(weakref.ref(optimizer.state[p][key]) for key in MOMENTS)
            old_record_identity = id(optimizer.state[p])
            public_state = optimizer.state
            public_groups = optimizer.param_groups
            foreign_record = foreign_optimizer.state[foreign_p]
            foreign_moments = tuple(foreign_record[key] for key in MOMENTS)
            foreign_values = tuple(value.detach().clone() for value in foreign_moments)
            foreign_parameter = foreign_p.detach().clone()
            held_grad = p.grad
            held_grad_copy = held_grad.detach().clone()
            _move_public_moments(optimizer, p, "cpu", whole)
            for key in MOMENTS:
                _resident(jt, pointer_probe, optimizer.state[p][key], "cpu")
            stale_before = _old_slot_count(group, index, old)
            if reconcile is not None:
                reconcile(optimizer)  # Future AUTHENTIC PUBLIC entry only.
            # Observe before reload/step can naturally repair private buffers.
            row = dict(case=case, mode=mode,
                original_private_slots_retaining_old_CUDA_moments=stale_before,
                private_slots_retaining_old_CUDA_moments_after_bridge=_old_slot_count(group, index, old),
                public_and_execution_moments_same_after_offload=_slot_identity(group, index, optimizer.state[p]),
                old_moment_objects_alive_after_offload=sum(ref() is not None for ref in old),
                weakref_death_is_not_allocator_or_OOM_proof=True,
                public_container_types_preserved=(type(optimizer.state) is defaultdict
                    and optimizer.state.default_factory is dict and type(optimizer.state[p]) is dict),
                public_state_and_groups_identity_preserved=(optimizer.state is public_state
                    and optimizer.param_groups is public_groups),
                inner_record_identity_matches_requested_operation=(
                    (id(optimizer.state[p]) != old_record_identity) is whole),
                no_GC_empty_cache_or_private_state_write=True, cpu_model_computation=False,
                passed=False)
            rows.append(row)
            row["foreign_owner_unchanged_during_migration"] = (
                foreign_optimizer.state[foreign_p] is foreign_record
                and _slot_identity(foreign_group, foreign_index, foreign_record)
                and all(foreign_record[key] is tensor and bool(torch.equal(tensor, saved))
                        for key, tensor, saved in zip(MOMENTS, foreign_moments, foreign_values))
                and bool(torch.equal(foreign_p, foreign_parameter)))
            row["held_gradient_preserved_during_migration"] = (
                p.grad is held_grad and bool(torch.equal(held_grad, held_grad_copy)))
            _resident(jt, pointer_probe, p, "cuda")
            _resident(jt, pointer_probe, held_grad, "cuda")
            _move_public_moments(optimizer, p, "cuda:0", whole)
            for key in MOMENTS:
                _resident(jt, pointer_probe, optimizer.state[p][key], "cuda")
            if reconcile is not None:
                reconcile(optimizer)
            row["public_and_execution_moments_same_after_reload"] = _slot_identity(group, index, optimizer.state[p])
            # Continue even when the offload ownership assertion is false, but
            # AND that original failure into the final payload. No fake green.
            for parameter, current_optimizer in ((p, optimizer), (reference_p, reference_optimizer)):
                current_optimizer.zero_grad(set_to_none=True)
                (parameter * (gradient * .5)).sum().backward()
                _resident(jt, pointer_probe, parameter.grad, "cuda")
                current_optimizer.step()
                _resident(jt, pointer_probe, parameter, "cuda")
            row["reload_next_step_parameter_exact_CUDA"] = bool(torch.equal(p, reference_p))
            row["reload_next_step_moments_exact_CUDA"] = True
            for key in MOMENTS:
                _resident(jt, pointer_probe, optimizer.state[p][key], "cuda")
                _resident(jt, pointer_probe, reference_optimizer.state[reference_p][key], "cuda")
                row["reload_next_step_moments_exact_CUDA"] &= bool(torch.equal(
                    optimizer.state[p][key], reference_optimizer.state[reference_p][key]))
            # Scalar step is the unchanged original Adam bookkeeping contract;
            # no host parameter/moment arrays or NumPy oracle are constructed.
            row["reload_step_counter_exact_two"] = (
                optimizer.state[p]["step"].item() == 2
                and reference_optimizer.state[reference_p]["step"].item() == 2)
            row["passed"] = (row["private_slots_retaining_old_CUDA_moments_after_bridge"] == 0
                and row["public_and_execution_moments_same_after_offload"]
                and row["old_moment_objects_alive_after_offload"] == 0
                and row["public_and_execution_moments_same_after_reload"]
                and all(row[key] for key in ("public_container_types_preserved",
                    "public_state_and_groups_identity_preserved", "inner_record_identity_matches_requested_operation",
                    "foreign_owner_unchanged_during_migration", "held_gradient_preserved_during_migration",
                    "reload_next_step_parameter_exact_CUDA", "reload_next_step_moments_exact_CUDA",
                    "reload_step_counter_exact_two")))
    after_counts = counts()
    diagnostic_same = after_counts == before_counts
    grad_mode_same = torch.is_grad_enabled() == before_grad_mode
    fallback = int(jt.core.backend_fallback_count())
    return dict(schema=1, scope="BF16_CUDA_AdamW_public_state_offload_reload_NOT_PPO",
        mode=mode, cases=rows, expected_cases_per_rank=2, observations_complete=len(rows) == 2,
        public_bridge_called=mode == "public_bridge", source_and_runtime_admission_required_from_caller=True,
        counts_before=_counts_projection(before_counts), counts_after=_counts_projection(after_counts),
        diagnostic_counts_unchanged=diagnostic_same, new_swallowed_count=sum(after_counts.values())-sum(before_counts.values()),
        fallback_count=fallback, grad_mode_restored=grad_mode_same,
        payload_passed=len(rows) == 2 and all(row["passed"] for row in rows)
                       and diagnostic_same and fallback == 0 and grad_mode_same,
        no_full_PPO_or_L_acceptance_claim=True)


class TestOptimizerStateMigrationCUDA(unittest.TestCase):
    """Real BF16 CUDA state migration, isolated-owner and reload trajectory."""

    def setUp(self):
        import torch
        if not torch.cuda.is_available():
            self.skipTest("real CUDA is unavailable; not a successful device validation")
        import ctypes
        import jittor as jt
        from jittor.compat.torch import namespace_owner
        self.assertIs(namespace_owner(torch), jt)
        self.assertEqual(torch.cuda.current_device(), 0)
        self.torch, self.jt = torch, jt
        self.driver = ctypes.CDLL("libcuda.so.1")
        self.driver.cuInit.argtypes = [ctypes.c_uint]
        self.driver.cuInit.restype = ctypes.c_int
        self.driver.cuPointerGetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.driver.cuPointerGetAttribute.restype = ctypes.c_int
        self.assertEqual(self.driver.cuInit(0), 0)

    def pointer_probe(self, jt, value):
        import ctypes
        self.assertIs(jt, self.jt)
        value.sync()
        self.assertGreater(value.numel(), 0)
        self.assertEqual(tuple(jt.core.dispatch_context([value])), ("cuda", 0))
        self.assertEqual(value.location(), "device")
        pointer = int(value.device_raw_ptr)
        self.assertGreater(pointer, 0)
        ordinal, memory = ctypes.c_int(-1), ctypes.c_int(-1)
        self.assertEqual(self.driver.cuPointerGetAttribute(
            ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)), 0)
        self.assertEqual(self.driver.cuPointerGetAttribute(
            ctypes.byref(memory), 2, ctypes.c_void_p(pointer)), 0)
        self.assertEqual((ordinal.value, memory.value), (0, 2))
        return dict(device_ordinal=ordinal.value, memory_type=memory.value,
                    nonempty_storage=True, actual_driver_pointer_validated=True)

    def test_public_reconcile_offload_reload_preserves_bf16_adamw(self):
        from jittor.compat.torch import reconcile_optimizer_state
        self.runtime_evidence = run(self.torch, self.jt, self.pointer_probe,
            mode="public_bridge", reconcile=reconcile_optimizer_state)
        self.assertEqual(len(self.runtime_evidence["cases"]), 2)
        self.assertTrue(self.runtime_evidence["observations_complete"])
        self.assertTrue(all(row["passed"] for row in self.runtime_evidence["cases"]))
        self.assertTrue(self.runtime_evidence["payload_passed"])


if __name__ == "__main__":
    unittest.main()

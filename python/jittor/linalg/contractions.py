# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Einstein summation planning and tensor contractions."""


def einsum(equation, *operands):
    r"""Evaluate the Einstein summation convention on the operands.

    Native jittor implementation: GEMM-shaped contractions dispatch to
    ``jt.matmul`` (cublas on GPU). Other cases are expressed with the
    existing jittor tensor ops used by this implementation, including
    ``reindex`` for diagonal handling, element-wise operations, reductions
    such as ``sum``, and reshapes/transposes as needed. No numpy round-trip,
    so ``float16`` and ``bfloat16`` operands stay on-device throughout and
    autograd is provided by the underlying jittor ops.
    """
    import numpy as np_cpu
    if len(operands) == 0:
        raise ValueError("einsum requires at least one operand")
    # torch.einsum also accepts the operands packed in a single list/tuple, e.g.
    # torch.einsum("bcxd,bcyd->bcxy", (query, key)) (used by longformer's windowed
    # attention). Unpack that form so the operands are individual Vars below.
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        operands = tuple(operands[0])
    # ``_parse_einsum_input`` calls ``asanyarray`` on the operands, so feed
    # it shape-compatible numpy stand-ins and keep the original jittor Vars.
    fake_ops = [np_cpu.empty([1] * len(o.shape), dtype=np_cpu.float32)
                for o in operands]
    in_subs_str, out_sub, _ = np_cpu.core.einsumfunc._parse_einsum_input(
        [equation, *fake_ops])
    in_subs = in_subs_str.split(",")
    operands = list(operands)

    new_subs, new_ops = [], []
    for i, (s, op) in enumerate(zip(in_subs, operands)):
        s, op = _einsum_unary_normalize(s, op, in_subs, out_sub, i)
        new_subs.append(s)
        new_ops.append(op)
    in_subs, operands = new_subs, new_ops

    while len(operands) > 1:
        i, j = _einsum_pick_pair(in_subs, out_sub)
        # Pop the higher index first so the lower-index pop stays valid.
        b = operands.pop(j); sb = in_subs.pop(j)
        a = operands.pop(i); sa = in_subs.pop(i)
        so = _einsum_intermediate_subs(sa, sb, in_subs, out_sub)
        operands.insert(0, _einsum_pair_contract(sa, sb, so, a, b))
        in_subs.insert(0, so)

    return _einsum_finalize(in_subs[0], out_sub, operands[0])


def _einsum_pick_pair(in_subs, out_sub):
    # Prefer a pair that shares at least one label, to avoid unnecessary
    # outer-products. Among shared-label pairs, prefer ones whose contraction
    # result is non-empty when there are still operands queued — collapsing
    # to a scalar mid-stream forces the leftover operands through a degenerate
    # ``scalar * tensor`` path.
    n = len(in_subs)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            sa, sb = in_subs[i], in_subs[j]
            shared = set(sa) & set(sb)
            if not shared:
                continue
            remaining = [in_subs[k] for k in range(n) if k != i and k != j]
            keep = set(out_sub)
            for s in remaining:
                keep |= set(s)
            result_chars = (set(sa) | set(sb)) & keep
            score = (1 if result_chars else 0, len(shared))
            if best is None or score > best[0]:
                best = (score, i, j)
    if best is not None:
        return best[1], best[2]
    return 0, 1


def _einsum_dedup(s):
    seen, out = set(), []
    for c in s:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _einsum_permute_to(s_from, s_to, op):
    if s_from == s_to:
        return op
    perm = [s_from.index(c) for c in s_to]
    if perm == list(range(len(perm))):
        return op
    return op.permute(perm)


def _einsum_unary_normalize(sub, op, all_subs, out_sub, my_index):
    # Diagonal extraction: collapse repeated indices via a reindex.
    if len(set(sub)) != len(sub):
        unique = _einsum_dedup(sub)
        out_shape = []
        for c in unique:
            sizes = [op.shape[k] for k, ch in enumerate(sub) if ch == c]
            for sz in sizes[1:]:
                if sz != sizes[0]:
                    raise ValueError(
                        f"einsum: repeated index '{c}' has mismatched dim sizes {sizes}")
            out_shape.append(sizes[0])
        formulas = [f"i{unique.index(c)}" for c in sub]
        op = op.reindex(out_shape, formulas)
        sub = "".join(unique)

    # Sum out indices that don't appear in any other operand or in the output.
    other_chars = set(out_sub)
    for k, s in enumerate(all_subs):
        if k != my_index:
            other_chars |= set(s)
    sum_dims = [i for i, c in enumerate(sub) if c not in other_chars]
    if sum_dims:
        op = op.sum(sum_dims)
        drop = set(sum_dims)
        sub = "".join(c for i, c in enumerate(sub) if i not in drop)
    return sub, op


def _einsum_intermediate_subs(sa, sb, remaining_subs, out_sub):
    keep = set(out_sub)
    for s in remaining_subs:
        keep |= set(s)
    return "".join(c for c in _einsum_dedup(sa + sb) if c in keep)


def _einsum_resolve_size(label, size_a, size_b):
    # numpy-style broadcasting: 1 ↔ N → N; equal → that size; mismatch → error.
    if size_a is None: return size_b
    if size_b is None: return size_a
    if size_a == size_b: return size_a
    if size_a == 1: return size_b
    if size_b == 1: return size_a
    raise ValueError(
        f"einsum: shape mismatch for index '{label}': {size_a} vs {size_b}")


def _einsum_broadcast_axes(op, target_shape):
    # Expand size-1 axes of ``op`` to match ``target_shape``. Lengths must match.
    cur_shape = list(op.shape)
    if len(cur_shape) != len(target_shape):
        raise ValueError(
            f"einsum: cannot broadcast shape {tuple(cur_shape)} to "
            f"{tuple(target_shape)}: rank mismatch")
    needs = False
    for axis, (cs, ts) in enumerate(zip(cur_shape, target_shape)):
        if cs != ts:
            if cs != 1:
                raise ValueError(
                    f"einsum: cannot broadcast shape {tuple(cur_shape)} to "
                    f"{tuple(target_shape)}: axis {axis} has size {cs}, "
                    f"expected {ts} or 1")
            needs = True
    if not needs:
        return op
    return op.broadcast(target_shape)


def _einsum_pair_contract(sa, sb, so, a, b):
    # If either operand is the result of an earlier full contraction (sub == ""),
    # it has no labelled axes; fall back to a plain element-wise multiply, which
    # will broadcast a 0-D / shape-(1,) scalar against the remaining operand.
    if not sa and not sb:
        out = a * b
        return out
    if not sa:
        return _einsum_finalize_scalar_pair(a, sb, so, b)
    if not sb:
        return _einsum_finalize_scalar_pair(b, sa, so, a)

    a_set, b_set, o_set = set(sa), set(sb), set(so)

    drop_a = [i for i, c in enumerate(sa) if c not in b_set and c not in o_set]
    if drop_a:
        a = a.sum(drop_a)
        ds = set(drop_a)
        sa = "".join(c for i, c in enumerate(sa) if i not in ds)
        a_set = set(sa)
    drop_b = [i for i, c in enumerate(sb) if c not in a_set and c not in o_set]
    if drop_b:
        b = b.sum(drop_b)
        ds = set(drop_b)
        sb = "".join(c for i, c in enumerate(sb) if i not in ds)
        b_set = set(sb)

    shared = a_set & b_set
    batch = [c for c in so if c in shared]
    contract = [c for c in sa if c in shared and c not in o_set]
    free_a = [c for c in sa if c not in shared]
    free_b = [c for c in sb if c not in shared]

    # Resolve broadcasted size per label for shared (batch + contract) axes.
    sizes_a = {c: a.shape[i] for i, c in enumerate(sa)}
    sizes_b = {c: b.shape[i] for i, c in enumerate(sb)}
    resolved = {}
    for c in shared:
        resolved[c] = _einsum_resolve_size(c, sizes_a[c], sizes_b[c])
    for c in free_a:
        resolved[c] = sizes_a[c]
    for c in free_b:
        resolved[c] = sizes_b[c]

    if not contract:
        return _einsum_outer(sa, sb, so, a, b, batch, free_a, free_b, resolved)

    return _einsum_matmul(sa, sb, so, a, b, batch, free_a, free_b, contract, resolved)


def _einsum_finalize_scalar_pair(scalar, sb, so, b):
    # ``scalar`` is the result of a prior full contraction (sub == "").
    # In jittor that's a shape-(1,) Var; multiply broadcasts it against ``b``.
    if scalar.ndim > 1:
        scalar = scalar.reshape([1])
    out = b * scalar
    drop = [i for i, c in enumerate(sb) if c not in so]
    if drop:
        out = out.sum(drop)
        ds = set(drop)
        sb = "".join(c for i, c in enumerate(sb) if i not in ds)
    return _einsum_permute_to(sb, so, out)


def _einsum_outer(sa, sb, so, a, b, batch, free_a, free_b, resolved):
    a_target = "".join(batch + free_a)
    a_perm = _einsum_permute_to(sa, a_target, a)
    # Broadcast batch axes of A to resolved sizes (free_a comes from A only).
    a_target_shape = [resolved[c] for c in batch] + [a_perm.shape[len(batch) + i]
                                                     for i in range(len(free_a))]
    a_perm = _einsum_broadcast_axes(a_perm, a_target_shape)
    for _ in free_b:
        a_perm = a_perm.unsqueeze(-1)

    b_target = "".join(batch + free_b)
    b_perm = _einsum_permute_to(sb, b_target, b)
    b_target_shape = [resolved[c] for c in batch] + [b_perm.shape[len(batch) + i]
                                                     for i in range(len(free_b))]
    b_perm = _einsum_broadcast_axes(b_perm, b_target_shape)
    insert_at = len(batch)
    for _ in free_a:
        b_perm = b_perm.unsqueeze(insert_at)

    out = a_perm * b_perm
    intermediate = "".join(batch + free_a + free_b)
    return _einsum_permute_to(intermediate, so, out)


def _einsum_matmul(sa, sb, so, a, b, batch, free_a, free_b, contract, resolved):
    import jittor as jt
    a_target = "".join(batch + free_a + contract)
    b_target = "".join(batch + contract + free_b)
    a_p = _einsum_permute_to(sa, a_target, a)
    b_p = _einsum_permute_to(sb, b_target, b)

    nb = len(batch)
    nfa = len(free_a)
    nc = len(contract)

    # Broadcast each operand's batch and contract axes to the resolved sizes;
    # free axes come from a single operand and stay as-is.
    batch_shape = [resolved[c] for c in batch]
    fa_shape = [a_p.shape[nb + i] for i in range(nfa)]
    fb_shape = [b_p.shape[nb + nc + i] for i in range(len(free_b))]
    contract_shape = [resolved[c] for c in contract]

    a_p = _einsum_broadcast_axes(a_p, batch_shape + fa_shape + contract_shape)
    b_p = _einsum_broadcast_axes(b_p, batch_shape + contract_shape + fb_shape)

    fa_size = 1
    for s in fa_shape: fa_size *= s
    c_size = 1
    for s in contract_shape: c_size *= s
    fb_size = 1
    for s in fb_shape: fb_size *= s

    a_flat = a_p.reshape(batch_shape + [fa_size, c_size])
    b_flat = b_p.reshape(batch_shape + [c_size, fb_size])

    out_flat = jt.matmul(a_flat, b_flat)

    target_shape = batch_shape + fa_shape + fb_shape
    if target_shape:
        out = out_flat.reshape(target_shape)
    else:
        # Pure dot product: collapse the [1,1] result of the GEMM to a 0-D scalar.
        out = out_flat.sum()

    intermediate = "".join(batch + free_a + free_b)
    return _einsum_permute_to(intermediate, so, out)


def _einsum_finalize(s_from, out_sub, op):
    drop = [i for i, c in enumerate(s_from) if c not in out_sub]
    if drop:
        op = op.sum(drop)
        ds = set(drop)
        s_from = "".join(c for i, c in enumerate(s_from) if i not in ds)
    return _einsum_permute_to(s_from, out_sub, op)

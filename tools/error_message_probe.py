#!/usr/bin/env python3
"""Print the text a user actually receives for the errors they actually hit.

This is not a test. A test asserts a property and goes red; a probe shows the
whole message so a reader can judge it the way a user would, and a diff of two
runs -- before and after a change -- is the acceptance record for that change.

Each case is one mistake a user makes with a public API: a shape that does not
match, a dimension that does not exist, an index past the end, an argument of
the wrong type, an operation a dtype does not support. The case is run, the
exception (or its absence) is caught, and the exact text is printed. Three
things are worth reading for in the output:

* **no error at all** -- a case marked ``NO ERROR`` computed something. That
  is either a missing check or a legal operation; NumPy and PyTorch decide
  which, and the probe cannot;
* **a bare message** -- ``list index out of range`` or an empty
  ``AssertionError`` tells the reader nothing about which op, which shape,
  which value;
* **the wrong type** -- ``RuntimeError`` where ``IndexError`` or ``TypeError``
  is what an ``except`` clause would look for.

Usage::

    PYTHONPATH=<repo>/python python tools/error_message_probe.py [--json out.json]
    # run under an isolated JITTOR_HOME so it does not queue behind other work
    # --device cpu|cuda   (default cpu)
"""

import argparse
import json
import pathlib
import sys


def _cases(jt, np):
    """Each entry is (name, category, thunk). The thunk returns a Var or a
    tuple of Vars, which the runner syncs so lazy errors surface too."""
    C = []

    def case(name, category):
        def deco(fn):
            C.append((name, category, fn))
            return fn
        return deco

    # ---- shape mismatch ------------------------------------------------
    @case("binary_add_shape_mismatch", "shape")
    def _(): return jt.ones((3, 4)) + jt.ones((5, 6))

    @case("matmul_inner_dim_mismatch", "shape")
    def _(): return jt.matmul(jt.ones((3, 4)), jt.ones((5, 6)))

    @case("matmul_batch_dim_mismatch", "shape")
    def _(): return jt.matmul(jt.ones((2, 3, 4)), jt.ones((3, 4, 5)))

    @case("matmul_scalar_operand", "shape")
    def _(): return jt.matmul(jt.array(1.0), jt.ones((3,)))

    @case("bmm_with_2d_operand", "shape")
    def _(): return jt.nn.bmm(jt.ones((3, 4)), jt.ones((4, 5)))

    @case("bmm_transpose_with_2d_operand", "shape")
    def _(): return jt.nn.bmm_transpose(jt.ones((3, 4)), jt.ones((5, 4)))

    @case("matmul_transpose_non_2d_b", "shape")
    def _(): return jt.nn.matmul_transpose(jt.ones((3, 4)), jt.ones((2, 5, 4)))

    @case("linear_in_features_mismatch", "shape")
    def _(): return jt.nn.Linear(4, 5)(jt.ones((3, 6)))

    @case("conv2d_in_channels_mismatch", "shape")
    def _(): return jt.nn.Conv2d(3, 8, 3)(jt.ones((1, 4, 8, 8)))

    @case("reshape_element_count_mismatch", "shape")
    def _(): return jt.ones((3, 4)).reshape((5, 5))

    @case("view_element_count_mismatch", "shape")
    def _(): return jt.ones((3, 4)).view(5, 5)

    @case("concat_shape_mismatch", "shape")
    def _(): return jt.concat([jt.ones((3, 4)), jt.ones((3, 5))], 0)

    @case("stack_shape_mismatch", "shape")
    def _(): return jt.stack([jt.ones((3, 4)), jt.ones((3, 5))])

    @case("expand_incompatible", "shape")
    def _(): return jt.ones((3, 1)).expand(4, 4)

    @case("broadcast_to_incompatible", "shape")
    def _(): return jt.ones((3, 1)).broadcast((4, 4))

    @case("setitem_shape_mismatch", "shape")
    def _():
        x = jt.zeros((3, 4))
        x[0] = jt.ones(5)
        return x

    @case("where_shape_mismatch", "shape")
    def _(): return jt.where(jt.ones((3,)) > 0, jt.ones((3,)), jt.ones((4,)))

    @case("cross_entropy_batch_mismatch", "shape")
    def _(): return jt.nn.cross_entropy_loss(jt.ones((3, 5)), jt.array([0, 1, 2, 3]))

    @case("split_sizes_do_not_sum", "shape")
    def _(): return jt.split(jt.ones((3, 4)), [1, 1], 0)

    @case("linalg_inv_non_square", "shape")
    def _(): return jt.linalg.inv(jt.ones((3, 4)))

    # ---- dimension / axis ----------------------------------------------
    @case("transpose_axis_out_of_range", "dim")
    def _(): return jt.ones((3, 4)).transpose(0, 5)

    @case("transpose_negative_axis_out_of_range", "dim")
    def _(): return jt.ones((3, 4)).transpose(0, -3)

    @case("permute_wrong_axis_count", "dim")
    def _(): return jt.ones((3, 4)).permute(0, 1, 2)

    @case("permute_repeated_axis", "dim")
    def _(): return jt.ones((3, 4)).permute(0, 0)

    @case("permute_repeated_axis_sequence", "dim")
    def _(): return jt.ones((3, 4)).permute((0, 0))

    @case("sum_dim_out_of_range", "dim")
    def _(): return jt.ones((3, 4)).sum(5)

    @case("max_dim_out_of_range", "dim")
    def _(): return jt.ones((3, 4)).max(5)

    @case("argmax_dim_out_of_range", "dim")
    def _(): return jt.ones((3, 4)).argmax(5)

    @case("cumsum_dim_out_of_range", "dim")
    def _(): return jt.cumsum(jt.ones((3, 4)), 5)

    @case("cumprod_dim_out_of_range", "dim")
    def _(): return jt.cumprod(jt.ones((3, 4)), 5)

    @case("unsqueeze_dim_out_of_range", "dim")
    def _(): return jt.ones((3, 4)).unsqueeze(5)

    @case("squeeze_dim_out_of_range", "dim")
    def _(): return jt.ones((3, 1)).squeeze(5)

    @case("concat_dim_out_of_range", "dim")
    def _(): return jt.concat([jt.ones((3, 4))], 5)

    @case("softmax_dim_out_of_range", "dim")
    def _(): return jt.nn.softmax(jt.ones((3, 4)), dim=5)

    @case("flatten_end_before_start", "dim")
    def _(): return jt.ones((2, 3, 4)).flatten(2, 1)

    @case("flatten_dim_out_of_range", "dim")
    def _(): return jt.ones((2, 3, 4)).flatten(0, 5)

    @case("chunk_dim_out_of_range", "dim")
    def _(): return jt.chunk(jt.ones((3, 4)), 2, 5)

    @case("flip_dim_out_of_range", "dim")
    def _(): return jt.flip(jt.ones((3, 4)), 5)

    @case("gather_index_ndim_mismatch", "dim")
    def _(): return jt.ones((3, 4)).gather(0, jt.array([0, 1, 2]))

    @case("gather_index_too_wide", "dim")
    def _(): return jt.ones((3, 4)).gather(0, jt.zeros((3, 6), dtype="int32"))

    @case("gather_dim_out_of_range", "dim")
    def _(): return jt.ones((3, 4)).gather(5, jt.zeros((3, 4), dtype="int32"))

    @case("index_select_2d_index", "dim")
    def _(): return jt.ones((3, 4)).index_select(0, jt.zeros((2, 2), dtype="int32"))

    @case("scatter_index_ndim_mismatch", "dim")
    def _(): return jt.zeros((3, 4)).scatter(0, jt.array([0, 1, 2]), jt.ones(3))

    @case("scatter_dim_out_of_range", "dim")
    def _():
        return jt.zeros((3, 4)).scatter(5, jt.zeros((3, 4), dtype="int32"),
                                        jt.ones((3, 4)))

    # ---- out of bounds -------------------------------------------------
    @case("getitem_int_out_of_bounds", "bounds")
    def _(): return jt.ones(5)[10]

    @case("getitem_var_index_out_of_bounds", "bounds")
    def _(): return jt.ones(5)[jt.array([99])]

    @case("getitem_too_many_indices", "bounds")
    def _(): return jt.ones((3, 4))[0, 0, 0]

    @case("embedding_index_out_of_range", "bounds")
    def _(): return jt.nn.Embedding(10, 4)(jt.array([0, 99]))

    # ---- dtype ---------------------------------------------------------
    @case("bitwise_and_on_float", "dtype")
    def _(): return jt.ones((3,)) & jt.ones((3,))

    @case("bool_true_div_bool", "dtype")
    def _(): return jt.array([True, False]) / jt.array([True, True])

    @case("cast_to_unknown_dtype", "dtype")
    def _(): return jt.ones((3,)).cast("float128")

    @case("array_of_strings", "dtype")
    def _(): return jt.array(["a", "b"])

    # ---- empty tensors -------------------------------------------------
    @case("sum_of_zero_length", "empty")
    def _(): return jt.zeros((0,)).sum()

    @case("max_of_zero_length", "empty")
    def _(): return jt.zeros((0,)).max()

    @case("argmax_of_zero_length", "empty")
    def _(): return jt.zeros((0,)).argmax(0)

    @case("mean_of_zero_length", "empty")
    def _(): return jt.zeros((0,)).mean()

    @case("min_of_zero_length", "empty")
    def _(): return jt.zeros((0,)).min()

    @case("prod_of_zero_length", "empty")
    def _(): return jt.zeros((0,)).prod()

    @case("max_of_zero_length_along_a_full_dim", "empty")
    def _(): return jt.zeros((0, 3)).max(0)

    @case("max_of_zero_length_along_a_nonempty_dim", "empty")
    def _(): return jt.zeros((0, 3)).max(1)

    @case("argmin_of_zero_length", "empty")
    def _(): return jt.zeros((0, 3)).argmin(0)

    # ---- wrong argument type / value -----------------------------------
    @case("ones_negative_dim", "arg")
    def _(): return jt.ones((-1, 3))

    @case("ones_string_shape", "arg")
    def _(): return jt.ones("abc")

    @case("sum_string_dim", "arg")
    def _(): return jt.ones((3, 4)).sum("x")

    @case("transpose_string_axis", "arg")
    def _(): return jt.ones((3, 4)).transpose("a", "b")

    @case("arange_zero_step", "arg")
    def _(): return jt.arange(0, 10, 0)

    @case("pad_odd_length", "arg")
    def _(): return jt.nn.pad(jt.ones((3, 4)), (1, 2, 3))

    @case("pad_unknown_mode", "arg")
    def _(): return jt.nn.pad(jt.ones((3, 4)), (1, 1), mode="mirror")

    @case("dropout_p_out_of_range", "arg")
    def _(): return jt.nn.dropout(jt.ones((3, 4)), p=1.5)

    @case("array_ragged_list", "arg")
    def _(): return jt.array([[1, 2], [3]])

    @case("concat_not_a_sequence", "arg")
    def _(): return jt.concat(jt.ones((3, 4)), 0)

    @case("norm_unsupported_p", "arg")
    def _(): return jt.ones((3, 4)).norm(p=3)

    @case("conv2d_groups_not_dividing", "arg")
    def _(): return jt.nn.Conv2d(3, 8, 3, groups=2)

    return C


def _run_one(jt, fn):
    """Return ("error", type_name, text) or ("ok", None, description)."""
    try:
        out = fn()
        if isinstance(out, (list, tuple)):
            for o in out:
                if hasattr(o, "sync"):
                    o.sync()
            desc = [(list(o.shape), str(o.dtype)) for o in out if hasattr(o, "shape")]
        elif hasattr(out, "sync"):
            out.sync()
            desc = (list(out.shape), str(out.dtype))
        else:
            desc = repr(out)[:200]
        return ("ok", None, desc)
    except BaseException as e:  # noqa: BLE001 -- we are recording, not handling
        return ("error", type(e).__name__, str(e))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", type=pathlib.Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--only", default=None,
                        help="substring filter on the case name")
    parser.add_argument("--full", action="store_true",
                        help="print every message in full, not the first lines")
    args = parser.parse_args(argv)

    import numpy as np
    import jittor as jt

    use_cuda = 1 if args.device == "cuda" else 0
    if use_cuda and not jt.has_cuda:
        print("CUDA requested but jt.has_cuda is false", file=sys.stderr)
        return 2

    records = []
    with jt.flag_scope(use_cuda=use_cuda):
        for name, category, fn in _cases(jt, np):
            if args.only and args.only not in name:
                continue
            kind, etype, payload = _run_one(jt, fn)
            records.append({
                "name": name, "category": category, "kind": kind,
                "type": etype, "text": payload if kind == "error" else None,
                "result": payload if kind == "ok" else None,
            })

    width = max(len(r["name"]) for r in records) if records else 0
    n_silent = 0
    for r in records:
        print("=" * 78)
        if r["kind"] == "ok":
            n_silent += 1
            print("%-*s  NO ERROR  -> %s" % (width, r["name"], r["result"]))
            continue
        text = r["text"]
        lines = text.splitlines() or [""]
        shown = lines if args.full else lines[:6]
        print("%-*s  %s (%d chars, %d lines)" % (
            width, r["name"], r["type"], len(text), len(lines)))
        for line in shown:
            print("    " + line)
        if len(lines) > len(shown):
            print("    ... (%d more lines)" % (len(lines) - len(shown)))
    print("=" * 78)
    print("%d cases, %d raised, %d silent" % (
        len(records), len(records) - n_silent, n_silent))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())

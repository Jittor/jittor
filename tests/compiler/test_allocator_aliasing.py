# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Dun Liang <randonlang@gmail.com>.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Aliasing and cross-stream contracts of the allocator layer.

``Var::share_with`` makes two vars point into one allocation; ``reshape``,
``getitem``/``setitem`` in place, ``clone``, ``tape`` and ``fused_adamw`` all
use it.  Nothing else in the runtime records that a var *is* a sub-range of
another one, so every piece of code that moves memory around underneath a var
has to ask.  These tests pin down the places that did not.

See ``agent/skills/jittor-allocator-flag-matrix`` for the flag combinations and
for the poison-then-read technique used below.
"""
import gc
import unittest

import numpy as np

import jittor as jt

def _var_mem_ptrs():
    """{node id: mem_ptr} for every Var in the current graph.

    ``operator<<(ostream&, const Var&)`` prints the mem_ptr in hex and
    ``dump_all_graphs().nodes_info`` is the only way to read it from Python.
    Aliasing is a property of that pointer, so asserting on it directly is the
    only way to test the alias rather than one of its symptoms.
    """
    out = {}
    for info in jt.dump_all_graphs().nodes_info:
        if not info.startswith("Var("):
            continue
        body = info[len("Var("):]
        node_id = body.split(":", 1)[0]
        # Var(id:f:b:p:iN:oN:sN:nN:gN,dtype,name,memptr)shape
        fields = body.split(",")
        out[node_id] = fields[3].split(")")[0]
    return out


def _alias_groups(ptrs):
    """Sets of node ids that share one mem_ptr (the null pointer aside)."""
    by_ptr = {}
    for node_id, ptr in ptrs.items():
        if ptr in ("0", ""):
            continue
        by_ptr.setdefault(ptr, set()).add(node_id)
    return [ids for ids in by_ptr.values() if len(ids) > 1]


@unittest.skipIf(not jt.has_cuda, "Cuda not found")
class TestMigrateKeepsShareAlias(unittest.TestCase):
    """``migrate_to_cpu``/``migrate_to_gpu`` must not silently unshare.

    Both used to allocate a private block, memcpy into it and free the old
    one.  For a var produced by ``Var::share_with`` that block belongs to
    another var, and the migration code had no way to know: the child's
    ``allocation`` had already been overwritten with the parent's.  The alias
    was therefore dropped without a word, and a later in-place write through
    one of the two vars stopped being visible through the other.
    """

    def test_reshape_alias_survives_a_host_read(self):
        # ``.numpy()`` on a device var goes through fetch_sync ->
        # migrate_to_cpu, which *moves* the var into host memory for good.
        with jt.flag_scope(use_cuda=1):
            a = jt.array(np.zeros(6, "float32")) + 0.0
            b = a.reshape((2, 3))           # b shares a's allocation
            jt.sync_all(True)
            groups = _alias_groups(_var_mem_ptrs())
            assert groups, "reshape produced no alias; the test would be void"
            group = max(groups, key=len)
            b.numpy()                       # migrates b off the device
            jt.sync_all(True)
            after = _var_mem_ptrs()
            still = {after[i] for i in group if i in after}
            assert len(still) <= 1, (
                "migrate_to_cpu unshared an aliased var; the group "
                f"{sorted(group)} now points at {sorted(still)}")
            del a, b
            gc.collect()

    def test_write_through_one_alias_is_visible_through_the_other(self):
        """The value-level consequence of the same defect.

        ``a[i] = v`` is rewritten by setitem_gopt into an op whose output
        shares ``a``'s allocation and writes in place, so a reshape view of
        ``a`` sees the write -- unless the view was migrated away first.
        """
        with jt.flag_scope(use_cuda=1):
            a = jt.array(np.zeros(6, "float32")) + 0.0
            b = a.reshape((2, 3))
            jt.sync_all(True)
            assert _alias_groups(_var_mem_ptrs()), "no alias, test is void"
            # the poison: reading b on the host moves b off the device
            np.testing.assert_array_equal(b.numpy(), np.zeros((2, 3), "float32"))
            a[1] = 7.0
            jt.sync_all(True)
            np.testing.assert_array_equal(
                b.numpy(),
                np.array([[0, 7, 0], [0, 0, 0]], "float32"),
                err_msg="a write through one alias was lost by migration")
            del a, b
            gc.collect()


if __name__ == "__main__":
    unittest.main()

# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Every rank must be running the same graph (8.11).

``MpiBroadcastOp::infer_shape`` used to end with::

    if (root == mpi_world_rank)
        y->share_with(x);

so the broadcast's output aliased its input on the root and was a fresh
allocation everywhere else. Two things were wrong with that, and the second is
the expensive one:

* shape inference decided an aliasing question, which is not a shape;
* the ranks stopped running the same graph. A graph that differs by rank fuses
  differently, schedules differently and allocates differently, and when that
  produces a wrong number the symptom appears nowhere near the ``share_with``
  that caused it. "The ranks agree" is a property worth asserting outright, so
  this file asserts it rather than trusting a comment.

The check compares a canonical description of the whole live graph: every
node's printout with its id and its raw address removed, plus the adjacency
between nodes. Addresses are replaced by an **alias class** -- the index of the
distinct address, in order of first appearance -- because that, and not the
address itself, is the thing ``share_with`` changes and the thing that has to
match across ranks.
"""
import os
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np

import jittor as jt

from _helpers.distributed import run_mpi_test

_N = 3

# Distinctive enough to be obvious in a failure dump, small enough to be free.
_SHAPE = (7,)


def _shared_dir():
    """One directory both ranks agree on, without a rendezvous of its own.

    ``mpirun`` starts every rank from the same parent, so the parent's pid
    names a directory all of them compute identically.
    """
    path = os.path.join(tempfile.gettempdir(),
                        "jt_graph_iso_%d" % os.getppid())
    os.makedirs(path, exist_ok=True)
    return path


def _barrier():
    """A collective is a barrier. Runs after the graph has been captured."""
    jt.array([1.0]).mpi_all_reduce("add").sync()


_NODE = re.compile(r"^(Var|Op)\((\d+):")


def _canonical(nodes_info, inputs, outputs):
    """A description of the graph that two ranks can compare literally.

    Node ids and raw addresses are per-process facts, so both are removed --
    but addresses are removed by *grouping*: two nodes at the same address
    become the same alias class, and a rank where the broadcast's output shares
    its input's buffer therefore reads differently from one where it does not.
    That is the whole point of the check.
    """
    classes = {}
    lines = []
    for index, info in enumerate(nodes_info):
        text = _NODE.sub(lambda m: m.group(1) + "(", info)
        # "Var(...,dtype,name,<hex>)shape" -- the address is the last field
        # before the closing paren.
        head, sep, tail = text.partition(")")
        if sep and "," in head:
            fields = head.split(",")
            address = fields[-1]
            if address not in classes:
                classes[address] = len(classes)
            fields[-1] = "#%d" % classes[address]
            text = ",".join(fields) + ")" + tail
        lines.append("%s|in=%s|out=%s"
                     % (text, sorted(inputs[index]), sorted(outputs[index])))
    return "\n".join(lines)


def _shape_of(info):
    """The shape a node's printout ends with. "Var(...)[7,]" -> (7,)."""
    tail = info.rpartition(")")[2].strip()
    return tuple(int(part) for part in tail.strip("[]").split(",") if part.strip())


def _capture():
    graphs = jt.core.dump_all_graphs()
    return _canonical(list(graphs.nodes_info),
                      [list(v) for v in graphs.inputs],
                      [list(v) for v in graphs.outputs])


@unittest.skipIf(not jt.in_mpi, "requires an MPI launch")
class TestMpiGraphIsomorphism(unittest.TestCase):

    def test_broadcast_leaves_every_rank_with_the_same_graph(self):
        rank = jt.rank
        # Rank-dependent contents, so a broadcast that did nothing would show.
        x = jt.array(np.full(_SHAPE, rank + 1, dtype="float32"))
        y = x.mpi_broadcast(0)
        y.sync()
        x.sync()
        # Capture before the barrier, so the barrier's own ops are not in it.
        mine = _capture()

        # The broadcast still works: every rank ends up with rank 0's values.
        np.testing.assert_array_equal(y.numpy(), np.full(_SHAPE, 1.0, "float32"))

        directory = _shared_dir()
        Path(directory, "rank%d.txt" % rank).write_text(mine)
        _barrier()
        reference = Path(directory, "rank0.txt").read_text()
        if rank != 0:
            self.assertEqual(
                mine.splitlines(), reference.splitlines(),
                "rank %d is running a different graph from rank 0. An alias "
                "class that differs by rank means one rank's output shares a "
                "buffer that another rank's does not -- which is exactly what "
                "MpiBroadcastOp::infer_shape used to do on the root." % rank)
        _barrier()

    def test_broadcast_output_does_not_alias_its_input(self):
        """Stated directly, so the failure names the cause and not a diff.

        The graph comparison above would also fail if *both* ranks aliased, so
        it needs this alongside it: on every rank, the two live vars of this
        shape sit at two different addresses.
        """
        x = jt.array(np.full(_SHAPE, jt.rank + 1, dtype="float32"))
        y = x.mpi_broadcast(0)
        y.sync()
        x.sync()
        addresses = []
        for info in jt.core.dump_all_graphs().nodes_info:
            if not info.startswith("Var(") or _shape_of(info) != _SHAPE:
                continue
            head, _, _ = info.partition(")")
            addresses.append(head.split(",")[-1])
        self.assertGreaterEqual(len(addresses), 2, addresses)
        self.assertEqual(len(addresses), len(set(addresses)),
                         "two live vars share one buffer on rank %d: %s -- the "
                         "broadcast's output is aliasing its input again"
                         % (jt.rank, addresses))


@unittest.skipIf(not jt.compile_extern.has_mpi, "no mpi found")
class TestMpiGraphIsomorphismEntry(unittest.TestCase):
    def test_entry(self):
        run_mpi_test(_N, "test_mpi_graph_isomorphism")


if __name__ == "__main__":
    unittest.main()

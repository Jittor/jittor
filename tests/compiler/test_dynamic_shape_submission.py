"""Dynamic outputs become concrete after construction, at submission boundaries."""
import numpy as np
import pytest
import jittor as jt
import jittor_utils


@pytest.fixture(scope="module")
def construction_probe():
    return jittor_utils.compile_module(r'''
#include "core/var_holder.h"
#include "ops/op_register.h"
namespace jittor {
DECLARE_FLAG(int, exec_called);
// @pyjt(construct_where)
vector<int64> construct_where(VarHolder* input) {
    auto make_where = op_constructor<vector<VarPtr>, Var*, NanoString>("where");
    int before = exec_called;
    auto outputs = make_where(input->var, ns_int32);
    return {exec_called-before, outputs.at(0)->num};
}
// @pyjt(finished)
bool finished(VarHolder* input) { return input->var->is_finished(); }
// @pyjt(compose_dynamic)
VarHolder* compose_dynamic(VarHolder* input, bool reduce) {
    auto make_where = op_constructor<vector<VarPtr>, Var*, NanoString>("where");
    auto make_binary = op_constructor<VarPtr, Var*, Var*, NanoString>("binary");
    int before = exec_called;
    auto indices = make_where(input->var, ns_int32);
    auto doubled = make_binary(indices.at(0).ptr, indices.at(0).ptr, ns_add);
    if (reduce) {
        auto make_reduce = op_constructor<VarPtr, Var*, NanoString, NanoVector, bool>("reduce");
        doubled = make_reduce(doubled.ptr, ns_add, NanoVector(), false);
    }
    CHECK(exec_called == before);
    return new VarHolder(doubled.ptr);
}
}''', jt.compiler.cc_flags)


def test_dynamic_constructor_does_not_execute_graph(construction_probe):
    with jt.flag_scope(use_cuda=0, lazy_execution=1):
        x = jt.array([0, 1, 0, 1])
        x.sync()
        assert construction_probe.construct_where(x) == [0, -4]
        indices, = jt.where(x)
        assert tuple(indices.shape) == (2,)
        np.testing.assert_array_equal(indices.numpy(), [1, 3])
        doubled = construction_probe.compose_dynamic(x, False)
        assert tuple(doubled.shape) == (2,)
        np.testing.assert_array_equal(doubled.numpy(), [2, 6])
        total = construction_probe.compose_dynamic(x, True)
        assert total.item() == 8


def test_item_does_not_submit_sibling_branch(construction_probe):
    with jt.flag_scope(use_cuda=0, lazy_execution=1):
        x = jt.array([1., 2., 3.])
        x.sync()
        sibling = x + 9
        total = (x * 2).sum()
        assert not construction_probe.finished(sibling)
        assert total.item() == 12
        assert not construction_probe.finished(sibling)
        np.testing.assert_array_equal(sibling.numpy(), [10, 11, 12])


def test_fetch_ready_and_backpressure_submit_after_construction():
    with jt.flag_scope(use_cuda=0, lazy_execution=1):
        x = jt.array([3.])
        x.sync()
        seen = []
        jt.fetch(x, lambda values: seen.append(float(values[0])))
        assert seen == [3.]
        for i in range(25):
            jt.fetch(x + i, lambda values: seen.append(float(values[0])))
        jt.sync_all(True)
        assert sorted(seen[1:]) == list(range(3, 28))

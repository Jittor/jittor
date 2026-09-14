
from _helpers.runtime_policy import preserve_policy as _test_preserve_policy

from _helpers import capability as _test_capability
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
from _helpers.assertions import expect_error
from _helpers.numerical_grad import ngrad
from itertools import permutations

def gen_data(shape):
    num = np.multiply.reduce(shape)
    a = np.arange(0, num)
    return a.reshape(shape)

class TestTransposeAxisArguments(unittest.TestCase):
    """What ``transpose``/``permute`` say about an axis argument they reject.

    Every case here used to answer with a container's own error -- ``list index
    out of range``, ``list indices must be integers or slices, not str`` -- or
    with a C++ check whose text is the condition that failed. None of them named
    the operation, the axis, the rank it was measured against, or the shape, so
    the reader got the fact that something was wrong and nothing else.

    The assertions are on the exception *type* and on the facts the message has
    to carry (the rejected axis, the rank, the shape). They deliberately do not
    pin the sentence.
    """

    def test_axis_past_the_end_names_the_axis_and_the_rank(self):
        with self.assertRaises(IndexError) as caught:
            jt.ones((3, 4)).transpose(0, 5)
        text = str(caught.exception)
        self.assertIn("transpose", text)
        self.assertIn("5", text)
        self.assertIn("2-D", text)
        self.assertIn("[-2, 1]", text)
        self.assertNotIn("list index out of range", text)

    def test_negative_axis_past_the_end_is_the_same_report(self):
        with self.assertRaises(IndexError) as caught:
            jt.ones((3, 4)).transpose(0, -3)
        text = str(caught.exception)
        self.assertIn("transpose", text)
        self.assertIn("-3", text)
        self.assertIn("[-2, 1]", text)

    def test_non_integer_axis_is_a_type_error_that_names_the_type(self):
        with self.assertRaises(TypeError) as caught:
            jt.ones((3, 4)).transpose("a", "b")
        text = str(caught.exception)
        self.assertIn("transpose", text)
        self.assertIn("str", text)
        self.assertNotIn("list indices", text)

    def test_a_0d_var_says_it_has_no_dims(self):
        # transpose_op.cc requires rank >= 1; before, the empty axes list
        # answered for it with ``list index out of range``.
        with self.assertRaises(IndexError) as caught:
            jt.array(1.0).transpose(0, 0)
        self.assertIn("0-D", str(caught.exception))

    def test_a_repeated_axis_in_the_sequence_form_is_rejected(self):
        # `permute` and `transpose` are the same callable, so the two-argument
        # form is a swap and `(0, 0)` there is a legal no-op. The sequence form
        # is a permutation, and a permutation cannot name a dim twice: it used
        # to reach transpose_op.cc and come back as "Invalid axes [0,0,]".
        for axes in ((0, 0), (1, 1)):
            with self.assertRaises(RuntimeError) as caught:
                jt.ones((2, 3)).permute(axes)
            text = str(caught.exception)
            self.assertIn("transpose", text)
            self.assertIn("twice", text)
            self.assertIn("[2, 3]", text)

    def test_a_repeated_axis_given_as_separate_arguments_is_rejected(self):
        with self.assertRaises(RuntimeError) as caught:
            jt.ones((2, 3, 4)).permute(0, 1, 1)
        self.assertIn("twice", str(caught.exception))

    def test_the_wrong_number_of_axes_names_both_counts(self):
        for axes in ((0,), (0, 1, 2)):
            with self.assertRaises(RuntimeError) as caught:
                jt.transpose(jt.ones((2, 3)), axes).sync()
            text = str(caught.exception)
            self.assertIn("transpose", text)
            self.assertIn(str(len(axes)), text)
            self.assertIn("2-D", text)
            self.assertIn("[2, 3]", text)

    def test_an_axis_out_of_range_inside_the_sequence_names_its_position(self):
        with self.assertRaises(IndexError) as caught:
            jt.ones((2, 3, 4)).permute(0, 1, 5)
        text = str(caught.exception)
        self.assertIn("dims[2]", text)
        self.assertIn("[-3, 2]", text)

    def test_the_legal_spellings_still_work(self):
        a = jt.array(gen_data([2, 3, 4])).float()
        reference = np.asarray(gen_data([2, 3, 4]), dtype="float32")
        np.testing.assert_allclose(a.transpose(0, 2).data,
                                   reference.transpose(2, 1, 0))
        np.testing.assert_allclose(a.transpose(-1, -3).data,
                                   reference.transpose(2, 1, 0))
        np.testing.assert_allclose(a.permute(2, 0, 1).data,
                                   reference.transpose(2, 0, 1))
        np.testing.assert_allclose(a.permute([2, 0, 1]).data,
                                   reference.transpose(2, 0, 1))
        np.testing.assert_allclose(a.permute(list(np.array([2, 0, 1]))).data,
                                   reference.transpose(2, 0, 1))
        np.testing.assert_allclose(a.transpose().data, reference.transpose())
        np.testing.assert_allclose(jt.ones((3,)).transpose(0, 0).data,
                                   np.ones(3, dtype="float32"))


class TestTransposeOp(unittest.TestCase):
    def test_invalid_axes_is_a_catchable_user_error(self):
        # `jt.transpose` now rejects a repeated axis in the frontend (see
        # TestTransposeAxisArguments); `fuse_transpose` is the remaining caller
        # of the op's own check, so it is what keeps that check covered.
        with self.assertRaisesRegex(RuntimeError, "Invalid axes"):
            jt.ones((2, 3)).fuse_transpose((0, 0)).sync()

    def test_axes_shorter_than_input_is_a_catchable_user_error(self):
        # `(0,)` counts up from 0, so it used to satisfy the constructor's
        # "axes[i]==i for all i" identity test and forward the input unchanged.
        # infer_shape, which holds the rank check, was never reached: a wrong
        # axes argument silently returned the untransposed input.
        for axes in ((0,), (0, 1, 2)):
            with self.assertRaisesRegex(RuntimeError, "axes.size"):
                jt.ones((2, 3)).fuse_transpose(axes).sync()

    def test_fuse_transpose_axes_shorter_than_input_is_a_catchable_user_error(self):
        for axes in ((0,), (0, 1, 2)):
            with self.assertRaisesRegex(RuntimeError, "axes.size"):
                jt.ones((2, 3)).fuse_transpose(axes).sync()

    def test_identity_axes_still_forward_the_input(self):
        a = jt.array(gen_data([2, 3])).float()
        np.testing.assert_allclose(jt.transpose(a, (0, 1)).data, a.data)
        np.testing.assert_allclose(a.fuse_transpose((0, 1)).data, a.data)

    def test_with_np(self):
        def check(a):
            perms = list(permutations(range(a.ndim))) + [None]
            for perm in perms:
                if perm:
                    x = np.transpose(a, perm)
                    y = jt.transpose(a, perm).data
                else:
                    x = np.transpose(a)
                    y = jt.transpose(a).data
                self.assertEqual(x.shape, y.shape)
                assert (x==y).all(), f"\n{x}\n{y}"
                
        # ia = [gen_data([2,3,4,5]), gen_data([5,3])]
        ia = [gen_data([2,2,2]), gen_data([2,3,4,5]), gen_data([5,3])]
        for a in ia: check(a)
        
    def test_grad(self):
        def check(a):
            perms = list(permutations(range(a.ndim))) + [None]
            for perm in perms:
                x = jt.array(a).float()
                if perm:
                    y = x.transpose(perm)
                else:
                    y = x.transpose()
                dx = jt.grad(y*y, x).data
                self.assertEqual(dx.shape, a.shape)
                assert (dx==a*2).all(), f"\n{dx}\n{a}\n{perm}"
        ia = [gen_data([2,2,2]), gen_data([2,3,4,5]), gen_data([5,3])]
        for a in ia: check(a)
        
    def test_matmul_grad(self):
        np.random.seed(0)
        for i in range(10):
            a = np.random.rand(2,3).astype("float32")
            b = np.random.rand(3,4).astype("float32")
            out, (da, db) = ngrad(lambda vars: np.matmul(vars[0],vars[1]).sum(), [a,b], 1e-1)
            ja = jt.array(a)
            jb = jt.array(b)
            jc = ja.matmul(jb)
            jda, jdb = jt.grad(jc, [ja,jb])
            assert ((da-jda.data)<1e-5).all(), (da, jda.data, da-jda.data)
            assert ((db-jdb.data)<1e-5).all(), (db-jdb.data)

    def test_permute(self):
        a = jt.ones([2,3,4])
        assert a.permute().shape == [4,3,2]
        assert a.permute(0,2,1).shape == [2,4,3]

    def test_transpose_3d2i(self):
        a = jt.ones([2,3,4])
        assert a.transpose(0,1).shape == (3,2,4)

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_cutt(self):
        a = jt.rand((10,2)) > 0.5
        b = a.transpose()
        a_data = np.array(a.data, copy=True)
        b_data = np.array(b.data, copy=True)
        np.testing.assert_array_equal(a_data.transpose(), b_data)

        a = jt.zeros((1,1))
        b = a.transpose((1,0))
        b.sync()

    @unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
    @jt.flag_scope(use_cuda=1)
    def test_cutt_bug(self):
        a = jt.rand(640000,4,3)
        b = a.transpose(0,2,1)
        b.sync(True)
        print(a.shape, b.shape)


class TestFuseTransposeOp(unittest.TestCase):

    def test_fuse_transpose1(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,12))
            b = a.fuse_transpose((1,2,0))+1
            np.testing.assert_allclose(
                a.data.transpose((1,2,0))+1,
                b.data
            )
        assert len(rep) == 3

    def test_fuse_transpose2(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,12))
            b = (a+1).fuse_transpose((1,2,0))
            np.testing.assert_allclose(
                a.data.transpose((1,2,0))+1,
                b.data
            )
        assert len(rep) == 3

    def test_fuse_transpose3(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,12))
            c = jt.rand((11,12,10))
            b = a.fuse_transpose((1,2,0))+c
            np.testing.assert_allclose(
                a.data.transpose((1,2,0))+c.data,
                b.data
            )
        assert len(rep) == 3

    def test_fuse_transpose4(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,12))
            c = jt.rand((10,11,12))
            b = (a+c).fuse_transpose((1,2,0))
            np.testing.assert_allclose(
                (a.data+c.data).transpose((1,2,0)),
                b.data
            )
        assert len(rep) == 3

    def test_fuse_transpose5(self):
        with jt.profile_scope() as rep:
            a = jt.rand((10,11,6,7))
            c = jt.rand((10,11,6,7))
            b = (a+c).fuse_transpose((1,0,2,3))
            np.testing.assert_allclose(
                (a.data+c.data).transpose((1,0,2,3)),
                b.data
            )
        assert len(rep) == 3


@_test_preserve_policy(jt, 'use_cuda')
@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestFuseTransposeCudaOp(TestFuseTransposeOp):
    def setUp(self):
        from contextlib import ExitStack as _TestPolicyStack
        _test_policy_stack = _TestPolicyStack()
        self.addCleanup(_test_policy_stack.close)
        self._previous_use_cuda = jt.introspection.policy.runtime.use_cuda
        _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=1))
    def tearDown(self):
        from contextlib import ExitStack as _TestPolicyStack
        with _TestPolicyStack() as _test_policy_stack:
            jt.sync_all()
            _test_policy_stack.enter_context(jt.runtime.scope(use_cuda=self._previous_use_cuda))

if __name__ == "__main__":
    unittest.main()

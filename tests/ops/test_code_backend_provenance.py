"""Explicit accelerator-source provenance survives native CodeOp construction."""

import numpy as np
import pytest

import jittor as jt


def test_unknown_backend_is_rejected_at_construction():
    with jt.flag_scope(use_cuda=0):
        with pytest.raises(RuntimeError, match="code backend must be"):
            jt.code((2,), "float32", [], cpu_src="@out(0)=1; @out(1)=2;",
                    backend="unknown_backend")


def test_cpu_forward_and_reverse_preserve_source_provenance():
    with jt.flag_scope(use_cuda=0):
        x = jt.array(np.array([1, 2, 3], dtype=np.float32))
        y = jt.code(x.shape, x.dtype, [x], backend="acl", cpu_src="""
            CHECK(backend == "acl");
            for (int i=0; i<in0_shape0; ++i) @out(i) = @in0(i)*2;
        """, cpu_grad_src=["""
            CHECK(backend == "acl");
            for (int i=0; i<in0_shape0; ++i) @out(i) = @dout(i)*2;
        """])
        gradient = jt.grad(y.sum(), x)
        np.testing.assert_array_equal(y.numpy(), [2, 4, 6])
        np.testing.assert_array_equal(gradient.numpy(), [2, 2, 2])


def test_multi_output_grad_preserves_source_provenance():
    with jt.flag_scope(use_cuda=0):
        x = jt.array(np.array([1, 2, 3], dtype=np.float32))
        y = jt.array(np.array([4, 5, 6], dtype=np.float32))
        _, product = jt.code([x.shape, x.shape], [x.dtype, x.dtype], [x, y],
            backend="acl", cpu_src="""
                CHECK(backend == "acl");
                for (int i=0; i<in0_shape0; ++i) {
                    @out0(i) = @in0(i) + @in1(i);
                    @out1(i) = @in0(i) * @in1(i);
                }
            """, cpu_grad_src=["""
                CHECK(backend == "acl");
                for (int i=0; i<in0_shape0; ++i) {
                    @out0(i) = @dout(i) * @in1(i);
                    @out1(i) = @dout(i) * @in0(i);
                }
            """], data={"multi_grad": 1, "multi_grad_output": 1})
        grad_x, grad_y = jt.grad(product.sum(), [x, y])
        np.testing.assert_array_equal(grad_x.numpy(), [4, 5, 6])
        np.testing.assert_array_equal(grad_y.numpy(), [1, 2, 3])


def test_preallocated_outputs_keep_provenance_and_identity():
    with jt.flag_scope(use_cuda=0):
        x = jt.array(np.array([1, 2, 3], dtype=np.float32))
        output = jt.empty(x.shape, x.dtype)
        alias = output
        returned = jt.code(inputs=[x], outputs=[output], backend="acl", cpu_src="""
            CHECK(backend == "acl");
            for (int i=0; i<in0_shape0; ++i) @out(i) = @in0(i)+5;
        """)
        assert output is alias
        np.testing.assert_array_equal(output.numpy(), [6, 7, 8])
        np.testing.assert_array_equal(returned[0].numpy(), alias.numpy())

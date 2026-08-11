import jittor as jt
from jittor.nn import ComplexNumber
import unittest
import numpy as np
from functools import partial

__skip_torch_test = False
try:
    import torch
except:
    __skip_torch_test = True

class TestResultAndGrad:
    def flatten_list(self, list_like):
        results = []
        if isinstance(list_like, (list, tuple)):
            for x in list_like:
                results.extend(self.flatten_list(x))
            return results
        else:
            return [list_like]

    def check_results(self, rlist1, rlist2):
        assert len(rlist1) == len(rlist2)
        for r1, r2 in zip(rlist1, rlist2):
            assert r1.shape == r2.shape
            assert np.allclose(r1, r2, rtol=1e-3, atol=1e-3)

    def grad_jittor(self, inputs, losses):
        grads = []
        for i in inputs:
            for loss in losses:
                if isinstance(i, ComplexNumber):
                    g = jt.grad(loss, i.value, retain_graph=True)
                    grads.append(g[..., 0].numpy() + 1j * g[..., 1].numpy())
                else:
                    g = jt.grad(loss, i, retain_graph=True)
                    grads.append(g.numpy())
        return grads

    def grad_torch(self, inputs, losses):
        grads = []
        for i in inputs:
            for loss in losses:
                g = torch.autograd.grad(loss, i, retain_graph=True)[0]
                grads.append(g.detach().cpu().numpy())
        return grads

    def run_jittor_op(self, op, input_list, weights=None, key_names=None, **kwargs):
        def _np_to_jittor(x):
            if isinstance(x, np.ndarray):
                if x.dtype == np.complex64 or x.dtype == np.complex128:
                    nx = np.stack([np.real(x), np.imag(x)], axis=-1)
                    return ComplexNumber(jt.array(nx, dtype=jt.float32), is_concat_value=True)
                elif x.dtype == np.float32 or x.dtype == np.float64:
                    return jt.array(x, dtype=jt.float32)
                else:
                    assert False
            elif isinstance(x, (list, tuple)):
                nx = [_np_to_jittor(vx) for vx in x]
                if isinstance(x, tuple):
                    return tuple(nx)
                return nx
            else:
                assert False
        def _jittor_to_np(x):
            if isinstance(x, jt.Var):
                return x.numpy()
            elif isinstance(x, ComplexNumber):
                return x.real.numpy() + 1j * x.imag.numpy()
            assert False
        ninput_list = [_np_to_jittor(x) for x in input_list]

        if key_names != None:
            assert len(ninput_list) == len(key_names)
            nkwargs = kwargs.copy()
            for k, v in zip(key_names, ninput_list):
                nkwargs[k] = v
            output_list = op(**nkwargs)
        else:
            output_list = op(*ninput_list, **kwargs)
        if isinstance(output_list, (jt.Var, ComplexNumber)):
            output_list = [output_list]
        output_list = self.flatten_list(output_list)
        losses = []
        if weights is None:
            weights = []
            for o in output_list:
                no = o.value if isinstance(o, ComplexNumber) else o
                w = np.random.randn(*no.shape)
                weights.append(w)
                losses.append(jt.sum(no * jt.array(w)))
        else:
            assert len(output_list) == len(weights)
            for o, w in zip(output_list, weights):
                no = o.value if isinstance(o, ComplexNumber) else o
                assert w.shape == no.shape
                losses.append(jt.sum(no * jt.array(w)))
        output_list = [_jittor_to_np(x) for x in output_list]
        return ninput_list, output_list, losses, weights

    def run_torch_op(self, op, input_list, weights=None, key_names=None, **kwargs):
        def _np_to_torch(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).requires_grad_(True)
            elif isinstance(x, (list, tuple)):
                nx = [_np_to_torch(vx) for vx in x]
                if isinstance(x, tuple):
                    return tuple(nx)
                return nx
            else:
                assert False
        def _torch_to_np(x:torch.Tensor) -> np.ndarray:
            return x.detach().cpu().numpy()
        ninput_list = [_np_to_torch(x) for x in input_list]
        if key_names != None:
            assert len(ninput_list) == len(key_names)
            nkwargs = kwargs.copy()
            for k, v in zip(key_names, ninput_list):
                nkwargs[k] = v
            output_list = op(**nkwargs)
        else:
            output_list = op(*ninput_list, **kwargs)
        if isinstance(output_list, torch.Tensor):
            output_list = [output_list]
        output_list = self.flatten_list(output_list)
        losses = []
        if weights is None:
            weights = []
            for o in output_list:
                no = torch.stack([torch.real(o), torch.imag(o)], dim=-1) if o.is_complex() else o
                w = np.random.randn(*no.shape)
                weights.append(w)
                losses.append(torch.sum(no * torch.from_numpy(w)))
        else:
            assert len(output_list) == len(weights)
            for o, w in zip(output_list, weights):
                no = torch.stack([torch.real(o), torch.imag(o)], dim=-1) if o.is_complex() else o
                assert w.shape == no.shape
                losses.append(torch.sum(no * torch.from_numpy(w)))
        output_list = [_torch_to_np(x) for x in output_list]
        return ninput_list, output_list, losses, weights

    def check_op_with_torch(self, jittor_op, torch_op, input_list, check_grad=True, jittor_knames=None, torch_knames=None, **kwargs):
        weights = None
        jittor_input, jittor_output, jittor_losses, weights = self.run_jittor_op(jittor_op, input_list, weights, key_names=jittor_knames, **kwargs)
        torch_input, torch_output, torch_losses, weights = self.run_torch_op(torch_op, input_list, weights, key_names=torch_knames, **kwargs)
        self.check_results(jittor_output, torch_output)

        if check_grad:
            jittor_grads = self.grad_jittor(jittor_input, jittor_losses)
            torch_grads = self.grad_torch(torch_input, torch_losses)
            self.check_results(jittor_grads, torch_grads)

    def check_op_with_numpy(self, jittor_op, numpy_op, input_list):
        _, jittor_output, _, _ = self.run_jittor_op(jittor_op, input_list, None)
        numpy_output = numpy_op(*input_list)
        if isinstance(numpy_output, np.ndarray):
            numpy_output = [numpy_output]

        self.check_results(jittor_output, numpy_output)

@unittest.skipIf(__skip_torch_test, "No Torch found")
class TestComplexLinalg(unittest.TestCase, TestResultAndGrad):
    def random_complex_matrix(self, shape):
        r = np.random.randn(*shape)
        i = np.random.randn(*shape)
        return r + 1j * i

    def test_complex_matmul(self):
        s1 = (50, 200)
        s2 = (200, 50)
        m1 = self.random_complex_matrix(s1)
        m2 = self.random_complex_matrix(s2)

        inputs = [m1, m2]
        self.check_op_with_torch(jt.matmul, torch.matmul, inputs)

    def test_complex_matmul_batch(self):
        s1 = (10, 50, 30)
        s2 = (10, 30, 40)
        m1 = self.random_complex_matrix(s1)
        m2 = self.random_complex_matrix(s2)

        inputs = [m1, m2]
        self.check_op_with_torch(jt.matmul, torch.matmul, inputs)

    def test_complex_inv(self):
        s1 = (200, 200)
        m1 = self.random_complex_matrix(s1)
        inputs = [m1]
        self.check_op_with_torch(jt.linalg.inv, torch.linalg.inv, inputs)

    def test_complex_inv_batch(self):
        s1 = (10, 50, 50)
        m1 = self.random_complex_matrix(s1)
        inputs = [m1]
        self.check_op_with_torch(jt.linalg.inv, torch.linalg.inv, inputs)

    def test_complex_eig(self):
        # Unstable
        s1 = (20, 20)
        m1 = self.random_complex_matrix(s1)
        inputs = [m1]
        self.check_op_with_numpy(jt.linalg.eig, np.linalg.eig, inputs)

    def test_complex_eig_batch(self):
        # Unstable
        s1 = (5, 10, 10)
        m1 = self.random_complex_matrix(s1)
        inputs = [m1]
        self.check_op_with_numpy(jt.linalg.eig, np.linalg.eig, inputs)

    def test_complex_qr(self):
        s1 = (50, 50)
        m1 = self.random_complex_matrix(s1)
        inputs = [m1]
        self.check_op_with_torch(jt.linalg.qr, torch.linalg.qr, inputs)

    def test_complex_qr_batch(self):
        s1 = (10, 20, 20)
        m1 = self.random_complex_matrix(s1)
        inputs = [m1]
        self.check_op_with_torch(jt.linalg.qr, torch.linalg.qr, inputs)

    def test_complex_svd(self):
        # Unstable
        s1 = (50, 50)
        m1 = self.random_complex_matrix(s1)
        inputs = [m1]
        self.check_op_with_numpy(jt.linalg.svd, np.linalg.svd, inputs)

    def test_complex_svd_batch(self):
        # Unstable
        s1 = (10, 20, 20)
        m1 = self.random_complex_matrix(s1)
        inputs = [m1]
        self.check_op_with_numpy(jt.linalg.svd, np.linalg.svd, inputs)

class TestTensordot(unittest.TestCase, TestResultAndGrad):
    def random_complex_matrix(self, shape):
        r = np.random.randn(*shape)
        i = np.random.randn(*shape)
        return r + 1j * i

    def random_real_matrix(self, shape):
        return np.random.randn(*shape)

    def test_complex_tensordot_numberdim(self):
        s1 = (3, 4, 5)
        s2 = (4, 5, 6)
        dims = 2
        m1 = self.random_complex_matrix(s1)
        m2 = self.random_complex_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(jt.nn.tensordot, torch.tensordot, inputs, dims = dims)

    def test_complex_tensordot_tupledim(self):
        s1 = (3, 5, 4, 6)
        s2 = (6, 4, 5, 3)
        dims = ([2, 1, 3], [1, 2, 0])
        m1 = self.random_complex_matrix(s1)
        m2 = self.random_complex_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(jt.nn.tensordot, torch.tensordot, inputs, dims = dims)

    def test_real_tensordot_numberdim(self):
        s1 = (3, 4, 5)
        s2 = (4, 5, 6)
        dims = 2
        m1 = self.random_real_matrix(s1)
        m2 = self.random_real_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(jt.nn.tensordot, torch.tensordot, inputs, dims = dims)

    def test_real_tensordot_tupledim(self):
        s1 = (3, 5, 4, 6)
        s2 = (6, 4, 5, 3)
        dims = ([2, 1, 3], [1, 2, 0])
        m1 = self.random_real_matrix(s1)
        m2 = self.random_real_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(jt.nn.tensordot, torch.tensordot, inputs, dims = dims)

class TestKron(unittest.TestCase, TestResultAndGrad):
    def random_complex_matrix(self, shape):
        r = np.random.randn(*shape)
        i = np.random.randn(*shape)
        return r + 1j * i

    def random_real_matrix(self, shape):
        return np.random.randn(*shape)

    def test_complex_firstlarge(self):
        s1 = (2, 3, 4)
        s2 = (5, 2)
        m1 = self.random_complex_matrix(s1)
        m2 = self.random_complex_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(jt.nn.kron, torch.kron, inputs)

    def test_complex_second_large(self):
        s1 = (2, 3)
        s2 = (5, 2, 4)
        m1 = self.random_complex_matrix(s1)
        m2 = self.random_complex_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(jt.nn.kron, torch.kron, inputs)

    def test_real_firstlarge(self):
        s1 = (2, 3, 4)
        s2 = (5, 2)
        m1 = self.random_real_matrix(s1)
        m2 = self.random_real_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(jt.nn.kron, torch.kron, inputs)

    def test_real_second_large(self):
        s1 = (2, 3)
        s2 = (5, 2, 4)
        m1 = self.random_real_matrix(s1)
        m2 = self.random_real_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(jt.nn.kron, torch.kron, inputs)

@unittest.skipIf(__skip_torch_test, "No Torch found")
class TestGradFunctional(unittest.TestCase, TestResultAndGrad):
    def random_complex_matrix(self, shape):
        r = np.random.randn(*shape)
        i = np.random.randn(*shape)
        return r + 1j * i

    def random_real_matrix(self, shape):
        return np.random.randn(*shape) * 0.0 + 1.0

    def test_real_jvp_exp(self):
        def exp_reducer(x):
            return x.exp().sum(dim=1)
        s1 = (5, 6)
        m1 = self.random_real_matrix(s1)
        m2 = self.random_real_matrix(s1)
        inputs = [m1, m2]
        self.check_op_with_torch(
            partial(jt.gradfunctional.jvp, func=exp_reducer, create_graph=True),
            partial(torch.autograd.functional.jvp, func=exp_reducer, create_graph=True),
            inputs,
            jittor_knames = ['inputs', 'v'],
            torch_knames = ['inputs', 'v'],
            check_grad=False)

    def test_complex_jvp_exp(self):
        def exp_reducer(x):
            return x.exp().sum(1)
        s1 = (5, 6)
        m1 = self.random_complex_matrix(s1)
        m2 = self.random_complex_matrix(s1)
        inputs = [m1, m2]
        self.check_op_with_torch(
            partial(jt.gradfunctional.jvp, func=exp_reducer, create_graph=True),
            partial(torch.autograd.functional.jvp, func=exp_reducer, create_graph=True),
            inputs,
            jittor_knames = ['inputs', 'v'],
            torch_knames = ['inputs', 'v'],
            check_grad=False,
            )

    def test_real_jvp_add(self):
        w1, w2 = np.random.rand(), np.random.rand()
        def adder(x, y):
            return w1 * x + w2 * y
        s1 = (5, 6)
        m1 = self.random_real_matrix(s1)
        m2 = self.random_real_matrix(s1)
        m3 = self.random_real_matrix(s1)
        m4 = self.random_real_matrix(s1)
        inputs = [(m1, m2), (m3, m4)]
        self.check_op_with_torch(
            partial(jt.gradfunctional.jvp, func=adder, create_graph=True),
            partial(torch.autograd.functional.jvp, func=adder, create_graph=True),
            inputs,
            jittor_knames = ['inputs', 'v'],
            torch_knames = ['inputs', 'v'],
            check_grad=False,
        )

    def test_complex_jvp_add(self):
        w1r, w1i = np.random.rand(), np.random.rand()
        w2r, w2i = np.random.rand(), np.random.rand()
        def adder_pt(x, y):
            return (w1r + 1j * w1i) * x + (w2r + 1j * w2i) * y
        def adder_jt(x, y):
            w1 = ComplexNumber(real=jt.array(w1r).reshape(1,1), imag = jt.array(w1i).reshape(1,1))
            w2 = ComplexNumber(real=jt.array(w2r).reshape(1,1), imag = jt.array(w2i).reshape(1,1))
            return w1 * x + w2 * y
        s1 = (5, 6)
        m1 = self.random_complex_matrix(s1)
        m2 = self.random_complex_matrix(s1)
        m3 = self.random_complex_matrix(s1)
        m4 = self.random_complex_matrix(s1)
        inputs = [(m1, m2), (m3, m4)]
        self.check_op_with_torch(
            partial(jt.gradfunctional.jvp, func=adder_jt, create_graph=True),
            partial(torch.autograd.functional.jvp, func=adder_pt, create_graph=True),
            inputs,
            jittor_knames = ['inputs', 'v'],
            torch_knames = ['inputs', 'v'],
            check_grad=False,
        )

    def test_real_vjp_exp(self):
        def exp_reducer(x):
            return x.exp().sum(dim=1)
        s1 = (5, 6)
        s2 = (5,)
        m1 = self.random_real_matrix(s1)
        m2 = self.random_real_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(
            partial(jt.gradfunctional.vjp, func=exp_reducer),
            partial(torch.autograd.functional.vjp, func=exp_reducer),
            inputs,
            jittor_knames = ['inputs', 'v'],
            torch_knames = ['inputs', 'v'],
            check_grad=False)

    def test_complex_vjp_exp(self):
        def exp_reducer(x):
            return x.exp().sum(1)
        s1 = (5, 6)
        s2 = (5,)
        m1 = self.random_real_matrix(s1)
        m2 = self.random_real_matrix(s2)
        inputs = [m1, m2]
        self.check_op_with_torch(
            partial(jt.gradfunctional.vjp, func=exp_reducer, create_graph=True),
            partial(torch.autograd.functional.vjp, func=exp_reducer, create_graph=True),
            inputs,
            jittor_knames = ['inputs', 'v'],
            torch_knames = ['inputs', 'v'],
            check_grad=False,
            )

    def test_real_vjp_add(self):
        w1, w2 = np.random.rand(), np.random.rand()
        def adder(x, y):
            return w1 * x + w2 * y
        s1 = (5, 6)
        m1 = self.random_real_matrix(s1)
        m2 = self.random_real_matrix(s1)
        m3 = self.random_real_matrix(s1)
        inputs = [(m1, m2), m3]
        self.check_op_with_torch(
            partial(jt.gradfunctional.vjp, func=adder, create_graph=True),
            partial(torch.autograd.functional.vjp, func=adder, create_graph=True),
            inputs,
            jittor_knames = ['inputs', 'v'],
            torch_knames = ['inputs', 'v'],
            check_grad=False,
        )

    def test_complex_vjp_add(self):
        w1r, w1i = np.random.rand(), np.random.rand()
        w2r, w2i = np.random.rand(), np.random.rand()
        def adder_pt(x, y):
            return (w1r + 1j * w1i) * x + (w2r + 1j * w2i) * y
        def adder_jt(x, y):
            w1 = ComplexNumber(real=jt.array(w1r).reshape(1,1), imag = jt.array(w1i).reshape(1,1))
            w2 = ComplexNumber(real=jt.array(w2r).reshape(1,1), imag = jt.array(w2i).reshape(1,1))
            return w1 * x + w2 * y
        s1 = (5, 6)
        m1 = self.random_complex_matrix(s1)
        m2 = self.random_complex_matrix(s1)
        m3 = self.random_complex_matrix(s1)
        inputs = [(m1, m2), (m3)]
        self.check_op_with_torch(
            partial(jt.gradfunctional.vjp, func=adder_jt, create_graph=True),
            partial(torch.autograd.functional.vjp, func=adder_pt, create_graph=True),
            inputs,
            jittor_knames = ['inputs', 'v'],
            torch_knames = ['inputs', 'v'],
            check_grad=False,
        )

if __name__ == "__main__":
    unittest.main()

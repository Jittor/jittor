# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import pytest
from jittor import Module
from jittor.models import resnet
import pickle
from PIL import Image
import platform

f32 = jt.float32

def matmul(a, b):
    (n, m), k = a.shape, b.shape[-1]
    a = a.broadcast([n,m,k], dims=[2])
    b = b.broadcast([n,m,k], dims=[0])
    return (a*b).sum(dim=1)


def relu(x):
    return jt.maximum(x, 0.0)
Relu = jt.make_module(relu)

class Model(Module):
    def __init__(self, input_size):
        self.linear1 = Linear(input_size, 10)
        self.relu1 = Relu()
        self.linear2 = Linear(10, 1)
    def execute(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        return self.linear2(x)

def print_stack_tree(data):
    tree = {}
    for n in data["node_data"].values():
        p = tree
        for s in n["stacks"]:
            name = s['name']
            if name not in p:
                p[name] = {}
            p = p[name]
    from pprint import pprint
    pprint(tree)

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.w = (jt.random((in_features, out_features))-0.5) / in_features**0.5
        self.b = jt.random((out_features,))-0.5 if bias else None
    def execute(self, x):
        x = matmul(x, self.w)
        if self.b is not None: 
            return x+self.b
        return x


class TestTraceVar(unittest.TestCase):
    def test_gradient_ready_callback_nodes_keep_callback_trace_context(self):
        jt.clear_trace_data()
        with jt.flag_scope(trace_py_var=2):
            x = jt.array(np.array([2.0, 3.0], dtype=np.float32)).stop_grad()
            weight = jt.array(np.array([4.0, 5.0], dtype=np.float32))
            loss = (x * weight).sum()
            callback_values = []

            def on_gradient_ready(index, gradient):
                self.assertEqual(index, 0)
                value = gradient + 1.0
                value.sync()
                callback_values.append(value)

            gradients = jt.core.grad_with_callback(
                loss, [weight], True, on_gradient_ready)
            jt.sync_all()
            trace = jt.dump_trace_data()

        self.assertEqual(len(gradients), 1)
        np.testing.assert_array_equal(gradients[0].numpy(), [2.0, 3.0])
        np.testing.assert_array_equal(callback_values[0].numpy(), [3.0, 4.0])
        callback_nodes = [
            node for node in trace["node_data"].values()
            if node.get("attrs", {}).get("name") == "binary.add"
            and any("test_trace_var.py" in stack.get("file_path", "")
                    for stack in node.get("stacks", []))
        ]
        self.assertTrue(callback_nodes)
        self.assertNotIn("grad_op_id", callback_nodes[-1]["attrs"])
        self.assertFalse(any(
            stack.get("name") == "Grad"
            for stack in callback_nodes[-1].get("stacks", [])))
        jt.clear_trace_data()

    def test_gradient_ready_callback_exception_restores_trace_context(self):
        jt.clear_trace_data()
        with jt.flag_scope(trace_py_var=2):
            x = jt.array(np.array([2.0, 3.0], dtype=np.float32)).stop_grad()
            weight = jt.array(np.array([4.0, 5.0], dtype=np.float32))
            loss = (x * weight).sum()

            def fail_callback(index, gradient):
                raise ValueError("callback failure probe")

            with self.assertRaises(ValueError):
                jt.core.grad_with_callback(loss, [weight], True, fail_callback)

            value = jt.array(np.array([7.0, 8.0], dtype=np.float32)) + 1.0
            value.sync()
            trace = jt.dump_trace_data()

        forward_nodes = [
            node for node in trace["node_data"].values()
            if node.get("attrs", {}).get("name") == "binary.add"
            and any("test_trace_var.py" in stack.get("file_path", "")
                    for stack in node.get("stacks", []))
        ]
        self.assertTrue(forward_nodes)
        self.assertNotIn("grad_op_id", forward_nodes[-1]["attrs"])
        self.assertFalse(any(
            stack.get("name") == "Grad"
            for stack in forward_nodes[-1].get("stacks", [])))
        jt.clear_trace_data()

    def test_gradient_ready_callback_non_none_return_is_a_catchable_user_error(self):
        x = jt.array(np.array([2.0, 3.0], dtype=np.float32)).stop_grad()
        weight = jt.array(np.array([4.0, 5.0], dtype=np.float32))
        loss = (x * weight).sum()

        def invalid_callback(index, gradient):
            return 1

        with self.assertRaisesRegex(
                RuntimeError, "gradient-ready callback must return None"):
            jt.core.grad_with_callback(loss, [weight], True, invalid_callback)

        # A rejected callback must leave the same runtime and weights usable.
        callback_values = []

        def valid_callback(index, gradient):
            callback_values.append((index, gradient))

        gradients = jt.core.grad_with_callback(
            (x * weight).sum(), [weight], True, valid_callback)
        self.assertEqual(len(gradients), 1)
        self.assertEqual(len(callback_values), 1)
        self.assertEqual(callback_values[0][0], 0)
        np.testing.assert_array_equal(gradients[0].numpy(), [2.0, 3.0])
        np.testing.assert_array_equal(callback_values[0][1].numpy(), [2.0, 3.0])

    def test_lifecycle_observer_is_registered_by_trace_flag(self):
        jt.clear_trace_data()
        with jt.flag_scope(trace_py_var=2):
            value = (jt.array([1.0]) + 2.0).sum()
            value.sync()
            data = jt.dump_trace_data()
        jt.clear_trace_data()

        self.assertTrue(data["node_data"])
        self.assertTrue(data["execute_op_info"])

    def test_simple_model(self):
        with jt.flag_scope(trace_py_var=2):

            model = Model(input_size=1)
            batch_size = 10
            x = jt.float32(np.random.rand(batch_size, 1))
            y = model(x)
            y.sync()


            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open(f"{jt.introspection.policy.startup.cache_path}/simple_model.pkl", "wb") as f:
                pickle.dump(data, f)

    def test_simple_model_train(self):
        with jt.flag_scope(trace_py_var=2):
            
            model = Model(input_size=1)
            opt = jt.optim.SGD(model.parameters(), 0.1)

            batch_size = 10
            x = jt.float32(np.random.rand(batch_size, 1))
            y = model(x)
            opt.step(y**2)
            jt.sync_all()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            # print_stack_tree(data)
            for k,v in data["execute_op_info"].items():
                for i in v['fused_ops']:
                    if i not in data["node_data"]:
                        assert 0, (i, "not found")

            for k,v in list(data["node_data"].items()):
                if v["attrs"]["name"] == "unname":
                    assert 0
            print(len(data["node_data"]))
            with open(f"{jt.introspection.policy.startup.cache_path}/simple_model_train.pkl", "wb") as f:
                pickle.dump(data, f)

    def test_resnet_infer(self):
        with jt.flag_scope(trace_py_var=2):

            resnet18 = resnet.Resnet18()
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)
            y.sync()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open(f"{jt.introspection.policy.startup.cache_path}/resnet.pkl", "wb") as f:
                pickle.dump(data, f)
            for k,v in data["execute_op_info"].items():
                for i in v['fused_ops']:
                    if i not in data["node_data"]:
                        assert 0, (i, "not found")

    @pytest.mark.network
    def test_resnet_infer_with_feature(self):
        cat_url = "https://ss1.bdstatic.com/70cFuXSh_Q1YnxGkpoWK1HF6hhy/it/u=3782485413,1118109468&fm=26&gp=0.jpg"
        import jittor_utils
        cat_path = f"{jt.introspection.policy.startup.cache_path}/cat.jpg"
        print("download")
        jittor_utils.download(cat_url, cat_path)
        with open(cat_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
            img = jt.array(np.array(img))
            print(img.shape, img.dtype)
            img = ((img.float() - 128) / 255).transpose(2,0,1)


        with jt.flag_scope(trace_py_var=2, trace_var_data=1):
            img = img[None,...]

            resnet18 = resnet.Resnet18(pretrained=True)
            x = jt.float32(img)
            y = resnet18(x)
            y.sync()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open(f"{jt.introspection.policy.startup.cache_path}/resnet_with_feature.pkl", "wb") as f:
                pickle.dump(data, f)
            for k,v in data["execute_op_info"].items():
                for i in v['fused_ops']:
                    if i not in data["node_data"]:
                        assert 0, (i, "not found")

    def test_resnet_trainx(self):
        with jt.flag_scope(trace_py_var=2):

            resnet18 = resnet.Resnet18()
            opt = jt.optim.SGD(resnet18.parameters(), 0.1)
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)

            opt.step(y**2)
            jt.sync_all()

            data = jt.dump_trace_data()
            jt.clear_trace_data()
            with open(f"{jt.introspection.policy.startup.cache_path}/resnet_train.pkl", "wb") as f:
                pickle.dump(data, f)
            for k,v in data["execute_op_info"].items():
                for i in v['fused_ops']:
                    if i not in data["node_data"]:
                        assert 0, (i, "not found")
            for k,v in data["node_data"].items():
                if 'name' not in v["attrs"]:
                    print(v)
                # assert 'name' in v["attrs"], v
                # for s in v["stacks"]:
                #     if "_opt" in s["name"] or "_model" in s["name"]:
                #         assert 0, v

    def test_resnet_train_profile(self):
        with jt.profile_scope(trace_py_var=1):

            resnet18 = resnet.Resnet18()
            opt = jt.optim.SGD(resnet18.parameters(), 0.1)
            x = jt.float32(np.random.rand(2, 3, 224, 224))
            y = resnet18(x)

            opt.step(y**2)
            jt.sync_all()


if __name__ == "__main__":
    unittest.main()

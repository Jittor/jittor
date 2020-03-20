# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import math
import numpy as np

try:
    jt.dirty_fix_pytorch_runtime_error()
    import torch
    from torch import nn
    from jittor.utils import pytorch_converter
except:
    torch = None

@unittest.skipIf(torch is None, "pytorch not found.")
class TestPytorchConverter(unittest.TestCase):
    def test_simple(self):
        def model(c):
            a = torch.Tensor([1,2,3,4,0])
            b = a+a
            b = b*2
            b = b[:2]
            a = a[1<a]
            return a[0]+b[0]+c[0]

        c = torch.Tensor([1,2,3])
        r1 = model(c)
        with pytorch_converter.trace_scope(["model"]):
            r2 = model(c)
        assert r1.numpy()==r2.numpy()
        r3 = model(c)
        assert r1.numpy() == r2.numpy() and r2.numpy() == r3.numpy(), (r1,r2,r3)
        ans = """root in:[] out:[]
    model in:[input_0] out:[out_11]
        inj_torch_Tensor___init__ in:[array_1] out:[] args:[array_1, [1, 2, 3, 4, 0]]
        inj_torch_Tensor___add__ in:[array_1, array_1] out:[out_2] args:[array_1, array_1]
        inj_torch_Tensor___mul__ in:[out_2] out:[out_3] args:[out_2, 2]
        inj_torch_Tensor___getitem__ in:[out_3] out:[out_4] args:[out_3, slice(None, 2, None)]
        inj_torch_Tensor___gt__ in:[array_1] out:[out_5] args:[array_1, 1]
        inj_torch_Tensor___getitem__ in:[array_1, out_5] out:[out_6] args:[array_1, out_5]
        inj_torch_Tensor___getitem__ in:[out_6] out:[out_7] args:[out_6, 0]
        inj_torch_Tensor___getitem__ in:[out_4] out:[out_8] args:[out_4, 0]
        inj_torch_Tensor___add__ in:[out_7, out_8] out:[out_9] args:[out_7, out_8]
        inj_torch_Tensor___getitem__ in:[input_0] out:[out_10] args:[input_0, 0]
        inj_torch_Tensor___add__ in:[out_9, out_10] out:[out_11] args:[out_9, out_10]
    model in:[input_0] out:[out_11] end
root in:[] out:[] end"""
        ct = pytorch_converter.call_tree
        assert str(ct) == ans
        code = ct.to_jt()
        lc = {}
        exec(code, globals(), lc)
        print(code)
        jt_model = lc["model"]
        assert jt_model(jt.array([1,2,3])).data == r1.numpy()

    def test_resnet(self):
        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
                super(Bottleneck, self).__init__()
                self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
                self.bn1 = nn.BatchNorm2d(planes)
                self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, dilation=dilation,
                                    padding=dilation, bias=False)
                self.bn2 = nn.BatchNorm2d(planes)
                self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
                self.bn3 = nn.BatchNorm2d(planes * 4)
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual
                out = self.relu(out)

                return out

        class ResNet(nn.Module):
            def __init__(self, block, layers=(3, 4, 23, 3)):
                self.inplanes = 64
                super(ResNet, self).__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                    bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                    elif isinstance(m, nn.BatchNorm2d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(planes * block.expansion),
                    )

                layers = [block(self.inplanes, planes, stride, downsample)]
                self.inplanes = planes * block.expansion
                for i in range(1, blocks):
                    layers.append(block(self.inplanes, planes, dilation=dilation))

                return nn.Sequential(*layers)

            def forward(self, x):
                x1 = self.conv1(x)
                x2 = self.bn1(x1)
                x2 = self.relu(x2)
                x2 = self.maxpool(x2)

                x2 = self.layer1(x2)
                x3 = self.layer2(x2)
                x3 = self.layer3(x3)
                x3 = self.layer4(x3)

                return x1, x2, x3

        return
        resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])
        x = torch.Tensor(np.random.rand(2, 3, 224, 224))
        with pytorch_converter.trace_scope():
            y = resnet50(x)

        ct = pytorch_converter.call_tree
        code = ct.to_jt()
        print(code)

    def test_convert_batchnorm(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.bn1 = nn.BatchNorm2d(64)
                self.bn2 = nn.BatchNorm2d(64)
                self.bn3 = nn.BatchNorm2d(64)
            def forward(self, x):
                y = self.bn1(x)
                z = self.bn2(x*x)
                x = self.bn3(y+z)
                return x
        model = TestModel()

        """ test_code:
            x = torch.Tensor(np.random.rand(16, 64, 15, 15).astype("float32"))
            jt_array = jt.array(x.numpy())
            jt_result = jt.nn.batch_norm(jt_array, is_train=False, eps=1e-05, momentum=0.1)
            torch_result = nn.BatchNorm2d(64)(x)
        """

        x = torch.Tensor(np.random.rand(16, 64, 15, 15).astype("float32"))
        with pytorch_converter.trace_scope():
            y = model(x)

        ct = pytorch_converter.call_tree
        ans = """root in:[] out:[]
    TestModel.forward in:[input_0] out:[out_27] args:{'self': TestModel(
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
), 'x': input_0}
        BatchNorm2d.forward in:[input_0] out:[out_7] args:{'self': BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 'input': input_0}
            functional.batch_norm in:[input_0, out_3, out_4, out_5, out_6] out:[out_7]
        BatchNorm2d.forward in:[input_0] out:[out_7] args:{'self': BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 'input': input_0} end
        inj_torch_Tensor___mul__ in:[input_0, input_0] out:[out_10] args:[input_0, input_0]
        BatchNorm2d.forward in:[out_10] out:[out_17] args:{'self': BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 'input': out_10}
            functional.batch_norm in:[out_10, out_13, out_14, out_15, out_16] out:[out_17]
        BatchNorm2d.forward in:[out_10] out:[out_17] args:{'self': BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 'input': out_10} end
        inj_torch_Tensor___add__ in:[out_7, out_17] out:[out_20] args:[out_7, out_17]
        BatchNorm2d.forward in:[out_20] out:[out_27] args:{'self': BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 'input': out_20}
            functional.batch_norm in:[out_20, out_23, out_24, out_25, out_26] out:[out_27]
        BatchNorm2d.forward in:[out_20] out:[out_27] args:{'self': BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), 'input': out_20} end
    TestModel.forward in:[input_0] out:[out_27] args:{'self': TestModel(
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (bn3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
), 'x': input_0} end
root in:[] out:[] end"""
        assert str(ct) == ans
        code = ct.to_jt()
        lc = {}
        exec(code, globals(), lc)
        print(code)
        
        jt_model = lc["TestModel"]
        assert (jt_model(jt.array(x.numpy())).data - y.detach().numpy()).mean() < 1e-5

    def test_convert_relu(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.rl1 = nn.ReLU(inplace=True)
                self.rl2 = nn.ReLU(inplace=True)
                self.rl3 = nn.ReLU(inplace=True)
            def forward(self, x):
                y = self.rl1(x)
                z = self.rl2(x*x)
                x = self.rl3(y+z)
                return x
        model = TestModel()

        """ test_code:
            x = torch.Tensor(np.random.rand(16, 64, 15, 15).astype("float32"))
            jt_array = jt.array(x.numpy())
            jt_result = jt.nn.batch_norm(jt_array, is_train=False, eps=1e-05, momentum=0.1)
            torch_result = nn.BatchNorm2d(64)(x)
        """

        x = torch.Tensor(np.random.rand(16, 3, 15, 15).astype("float32"))
        with pytorch_converter.trace_scope():
            y = model(x)

        ct = pytorch_converter.call_tree
        ans = """root in:[] out:[]
    TestModel.forward in:[input_0] out:[out_8] args:{'self': TestModel(
  (rl1): ReLU(inplace=True)
  (rl2): ReLU(inplace=True)
  (rl3): ReLU(inplace=True)
), 'x': input_0}
        ReLU.forward in:[input_0] out:[input_0] args:{'self': ReLU(inplace=True), 'input': input_0}
            functional.relu in:[input_0] out:[input_0]
        ReLU.forward in:[input_0] out:[input_0] args:{'self': ReLU(inplace=True), 'input': input_0} end
        inj_torch_Tensor___mul__ in:[input_0, input_0] out:[out_4] args:[input_0, input_0]
        ReLU.forward in:[out_4] out:[out_4] args:{'self': ReLU(inplace=True), 'input': out_4}
            functional.relu in:[out_4] out:[out_4]
        ReLU.forward in:[out_4] out:[out_4] args:{'self': ReLU(inplace=True), 'input': out_4} end
        inj_torch_Tensor___add__ in:[input_0, out_4] out:[out_8] args:[input_0, out_4]
        ReLU.forward in:[out_8] out:[out_8] args:{'self': ReLU(inplace=True), 'input': out_8}
            functional.relu in:[out_8] out:[out_8]
        ReLU.forward in:[out_8] out:[out_8] args:{'self': ReLU(inplace=True), 'input': out_8} end
    TestModel.forward in:[input_0] out:[out_8] args:{'self': TestModel(
  (rl1): ReLU(inplace=True)
  (rl2): ReLU(inplace=True)
  (rl3): ReLU(inplace=True)
), 'x': input_0} end
root in:[] out:[] end"""
        assert str(ct) == ans
        code = ct.to_jt()
        lc = {}
        exec(code, globals(), lc)
        print(code)
        
        jt_model = lc["TestModel"]
        assert (jt_model(jt.array(x.numpy())).data == y.detach().numpy()).all()

    def test_convert_pool(self):
        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
                self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
                self.mp3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
            def forward(self, x):
                y = self.mp1(x)
                z = self.mp2(x*x)
                x = self.mp3(y+z)
                return x
        model = TestModel()

        x = torch.Tensor(np.random.rand(16, 3, 15, 15).astype("float32"))
        with pytorch_converter.trace_scope():
            y = model(x)
        ct = pytorch_converter.call_tree
        ans = """root in:[] out:[]
    TestModel.forward in:[input_0] out:[out_11] args:{'self': TestModel(
  (mp1): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (mp2): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (mp3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
), 'x': input_0}
        MaxPool2d.forward in:[input_0] out:[out_1] args:{'self': MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False), 'input': input_0}
            functional._max_pool2d in:[input_0] out:[out_1]
        MaxPool2d.forward in:[input_0] out:[out_1] args:{'self': MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False), 'input': input_0} end
        inj_torch_Tensor___mul__ in:[input_0, input_0] out:[out_5] args:[input_0, input_0]
        MaxPool2d.forward in:[out_5] out:[out_6] args:{'self': MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False), 'input': out_5}
            functional._max_pool2d in:[out_5] out:[out_6]
        MaxPool2d.forward in:[out_5] out:[out_6] args:{'self': MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False), 'input': out_5} end
        inj_torch_Tensor___add__ in:[out_1, out_6] out:[out_10] args:[out_1, out_6]
        MaxPool2d.forward in:[out_10] out:[out_11] args:{'self': MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False), 'input': out_10}
            functional._max_pool2d in:[out_10] out:[out_11]
        MaxPool2d.forward in:[out_10] out:[out_11] args:{'self': MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False), 'input': out_10} end
    TestModel.forward in:[input_0] out:[out_11] args:{'self': TestModel(
  (mp1): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (mp2): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (mp3): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
), 'x': input_0} end
root in:[] out:[] end"""
        assert str(ct) == ans
        code = ct.to_jt()
        lc = {}
        exec(code, globals(), lc)
        print(code)
        
        jt_model = lc["TestModel"]
        assert (jt_model(jt.array(x.numpy())).data == y.detach().numpy()).all()


if __name__ == "__main__":
    unittest.main()

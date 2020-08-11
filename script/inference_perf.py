# ***************************************************************
# Copyright (c) 2020 Jittor. Authors: 
#     Wenyang Zhou <576825820@qq.com>
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import numpy as np
import jittor as jt
import torch
import time
import jittor.models as jtmodels
import torchvision.models as tcmodels
import os

jt.flags.use_cuda = 1
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
jt.cudnn.set_algorithm_cache_size(10000)

threshold = 1e-3

models = [
    # 'squeezenet1_0',
    'squeezenet1_1',
    'alexnet',
    # 'resnet18',
    # 'resnet34',
    'resnet50',
    # 'resnet101',
    'resnet152',
    'resnext50_32x4d',
    'resnext101_32x8d',
    'vgg11',
    # 'vgg11_bn',
    # 'vgg13',
    # 'vgg13_bn',
    # 'vgg16',
    # 'vgg16_bn',
    # 'vgg19',
    # 'vgg19_bn',
    'wide_resnet50_2',
    'wide_resnet101_2',
]

def to_cuda(x):
    if jt.has_cuda:
        return x.cuda()
    return x

def test_allmodels(bs=1):
    # Define numpy input image
    test_img = np.random.random((bs,3,224,224)).astype('float32')
    # Define pytorch & jittor input image
    pytorch_test_img = to_cuda(torch.Tensor(test_img))
    jittor_test_img = jt.array(test_img)
    for model in models:
        if model == "inception_v3":
            test_img = np.random.random((bs,3,300,300)).astype('float32')
            pytorch_test_img = to_cuda(torch.Tensor(test_img))
            jittor_test_img = jt.array(test_img)

        jittor_test_img.stop_grad()
        pytorch_test_img.requires_grad = False

        # Define pytorch & jittor model
        pytorch_model = to_cuda(tcmodels.__dict__[model]())
        jittor_model = jtmodels.__dict__[model]()
        # Set eval to avoid dropout layer
        pytorch_model.eval()
        jittor_model.eval()
        # Jittor loads pytorch parameters to ensure forward alignment
        jittor_model.load_parameters(pytorch_model.state_dict())

        total = 512
        warmup = max(2, total // bs // 8)
        rerun = max(2, total // bs)

        print("=" * 20 + model + "=" * 20)

        # Jittor warms up
        for i in range(warmup):
            jittor_result = jittor_model(jittor_test_img)
        jt.sync_all(True)
        # Test jittor and once forward time
        sta = time.time()
        for i in range(rerun):
            jittor_result = jittor_model(jittor_test_img)
            jittor_result.sync()
        jt.sync_all(True)
        end = time.time()
        print(f"- Jittor {model} forward average time cost: {round((time.time() - sta) / rerun,5)}, Batch Size: {bs}, FPS: {round(bs * rerun / (end - sta),2)}")

        # pytorch warmup
        for i in range(warmup):
            pytorch_result = pytorch_model(pytorch_test_img)
        # Test pytorch and once forward time
        torch.cuda.synchronize()
        sta = time.time()
        for i in range(rerun):
            pytorch_result = pytorch_model(pytorch_test_img)
        torch.cuda.synchronize()
        end = time.time()
        print(f"- Pytorch {model} forward average time cost: {round((end - sta) / rerun,5)}, Batch Size: {bs}, FPS: {round(bs * rerun / (end - sta),2)}")

        # Judge pytorch & jittor forward relative error. If the differece is lower than threshold, this test passes.
        x = pytorch_result.detach().cpu().numpy() + 1
        y = jittor_result.numpy() + 1
        relative_error = abs(x - y) / abs(y)
        diff = relative_error.mean()
        assert diff < threshold, f"[*] {model} forward fails..., Relative Error: {diff}"
        print(f"[*] {model} forword passes with Relative Error {diff}")
        torch.cuda.empty_cache()
        jt.clean()
        jt.gc()
        

with torch.no_grad():
    for bs in [1,2,4,8,16,32,64,128]:
    # for bs in [128]:
        test_allmodels(bs)
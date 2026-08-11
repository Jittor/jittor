# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved. 
#     Dun Liang <randonlang@gmail.com>. 
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import jittor as jt
import unittest
import os
import numpy as np
import random
from _helpers.torch_runtime import import_torch_modules, modules_available

mid = 0
if hasattr(os, "uname") and os.uname()[1] == "jittor-ce":
    mid = 1
traindir = ["/data1/cjld/imagenet/train/", "/home/cjld/imagenet/train/"][mid]
check_num_batch = 5
pass_this_test = not modules_available("torch", "torchvision") or not os.path.isdir(
    traindir
)
msg = "optional Torch runtime or ImageNet fixture is unavailable"
torch = None
train_dataset = None


def setUpModule():
    global torch, train_dataset
    if pass_this_test:
        return
    torch, datasets, transforms = import_torch_modules(
        "torch", "torchvision.datasets", "torchvision.transforms"
    )
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

@unittest.skipIf(pass_this_test, f"can not run imagenet dataset test: {msg}")
class TestImageFolder(unittest.TestCase):
    def test_imagenet(self):
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=256, shuffle=False)

        random.seed(0)
        tc_data = []
        for i, data in enumerate(train_loader):
            tc_data.append(data)
            print("get", data[0].shape)
            if i==check_num_batch: break

        from jittor.dataset.dataset import ImageFolder
        import jittor.transform as transform

        dataset = ImageFolder(traindir).set_attrs(batch_size=256, shuffle=False)

        dataset.set_attrs(transform = transform.Compose([
            transform.RandomCropAndResize(224),
            transform.RandomHorizontalFlip(),
            transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))

        random.seed(0)

        for i, (images, labels) in enumerate(dataset):
            print("compare", i)
            assert np.allclose(images.numpy(), tc_data[i][0].numpy())
            assert np.allclose(labels.numpy(), tc_data[i][1].numpy())
            if i==check_num_batch: break

if __name__ == "__main__":
    unittest.main()

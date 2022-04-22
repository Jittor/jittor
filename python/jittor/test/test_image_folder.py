# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
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

pass_this_test = False
msg = ""
mid = 0
if hasattr(os, "uname") and os.uname()[1] == "jittor-ce":
    mid = 1
try:
    # check can we run this test
    # test code
    jt.dirty_fix_pytorch_runtime_error()
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import torch

    traindir = ["/data1/cjld/imagenet/train/","/home/cjld/imagenet/train/"][mid]
    check_num_batch = 5
    assert os.path.isdir(traindir)

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
except Exception as e:
    pass_this_test = True
    msg = str(e)

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

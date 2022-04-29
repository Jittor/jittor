# ***************************************************************
# Copyright (c) 2022 Jittor.
# All Rights Reserved. 
# Maintainers:
#     Dun Liang <randonlang@gmail.com>. 
#
# Contributors:
#     Xin Yao <yaox12@outlook.com>
# 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************

import unittest
import random
from PIL import Image
import numpy as np
from numpy.testing import assert_array_almost_equal
import jittor as jt
import jittor.transform as transform

try:
    from scipy import stats
except ImportError:
    stats = None


class Tester(unittest.TestCase):

    def test_crop(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2

        img = np.ones([height, width, 3])
        oh1 = (height - oheight) // 2
        ow1 = (width - owidth) // 2
        # imgnarrow = img[oh1:oh1 + oheight, ow1:ow1 + owidth, :]
        # imgnarrow.fill(0)
        img[oh1:oh1 + oheight, ow1:ow1 + owidth, :] = 0
        # img = jt.array(img)
        result = transform.Compose([
            transform.ToPILImage(),
            transform.CenterCrop((oheight, owidth)),
            transform.ToTensor(),
        ])(img)
        self.assertEqual(result.sum(), 0,
                         f"height: {height} width: {width} oheight: {oheight} owdith: {owidth}")
        oheight += 1
        owidth += 1
        result = transform.Compose([
            transform.ToPILImage(),
            transform.CenterCrop((oheight, owidth)),
            transform.ToTensor(),
        ])(img)
        sum1 = result.sum()
        # TODO: not pass
        # self.assertGreater(sum1, 1,
        #                    f"height: {height} width: {width} oheight: {oheight} owdith: {owidth}")
        oheight += 1
        owidth += 1
        result = transform.Compose([
            transform.ToPILImage(),
            transform.CenterCrop((oheight, owidth)),
            transform.ToTensor(),
        ])(img)
        sum2 = result.sum()
        self.assertGreater(sum2, 0,
                           f"height: {height} width: {width} oheight: {oheight} owdith: {owidth}")
        self.assertGreaterEqual(sum2, sum1,
                           f"height: {height} width: {width} oheight: {oheight} owdith: {owidth}")

    def test_resize(self):
        height = random.randint(24, 32) * 2
        width = random.randint(24, 32) * 2
        osize = random.randint(5, 12) * 2

        img = jt.ones([height, width, 3])
        result = transform.Compose([
            transform.ToPILImage(),
            transform.Resize(osize),
            transform.ToTensor(),
        ])(img)
        self.assertIn(osize, result.shape)
        if height < width:
            self.assertLessEqual(result.shape[1], result.shape[2])
        elif width < height:
            self.assertGreaterEqual(result.shape[1], result.shape[2])

        result = transform.Compose([
            transform.ToPILImage(),
            transform.Resize([osize, osize]),
            transform.ToTensor(),
        ])(img)
        self.assertIn(osize, result.shape)
        self.assertEqual(result.shape[1], osize)
        self.assertEqual(result.shape[2], osize)

        oheight = random.randint(5, 12) * 2
        owidth = random.randint(5, 12) * 2
        result = transform.Compose([
            transform.ToPILImage(),
            transform.Resize((oheight, owidth)),
            transform.ToTensor(),
        ])(img)
        self.assertEqual(result.shape[1], oheight)
        self.assertEqual(result.shape[2], owidth)

        result = transform.Compose([
            transform.ToPILImage(),
            transform.Resize([oheight, owidth]),
            transform.ToTensor(),
        ])(img)
        self.assertEqual(result.shape[1], oheight)
        self.assertEqual(result.shape[2], owidth)

    def test_random_crop(self):
        height = random.randint(10, 32) * 2
        width = random.randint(10, 32) * 2
        oheight = random.randint(5, (height - 2) / 2) * 2
        owidth = random.randint(5, (width - 2) / 2) * 2
        img = np.ones((height, width, 3))
        result = transform.Compose([
            transform.ToPILImage(),
            transform.RandomCrop((oheight, owidth)),
            transform.ToTensor(),
        ])(img)
        self.assertEqual(result.shape[1], oheight)
        self.assertEqual(result.shape[2], owidth)

        result = transform.Compose([
            transform.ToPILImage(),
            transform.RandomCrop((oheight, owidth)),
            transform.ToTensor(),
        ])(img)
        self.assertEqual(result.shape[1], oheight)
        self.assertEqual(result.shape[2], owidth)

        result = transform.Compose([
            transform.ToPILImage(),
            transform.RandomCrop((height, width)),
            transform.ToTensor()
        ])(img)
        self.assertEqual(result.shape[1], height)
        self.assertEqual(result.shape[2], width)
        self.assertTrue(np.allclose(img, result.transpose(1,2,0)))

        with self.assertRaises(AssertionError):
            result = transform.Compose([
                transform.ToPILImage(),
                transform.RandomCrop((height + 1, width + 1)),
                transform.ToTensor(),
            ])(img)

    def test_lambda(self):
        trans = transform.Lambda(lambda x: x.add(10))
        x = jt.random([10])
        y = trans(x)
        self.assertTrue(np.allclose(y.data, jt.add(x, 10).data))

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_apply(self):
        random_state = random.getstate()
        random.seed(42)
        random_apply_transform = transform.RandomApply(
            [
                transform.RandomHorizontalFlip(),
                transform.RandomVerticalFlip(),
            ], p=0.4
        )
        img = transform.ToPILImage()(jt.random((3, 10, 10)))
        num_samples = 250
        num_applies = 0
        for _ in range(num_samples):
            out = random_apply_transform(img)
            if out != img:
                num_applies += 1

        p_value = stats.binom_test(num_applies, num_samples, p=0.3)
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_choice(self):
        random_state = random.getstate()
        random.seed(42)
        random_choice_transform = transform.RandomChoice(
            [
                transform.Resize(15),
                transform.Resize(20),
                transform.CenterCrop(10)
            ]
        )
        img = transform.ToPILImage()(jt.random((25, 25, 3)))
        num_samples = 250
        num_resize_15 = 0
        num_resize_20 = 0
        num_crop_10 = 0
        for _ in range(num_samples):
            out = random_choice_transform(img)
            if out.size == (15, 15):
                num_resize_15 += 1
            elif out.size == (20, 20):
                num_resize_20 += 1
            elif out.size == (10, 10):
                num_crop_10 += 1

        p_value = stats.binom_test(num_resize_15, num_samples, p=0.33333)
        self.assertGreater(p_value, 0.0001)
        p_value = stats.binom_test(num_resize_20, num_samples, p=0.33333)
        self.assertGreater(p_value, 0.0001)
        p_value = stats.binom_test(num_crop_10, num_samples, p=0.33333)
        self.assertGreater(p_value, 0.0001)

        random.setstate(random_state)

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_order(self):
        random_state = random.getstate()
        random.seed(42)
        random_order_transform = transform.RandomOrder(
            [
                transform.Resize(20),
                transform.CenterCrop(10)
            ]
        )
        img = transform.ToPILImage()(jt.random((3, 25, 25)))
        num_samples = 250
        num_normal_order = 0
        resize_crop_out = transform.CenterCrop(10)(transform.Resize(20)(img))
        for _ in range(num_samples):
            out = random_order_transform(img)
            if out == resize_crop_out:
                num_normal_order += 1

        p_value = stats.binom_test(num_normal_order, num_samples, p=0.5)
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

    def test_to_tensor(self):
        test_channels = [1, 3, 4]
        height, width = 4, 4
        trans = transform.ToTensor()

        with self.assertRaises(TypeError):
            trans(np.random.rand(1, height, width).tolist())

        with self.assertRaises(ValueError):
            trans(np.random.rand(height))
            trans(np.random.rand(1, 1, height, width))

        for channels in test_channels:
            input_data = np.random.randint(low=0, high=255, size=(height, width, channels)).astype(np.float32) / np.float32(255.0)
            img = transform.ToPILImage()(input_data)
            output = trans(img)
            expect = input_data.transpose(2,0,1)
            self.assertTrue(np.allclose(expect, output), f"{expect.shape}\n{output.shape}")

            ndarray = np.random.randint(low=0, high=255, size=(channels, height, width)).astype(np.uint8)
            output = trans(ndarray)
            expected_output = ndarray / 255.0
            np.testing.assert_allclose(output, expected_output)

            ndarray = np.random.rand(channels, height, width).astype(np.float32)
            output = trans(ndarray)
            expected_output = ndarray
            self.assertTrue(np.allclose(output, expected_output))

        # separate test for mode '1' PIL images
        input_data = np.random.binomial(1, 0.5, size=(height, width, 1)).astype(np.uint8)
        img = transform.ToPILImage()(input_data * 255).convert('1')
        output = trans(img)
        self.assertTrue(np.allclose(input_data[:,:,0], output[0]), f"{input_data.shape}\n{output.shape}")

    def test_1_channel_tensor_to_pil_image(self):
        to_tensor = transform.ToTensor()
        shape = (4, 4, 1)

        img_data_float = jt.array(np.random.rand(*shape), dtype='float32')
        img_data_byte = jt.array(np.random.randint(0, 255, shape), dtype='uint8')
        img_data_short = jt.array(np.random.randint(0, 32767, shape), dtype='int16')
        img_data_int = jt.array(np.random.randint(0, 2147483647, shape), dtype='int32')

        inputs = [img_data_float, img_data_byte, img_data_short, img_data_int]
        expected_outputs = [img_data_float.multiply(255).int().float().divide(255).numpy(),
                            img_data_byte.float().divide(255.0).numpy(),
                            img_data_short.numpy(),
                            img_data_int.numpy()]
        expected_modes = ['F', 'L', 'I;16', 'I']

        for img_data, expected_output, mode in zip(inputs, expected_outputs, expected_modes):
            for t in [transform.ToPILImage(), transform.ToPILImage(mode=mode)]:
                img = t(img_data)
                self.assertEqual(img.mode, mode)
                np.testing.assert_allclose(expected_output[:,:,0], to_tensor(img)[0], atol=0.01)
        # 'F' mode for torch.FloatTensor
        img_F_mode = transform.ToPILImage(mode='F')(img_data_float)
        self.assertEqual(img_F_mode.mode, 'F')

    def test_1_channel_ndarray_to_pil_image(self):
        img_data_float = np.random.rand(4, 4, 1).astype(np.float32)
        img_data_byte = np.random.randint(0, 255, (4, 4, 1)).astype(np.uint8)
        img_data_short = np.random.randint(0, 32767, (4, 4, 1)).astype(np.int16)
        img_data_int = np.random.randint(0, 2147483647, (4, 4, 1)).astype(np.int32)

        inputs = [img_data_float, img_data_byte, img_data_short, img_data_int]
        expected_modes = ['F', 'L', 'I;16', 'I']
        for img_data, mode in zip(inputs, expected_modes):
            for t in [transform.ToPILImage(), transform.ToPILImage(mode=mode)]:
                img = t(img_data)
                self.assertEqual(img.mode, mode)
                self.assertTrue(np.allclose(img_data[:, :, 0], img))

    def test_2_channel_ndarray_to_pil_image(self):
        def verify_img_data(img_data, mode):
            if mode is None:
                img = transform.ToPILImage()(img_data)
                self.assertEqual(img.mode, 'LA')  # default should assume LA
            else:
                img = transform.ToPILImage(mode=mode)(img_data)
                self.assertEqual(img.mode, mode)
            split = img.split()
            for i in range(2):
                self.assertTrue(np.allclose(img_data[:, :, i], split[i]))

        img_data = np.random.randint(0, 255, (4, 4, 2)).astype(np.uint8)
        for mode in [None, 'LA']:
            verify_img_data(img_data, mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 4 or 1 or 3 channel images
            transform.ToPILImage(mode='RGBA')(img_data)
            transform.ToPILImage(mode='P')(img_data)
            transform.ToPILImage(mode='RGB')(img_data)

    def test_2_channel_tensor_to_pil_image(self):
        def verify_img_data(img_data, expected_output, mode):
            if mode is None:
                img = transform.ToPILImage()(img_data)
                self.assertEqual(img.mode, 'LA')  # default should assume LA
            else:
                img = transform.ToPILImage(mode=mode)(img_data)
                self.assertEqual(img.mode, mode)
            split = img.split()
            for i in range(2):
                self.assertTrue(np.allclose(expected_output[:,:,i], transform.to_tensor(split[i])))

        img_data = jt.random((4, 4, 2))
        expected_output = img_data.multiply(255).int().float().divide(255)
        for mode in [None, 'LA']:
            verify_img_data(img_data, expected_output, mode=mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 4 or 1 or 3 channel images
            transform.ToPILImage(mode='RGBA')(img_data)
            transform.ToPILImage(mode='P')(img_data)
            transform.ToPILImage(mode='RGB')(img_data)

    def test_3_channel_tensor_to_pil_image(self):
        def verify_img_data(img_data, expected_output, mode):
            if mode is None:
                img = transform.ToPILImage()(img_data)
                self.assertEqual(img.mode, 'RGB')  # default should assume RGB
            else:
                img = transform.ToPILImage(mode=mode)(img_data)
                self.assertEqual(img.mode, mode)
            split = img.split()
            for i in range(3):
                self.assertTrue(np.allclose(expected_output[:,:,i], transform.to_tensor(split[i])))

        img_data = jt.random((4, 4, 3))
        expected_output = img_data.multiply(255).int().float().divide(255)
        for mode in [None, 'RGB', 'HSV', 'YCbCr']:
            verify_img_data(img_data, expected_output, mode=mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 4 or 1 or 2 channel images
            transform.ToPILImage(mode='RGBA')(img_data)
            transform.ToPILImage(mode='P')(img_data)
            transform.ToPILImage(mode='LA')(img_data)

        with self.assertRaises(ValueError):
            transform.ToPILImage()(jt.random((1, 3, 4, 4)))

    def test_3_channel_ndarray_to_pil_image(self):
        def verify_img_data(img_data, mode):
            if mode is None:
                img = transform.ToPILImage()(img_data)
                self.assertEqual(img.mode, 'RGB')  # default should assume RGB
            else:
                img = transform.ToPILImage(mode=mode)(img_data)
                self.assertEqual(img.mode, mode)
            split = img.split()
            for i in range(3):
                self.assertTrue(np.allclose(img_data[:, :, i], split[i]))

        img_data = np.random.randint(0, 255, (4, 4, 3)).astype(np.uint8)
        for mode in [None, 'RGB', 'HSV', 'YCbCr']:
            verify_img_data(img_data, mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 4 or 1 or 2 channel images
            transform.ToPILImage(mode='RGBA')(img_data)
            transform.ToPILImage(mode='P')(img_data)
            transform.ToPILImage(mode='LA')(img_data)

    def test_4_channel_tensor_to_pil_image(self):
        def verify_img_data(img_data, expected_output, mode):
            if mode is None:
                img = transform.ToPILImage()(img_data)
                self.assertEqual(img.mode, 'RGBA')  # default should assume RGBA
            else:
                img = transform.ToPILImage(mode=mode)(img_data)
                self.assertEqual(img.mode, mode)

            split = img.split()
            for i in range(4):
                self.assertTrue(np.allclose(expected_output[:,:,i], transform.to_tensor(split[i])[0]))

        img_data = jt.random((4, 4, 4))
        expected_output = img_data.multiply(255).int().float().divide(255)
        for mode in [None, 'RGBA', 'CMYK', 'RGBX']:
            verify_img_data(img_data, expected_output, mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 3 or 1 or 2 channel images
            transform.ToPILImage(mode='RGB')(img_data)
            transform.ToPILImage(mode='P')(img_data)
            transform.ToPILImage(mode='LA')(img_data)

    def test_4_channel_ndarray_to_pil_image(self):
        def verify_img_data(img_data, mode):
            if mode is None:
                img = transform.ToPILImage()(img_data)
                self.assertEqual(img.mode, 'RGBA')  # default should assume RGBA
            else:
                img = transform.ToPILImage(mode=mode)(img_data)
                self.assertEqual(img.mode, mode)
            split = img.split()
            for i in range(4):
                self.assertTrue(np.allclose(img_data[:, :, i], split[i]))

        img_data = np.random.randint(0, 255, (4, 4, 4)).astype(np.uint8)
        for mode in [None, 'RGBA', 'CMYK', 'RGBX']:
            verify_img_data(img_data, mode)

        with self.assertRaises(ValueError):
            # should raise if we try a mode for 3 or 1 or 2 channel images
            transform.ToPILImage(mode='RGB')(img_data)
            transform.ToPILImage(mode='P')(img_data)
            transform.ToPILImage(mode='LA')(img_data)

    def test_2d_tensor_to_pil_image(self):
        to_tensor = transform.ToTensor()

        img_data_float = jt.array(np.random.rand(4, 4), dtype='float32')
        img_data_byte = jt.array(np.random.randint(0, 255, (4, 4)), dtype='uint8')
        img_data_short = jt.array(np.random.randint(0, 32767, (4, 4)), dtype='int16')
        img_data_int = jt.array(np.random.randint(0, 2147483647, (4, 4)), dtype='int32')

        inputs = [img_data_float, img_data_byte, img_data_short, img_data_int]
        expected_outputs = [img_data_float.multiply(255).int().float().divide(255).numpy(),
                            img_data_byte.float().divide(255.0).numpy(),
                            img_data_short.numpy(),
                            img_data_int.numpy()]
        expected_modes = ['F', 'L', 'I;16', 'I']

        for img_data, expected_output, mode in zip(inputs, expected_outputs, expected_modes):
            for t in [transform.ToPILImage(), transform.ToPILImage(mode=mode)]:
                img = t(img_data)
                self.assertEqual(img.mode, mode)
                self.assertTrue(np.allclose(expected_output, to_tensor(img), atol=0.01, rtol=0.01))

    def test_2d_ndarray_to_pil_image(self):
        img_data_float = np.random.rand(4, 4).astype(np.float32)
        img_data_byte = np.random.randint(0, 255, (4, 4)).astype(np.uint8)
        img_data_short = np.random.randint(0, 32767, (4, 4)).astype(np.int16)
        img_data_int = np.random.randint(0, 2147483647, (4, 4)).astype(np.int32)

        inputs = [img_data_float, img_data_byte, img_data_short, img_data_int]
        expected_modes = ['F', 'L', 'I;16', 'I']
        for img_data, mode in zip(inputs, expected_modes):
            for t in [transform.ToPILImage(), transform.ToPILImage(mode=mode)]:
                img = t(img_data)
                self.assertEqual(img.mode, mode)
                self.assertTrue(np.allclose(img_data, img))

    def test_tensor_bad_types_to_pil_image(self):
        with self.assertRaises(ValueError):
            transform.ToPILImage()(jt.ones((1, 3, 4, 4)))

    def test_ndarray_bad_types_to_pil_image(self):
        trans = transform.ToPILImage()
        with self.assertRaises(TypeError):
            trans(np.ones([4, 4, 1], np.int64))
            trans(np.ones([4, 4, 1], np.uint16))
            trans(np.ones([4, 4, 1], np.uint32))
            trans(np.ones([4, 4, 1], np.float64))

        with self.assertRaises(ValueError):
            transform.ToPILImage()(np.ones([1, 4, 4, 3]))

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_vertical_flip(self):
        random_state = random.getstate()
        random.seed(42)
        img = transform.ToPILImage()(jt.random((3, 10, 10)))
        vimg = img.transpose(Image.FLIP_TOP_BOTTOM)

        num_samples = 250
        num_vertical = 0
        for _ in range(num_samples):
            out = transform.RandomVerticalFlip()(img)
            if out == vimg:
                num_vertical += 1

        p_value = stats.binom_test(num_vertical, num_samples, p=0.5)
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

        num_samples = 250
        num_vertical = 0
        for _ in range(num_samples):
            out = transform.RandomVerticalFlip(p=0.7)(img)
            if out == vimg:
                num_vertical += 1

        p_value = stats.binom_test(num_vertical, num_samples, p=0.7)
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_horizontal_flip(self):
        random_state = random.getstate()
        random.seed(42)
        img = transform.ToPILImage()(jt.random((3, 10, 10)))
        himg = img.transpose(Image.FLIP_LEFT_RIGHT)

        num_samples = 250
        num_horizontal = 0
        for _ in range(num_samples):
            out = transform.RandomHorizontalFlip()(img)
            if out == himg:
                num_horizontal += 1

        p_value = stats.binom_test(num_horizontal, num_samples, p=0.5)
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

        num_samples = 250
        num_horizontal = 0
        for _ in range(num_samples):
            out = transform.RandomHorizontalFlip(p=0.7)(img)
            if out == himg:
                num_horizontal += 1

        p_value = stats.binom_test(num_horizontal, num_samples, p=0.7)
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

    @unittest.skipIf(stats is None, 'scipy.stats is not available')
    def test_normalize(self):
        def samples_from_standard_normal(tensor):
            p_value = stats.kstest(list(tensor.reshape(-1).data), 'norm', args=(0, 1)).pvalue
            return p_value > 0.0001

        random_state = random.getstate()
        random.seed(42)
        for channels in [1, 3]:
            img = jt.random((channels, 10, 10))
            mean = [img[c].mean().item() for c in range(channels)]
            std = [img[c].std().item() for c in range(channels)]
            normalized = transform.ImageNormalize(mean, std)(img)
            self.assertTrue(samples_from_standard_normal(normalized))
        random.setstate(random_state)

    def test_normalize_different_dtype(self):
        for dtype1 in ['float32', 'float64']:
            img = jt.random((3, 10, 10), dtype=dtype1)
            for dtype2 in ['int64', 'float32', 'float64']:
                mean = jt.array([1, 2, 3], dtype=dtype2)
                std = jt.array([1, 2, 1], dtype=dtype2)
                # checks that it doesn't crash
                transform.image_normalize(img, mean, std)

    def test_normalize_3d_tensor(self):
        jt.seed(28)
        n_channels = 3
        img_size = 10
        mean = jt.random((n_channels,)).data
        std = jt.random((n_channels,)).data
        img = jt.random((n_channels, img_size, img_size)).data
        target = transform.image_normalize(img, mean, std)

        mean_unsqueezed = mean.reshape(-1, 1, 1)
        std_unsqueezed = std.reshape(-1, 1, 1)
        result1 = transform.image_normalize(img, mean_unsqueezed, std_unsqueezed)
        result2 = transform.image_normalize(img,
            mean_unsqueezed,
            std_unsqueezed)
        assert_array_almost_equal(target, result1)
        assert_array_almost_equal(target, result2)

    def test_adjust_brightness(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = transform.adjust_brightness(x_pil, 1)
        y_np = np.array(y_pil)
        self.assertTrue(np.allclose(y_np, x_np))

        # test 1
        y_pil = transform.adjust_brightness(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [0, 2, 6, 27, 67, 113, 18, 4, 117, 45, 127, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

        # test 2
        y_pil = transform.adjust_brightness(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 10, 26, 108, 255, 255, 74, 16, 255, 180, 255, 2]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

    def test_adjust_contrast(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = transform.adjust_contrast(x_pil, 1)
        y_np = np.array(y_pil)
        self.assertTrue(np.allclose(y_np, x_np))

        # test 1
        y_pil = transform.adjust_contrast(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [43, 45, 49, 70, 110, 156, 61, 47, 160, 88, 170, 43]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

        # test 2
        y_pil = transform.adjust_contrast(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 0, 0, 22, 184, 255, 0, 0, 255, 94, 255, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

    # @unittest.skipIf(Image.__version__ >= '7', "Temporarily disabled")
    def test_adjust_saturation(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = transform.adjust_saturation(x_pil, 1)
        y_np = np.array(y_pil)
        self.assertTrue(np.allclose(y_np, x_np))

        # test 1
        y_pil = transform.adjust_saturation(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [2, 4, 8, 87, 128, 173, 39, 25, 138, 133, 216, 89]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

        # test 2
        y_pil = transform.adjust_saturation(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 6, 22, 0, 149, 255, 32, 0, 255, 3, 255, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

    def test_adjust_hue(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        with self.assertRaises(ValueError):
            transform.adjust_hue(x_pil, -0.7)
            transform.adjust_hue(x_pil, 1)

        # test 0: almost same as x_data but not exact.
        # probably because hsv <-> rgb floating point ops
        y_pil = transform.adjust_hue(x_pil, 0)
        y_np = np.array(y_pil)
        y_ans = [0, 5, 13, 54, 139, 226, 35, 8, 234, 91, 255, 1]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

        # test 1
        y_pil = transform.adjust_hue(x_pil, 0.25)
        y_np = np.array(y_pil)
        y_ans = [13, 0, 12, 224, 54, 226, 234, 8, 99, 1, 222, 255]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

        # test 2
        y_pil = transform.adjust_hue(x_pil, -0.25)
        y_np = np.array(y_pil)
        y_ans = [0, 13, 2, 54, 226, 58, 8, 234, 152, 255, 43, 1]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

    def test_adjust_gamma(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')

        # test 0
        y_pil = transform.adjust_gamma(x_pil, 1)
        y_np = np.array(y_pil)
        self.assertTrue(np.allclose(y_np, x_np))

        # test 1
        y_pil = transform.adjust_gamma(x_pil, 0.5)
        y_np = np.array(y_pil)
        y_ans = [0, 35, 57, 117, 186, 241, 97, 45, 245, 152, 255, 16]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

        # test 2
        y_pil = transform.adjust_gamma(x_pil, 2)
        y_np = np.array(y_pil)
        y_ans = [0, 0, 0, 11, 71, 201, 5, 0, 215, 31, 255, 0]
        y_ans = np.array(y_ans, dtype=np.uint8).reshape(x_shape)
        self.assertTrue(np.allclose(y_np, y_ans))

    def test_adjusts_L_mode(self):
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_rgb = Image.fromarray(x_np, mode='RGB')

        x_l = x_rgb.convert('L')
        self.assertEqual(transform.adjust_brightness(x_l, 2).mode, 'L')
        self.assertEqual(transform.adjust_saturation(x_l, 2).mode, 'L')
        self.assertEqual(transform.adjust_contrast(x_l, 2).mode, 'L')
        self.assertEqual(transform.adjust_hue(x_l, 0.4).mode, 'L')
        self.assertEqual(transform.adjust_gamma(x_l, 0.5).mode, 'L')

    def test_color_jitter(self):
        color_jitter = transform.ColorJitter(2, 2, 2, 0.1)

        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')

        for i in range(10):
            y_pil = color_jitter(x_pil)
            self.assertEqual(y_pil.mode, x_pil.mode)

            y_pil_2 = color_jitter(x_pil_2)
            self.assertEqual(y_pil_2.mode, x_pil_2.mode)

    def test_gray(self):
        """Unit tests for grayscale transform"""

        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')
        gray_np = np.array(x_pil_2)

        # Test Set: Gray an image with desired number of output channels
        # Case 1: RGB -> 1 channel grayscale
        trans1 = transform.Gray(num_output_channels=1)
        gray_pil_1 = trans1(x_pil)
        gray_np_1 = np.array(gray_pil_1)
        # self.assertEqual(gray_pil_1.mode, 'L', 'mode should be L')
        self.assertEqual(gray_np_1.shape[1:], tuple(x_shape[0:2]), 'should be 1 channel')
        assert np.allclose(gray_np/255, gray_np_1[0], atol=0.01)

        # Case 2: RGB -> 3 channel grayscale
        trans2 = transform.Gray(num_output_channels=3)
        gray_pil_2 = trans2(x_pil)
        gray_np_2 = np.array(gray_pil_2)
        # self.assertEqual(gray_pil_2.mode, 'RGB', 'mode should be RGB')
        self.assertEqual(gray_np_2.shape, tuple(x_shape), 'should be 3 channel')
        np.testing.assert_equal(gray_np_2[:, :, 0], gray_np_2[:, :, 1])
        np.testing.assert_equal(gray_np_2[:, :, 1], gray_np_2[:, :, 2])
        assert np.allclose(gray_np/255, gray_np_2[:, :, 0], atol=0.01)

        # Case 3: 1 channel grayscale -> 1 channel grayscale
        trans3 = transform.Gray(num_output_channels=1)
        gray_pil_3 = trans3(x_pil_2)
        gray_np_3 = np.array(gray_pil_3)
        # self.assertEqual(gray_pil_3.mode, 'L', 'mode should be L')
        self.assertEqual(gray_np_3.shape[1:], tuple(x_shape[0:2]), 'should be 1 channel')
        np.testing.assert_allclose(gray_np/255, gray_np_3[0], atol=0.01)

        # Case 4: 1 channel grayscale -> 3 channel grayscale
        trans4 = transform.Gray(num_output_channels=3)
        gray_pil_4 = trans4(x_pil_2)
        gray_np_4 = np.array(gray_pil_4)
        # self.assertEqual(gray_pil_4.mode, 'RGB', 'mode should be RGB')
        self.assertEqual(gray_np_4.shape, tuple(x_shape), 'should be 3 channel')
        np.testing.assert_equal(gray_np_4[:, :, 0], gray_np_4[:, :, 1])
        np.testing.assert_equal(gray_np_4[:, :, 1], gray_np_4[:, :, 2])
        np.testing.assert_allclose(gray_np/255, gray_np_4[:, :, 0], atol=0.01)

    @unittest.skipIf(stats is None, 'scipy.stats not available')
    def test_random_gray(self):
        """Unit tests for random grayscale transform"""

        # Test Set 1: RGB -> 3 channel grayscale
        random_state = random.getstate()
        random.seed(42)
        x_shape = [2, 2, 3]
        x_np = np.random.randint(0, 256, x_shape, np.uint8)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')
        gray_np = np.array(x_pil_2)

        num_samples = 250
        num_gray = 0
        for _ in range(num_samples):
            gray_pil_2 = transform.RandomGray(p=0.5)(x_pil)
            gray_np_2 = np.array(gray_pil_2)
            if np.array_equal(gray_np_2[:, :, 0], gray_np_2[:, :, 1]) and \
                    np.array_equal(gray_np_2[:, :, 1], gray_np_2[:, :, 2]) and \
                    np.array_equal(gray_np, gray_np_2[:, :, 0]):
                num_gray = num_gray + 1

        p_value = stats.binom_test(num_gray, num_samples, p=0.5)
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

        # Test Set 2: grayscale -> 1 channel grayscale
        random_state = random.getstate()
        random.seed(42)
        x_shape = [2, 2, 3]
        x_np = np.random.randint(0, 256, x_shape, np.uint8)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')
        gray_np = np.array(x_pil_2)

        num_samples = 250
        num_gray = 0
        for _ in range(num_samples):
            gray_pil_3 = transform.RandomGray(p=0.5)(x_pil_2)
            gray_np_3 = np.array(gray_pil_3)
            if np.array_equal(gray_np, gray_np_3):
                num_gray = num_gray + 1

        p_value = stats.binom_test(num_gray, num_samples, p=1.0)  # Note: grayscale is always unchanged
        random.setstate(random_state)
        self.assertGreater(p_value, 0.0001)

        # Test set 3: Explicit tests
        x_shape = [2, 2, 3]
        x_data = [0, 5, 13, 54, 135, 226, 37, 8, 234, 90, 255, 1]
        x_np = np.array(x_data, dtype=np.uint8).reshape(x_shape)
        x_pil = Image.fromarray(x_np, mode='RGB')
        x_pil_2 = x_pil.convert('L')
        gray_np = np.array(x_pil_2)

        # Case 3a: RGB -> 3 channel grayscale (grayscaled)
        trans2 = transform.RandomGray(p=1.0)
        gray_pil_2 = trans2(x_pil)
        gray_np_2 = np.array(gray_pil_2)
        self.assertEqual(gray_pil_2.mode, 'RGB', 'mode should be RGB')
        self.assertEqual(gray_np_2.shape, tuple(x_shape), 'should be 3 channel')
        np.testing.assert_equal(gray_np_2[:, :, 0], gray_np_2[:, :, 1])
        np.testing.assert_equal(gray_np_2[:, :, 1], gray_np_2[:, :, 2])
        np.testing.assert_equal(gray_np, gray_np_2[:, :, 0])

        # Case 3b: RGB -> 3 channel grayscale (unchanged)
        trans2 = transform.RandomGray(p=0.0)
        gray_pil_2 = trans2(x_pil)
        gray_np_2 = np.array(gray_pil_2)
        self.assertEqual(gray_pil_2.mode, 'RGB', 'mode should be RGB')
        self.assertEqual(gray_np_2.shape, tuple(x_shape), 'should be 3 channel')
        np.testing.assert_equal(x_np, gray_np_2)

        # Case 3c: 1 channel grayscale -> 1 channel grayscale (grayscaled)
        trans3 = transform.RandomGray(p=1.0)
        gray_pil_3 = trans3(x_pil_2)
        gray_np_3 = np.array(gray_pil_3)
        self.assertEqual(gray_pil_3.mode, 'L', 'mode should be L')
        self.assertEqual(gray_np_3.shape, tuple(x_shape[0:2]), 'should be 1 channel')
        np.testing.assert_equal(gray_np, gray_np_3)

        # Case 3d: 1 channel grayscale -> 1 channel grayscale (unchanged)
        trans3 = transform.RandomGray(p=0.0)
        gray_pil_3 = trans3(x_pil_2)
        gray_np_3 = np.array(gray_pil_3)
        self.assertEqual(gray_pil_3.mode, 'L', 'mode should be L')
        self.assertEqual(gray_np_3.shape, tuple(x_shape[0:2]), 'should be 1 channel')
        np.testing.assert_equal(gray_np, gray_np_3)

    def test_RandomPerspective(self):
        img = jt.random((30,40,3))
        result = transform.Compose([
            transform.ToPILImage(),
            transform.RandomPerspective(p=1),
            transform.ToTensor(),
        ])(img)


    def test_RandomResizedCrop(self):
        img = jt.random((30,40,3))
        result = transform.Compose([
            transform.ToPILImage(),
            transform.RandomResizedCrop(20),
            transform.ToTensor(),
        ])(img)


    def test_FiveCrop(self):
        img = jt.random((30,40,3))
        result = transform.Compose([
            transform.ToPILImage(),
            transform.FiveCrop(20),
            transform.ToTensor(),
        ])(img)


    def test_TenCrop(self):
        img = jt.random((30,40,3))
        result = transform.Compose([
            transform.ToPILImage(),
            transform.TenCrop(20),
            transform.ToTensor(),
        ])(img)


    def test_RandomRotation(self):
        img = jt.random((30,40,3))
        result = transform.Compose([
            transform.ToPILImage(),
            transform.RandomRotation(20),
            transform.ToTensor(),
        ])(img)


    def test_RandomAffine(self):
        img = jt.random((30,40,3))
        result = transform.Compose([
            transform.ToPILImage(),
            transform.RandomAffine(20),
            transform.ToTensor(),
        ])(img)

    def test_not_pil_image(self):
        img = jt.random((30,40,3))
        result = transform.Compose([
            transform.RandomAffine(20),
            transform.ToTensor(),
        ])(img)

        img = jt.random((30,40,3))
        result = transform.Compose([
            transform.ToPILImage(),
            transform.Gray(),
            transform.Resize(20),
            transform.ToTensor(),
        ])(img)




if __name__ == '__main__':
    unittest.main()

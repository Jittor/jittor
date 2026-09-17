"""Sampler RNG continuation through the maintained Torch random owners."""

import random
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SubsetRandomSampler, TensorDataset


def values(tensor):
    return tensor.detach().clone().cpu().numpy().copy()


def sample(kind, generator=None):
    if kind == "random":
        return list(RandomSampler(range(16), generator=generator))
    if kind == "replacement":
        return list(RandomSampler(range(16), replacement=True, num_samples=23,
                                  generator=generator))
    if kind == "subset":
        return list(SubsetRandomSampler([7, 1, 4, 4, 9, 0], generator=generator))
    dataset = TensorDataset(torch.arange(16, device="cpu"))
    loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)
    return [int(value) for batch in loader for value in batch[0].tolist()]


class TestSamplerRNG(unittest.TestCase):
    def test_default_sampling_tracks_torch_seed_and_state(self):
        for kind in ("random", "replacement", "subset", "loader"):
            with self.subTest(kind=kind):
                random.seed(91)
                torch.manual_seed(777)
                first = sample(kind)
                torch.manual_seed(777)
                self.assertEqual(sample(kind), first)
                state = torch.get_rng_state()
                expected = sample(kind)
                torch.set_rng_state(state)
                self.assertEqual(sample(kind), expected)
                if kind == "subset":
                    self.assertEqual(sorted(expected), [0, 1, 4, 4, 7, 9])

    def test_explicit_generator_restore_is_isolated_from_defaults(self):
        for kind in ("random", "subset", "loader"):
            with self.subTest(kind=kind):
                torch.manual_seed(42)
                generator = torch.Generator(device="cpu").manual_seed(777)
                state = generator.get_state()
                default = torch.get_rng_state()
                expected = sample(kind, generator)
                np.testing.assert_array_equal(values(torch.get_rng_state()), values(default))
                generator.set_state(state)
                self.assertEqual(sample(kind, generator), expected)
                np.testing.assert_array_equal(values(torch.get_rng_state()), values(default))

    def test_nonreplacement_requested_samples_repeat_permutations(self):
        generator = torch.Generator(device="cpu").manual_seed(777)
        sampled = list(RandomSampler(range(8), num_samples=19, generator=generator))
        self.assertEqual(len(sampled), 19)
        self.assertEqual(sorted(sampled[:8]), list(range(8)))
        self.assertEqual(sorted(sampled[8:16]), list(range(8)))
        self.assertEqual(len(set(sampled[16:])), 3)

    def test_explicit_replacement_support_is_not_silently_ignored(self):
        generator = torch.Generator(device="cpu").manual_seed(777)
        state = generator.get_state()
        expected = [7, 15, 11, 6, 7, 1, 7, 13, 15, 4, 7, 9, 14, 10, 8, 7,
                    2, 13, 14, 0, 11, 1, 2]
        self.assertEqual(sample("replacement", generator), expected)
        generator.set_state(state)
        self.assertEqual(sample("replacement", generator), expected)

    @unittest.skipUnless(hasattr(torch, "_torch_compat_install_context"),
                         "Jittor explicit Generator device policy")
    def test_explicit_replacement_generator_is_cpu_only(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA generator construction")
        generator = torch.Generator(device="cuda").manual_seed(7)
        with self.assertRaisesRegex(RuntimeError, "explicit Generator.*CPU only"):
            sample("replacement", generator)

    def test_explicit_replacement_does_not_advance_default_rng(self):
        torch.manual_seed(42)
        default_state = torch.get_rng_state()
        generator = torch.Generator(device="cpu").manual_seed(777)
        sample("replacement", generator)
        np.testing.assert_array_equal(values(torch.get_rng_state()), values(default_state))

    def test_explicit_replacement_with_workers_keeps_sample_count(self):
        generator = torch.Generator(device="cpu").manual_seed(777)
        dataset = TensorDataset(torch.arange(16, device="cpu"))
        loader = DataLoader(dataset, batch_size=4, sampler=RandomSampler(
            dataset, replacement=True, num_samples=23, generator=generator),
                            num_workers=1)
        sampled = [int(value) for batch in loader for value in batch[0].tolist()]
        self.assertEqual(len(sampled), 23)
        self.assertTrue(all(0 <= index < len(dataset) for index in sampled))

    def test_invalid_sampler_parameters_are_rejected(self):
        for replacement in (0, 1, "yes"):
            with self.assertRaises(TypeError):
                RandomSampler(range(8), replacement=replacement)
        for count in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                RandomSampler(range(8), num_samples=count)

    @unittest.skipUnless(hasattr(torch, "_torch_compat_install_context"),
                         "Jittor worker-thread seed metadata policy")
    def test_worker_seed_metadata_uses_full_cpu_seed(self):
        class SeedDataset:
            def __len__(self):
                return 2

            def __getitem__(self, index):
                info = torch.utils.data.get_worker_info()
                return str(info.seed)

        seed = (1 << 63) + 17
        torch.manual_seed(seed)
        loader = DataLoader(SeedDataset(), batch_size=1, num_workers=1)
        self.assertEqual([int(batch[0]) for batch in loader], [seed, seed])


if __name__ == "__main__":
    unittest.main()

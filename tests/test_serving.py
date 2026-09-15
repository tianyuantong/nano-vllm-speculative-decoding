import unittest
from types import SimpleNamespace

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.layers.sampler import Sampler
from nanovllm.sampling_params import SamplingParams


class TestGreedySampling(unittest.TestCase):

    def test_temperature_boundary(self):
        SamplingParams(temperature=0)
        SamplingParams(temperature=0.6)

        for temperature in (-0.1, 1e-12, 1e-10):
            with self.assertRaises(AssertionError):
                SamplingParams(temperature=temperature)

    def test_prepare_sample_modes(self):
        runner = ModelRunner.__new__(ModelRunner)

        greedy = [
            SimpleNamespace(temperature=0),
            SimpleNamespace(temperature=0),
        ]
        self.assertIsNone(runner.prepare_sample(greedy))

        random = [
            SimpleNamespace(temperature=0.6),
            SimpleNamespace(temperature=0.8),
        ]
        temperatures = runner.prepare_sample(random)
        self.assertTrue(temperatures.is_cuda)
        torch.testing.assert_close(
            temperatures.cpu(),
            torch.tensor([0.6, 0.8]),
        )

        mixed = [
            SimpleNamespace(temperature=0),
            SimpleNamespace(temperature=0.6),
        ]
        with self.assertRaises(ValueError):
            runner.prepare_sample(mixed)

    def test_greedy_matches_argmax_and_repeats(self):
        sampler = Sampler()

        for batch_size in (1, 2, 16):
            logits = torch.randn(
                batch_size,
                128,
                device="cuda",
            )
            expected = logits.argmax(dim=-1)

            for _ in range(3):
                actual = sampler(logits, None)
                torch.testing.assert_close(actual, expected)

    def test_greedy_bf16_tie_matches_argmax(self):
        sampler = Sampler()
        logits = torch.tensor(
            [
                [1.0, 3.0, 3.0, 2.0],
                [5.0, 5.0, 1.0, 5.0],
            ],
            dtype=torch.bfloat16,
            device="cuda",
        )

        expected = logits.argmax(dim=-1)
        actual = sampler(logits, None)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.tolist(), [1, 0])

    def test_random_path_still_runs(self):
        sampler = Sampler()
        logits = torch.randn(2, 128, device="cuda")
        temperatures = torch.tensor(
            [0.6, 0.8],
            dtype=torch.float32,
            device="cuda",
        )

        output = sampler(logits, temperatures)

        self.assertEqual(output.shape, (2,))
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output < 128))


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from nimbus_infer.sampling import (
    SamplingParams,
    apply_repetition_penalty,
    sample,
    top_k_filter,
    top_p_filter,
)


class TestSamplingFilters(unittest.TestCase):
    def test_top_k_keeps_only_k(self):
        logits = np.array([1.0, 5.0, 3.0, 2.0, 4.0], dtype=np.float32)
        filtered = top_k_filter(logits, 2)
        finite = np.isfinite(filtered)
        self.assertEqual(int(finite.sum()), 2)
        self.assertTrue(filtered[1] == 5.0 and filtered[4] == 4.0)

    def test_top_p_truncates_tail(self):
        logits = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        filtered = top_p_filter(logits, 0.5)
        # Almost all probability mass is on index 0, so the tail must be cut.
        self.assertEqual(int(np.isfinite(filtered).sum()), 1)

    def test_repetition_penalty_downweights_seen_tokens(self):
        logits = np.array([2.0, 2.0], dtype=np.float32)
        adjusted = apply_repetition_penalty(logits, [0], 2.0)
        self.assertLess(adjusted[0], adjusted[1])

    def test_repetition_penalty_negative_value(self):
        logits = np.array([-2.0], dtype=np.float32)
        adjusted = apply_repetition_penalty(logits, [0], 2.0)
        self.assertAlmostEqual(float(adjusted[0]), -4.0, places=6)


class TestSampling(unittest.TestCase):
    def test_greedy_picks_argmax(self):
        logits = np.array([0.1, 3.0, 0.2], dtype=np.float32)
        params = SamplingParams(temperature=0.0)
        for _ in range(5):
            self.assertEqual(sample(logits, params), 1)

    def test_top_k_one_is_greedy(self):
        logits = np.array([0.1, 3.0, 2.9], dtype=np.float32)
        params = SamplingParams(temperature=1.0, top_k=1)
        rng = np.random.default_rng(0)
        for _ in range(20):
            self.assertEqual(sample(logits, params, rng=rng), 1)

    def test_seed_is_reproducible(self):
        logits = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        params = SamplingParams(temperature=1.0, seed=123)
        a = [sample(logits, params, rng=np.random.default_rng(123)) for _ in range(1)]
        b = [sample(logits, params, rng=np.random.default_rng(123)) for _ in range(1)]
        self.assertEqual(a, b)

    def test_sampling_respects_distribution(self):
        logits = np.array([0.0, 100.0], dtype=np.float32)
        params = SamplingParams(temperature=1.0, seed=7)
        rng = np.random.default_rng(7)
        draws = [sample(logits, params, rng=rng) for _ in range(50)]
        self.assertTrue(all(d == 1 for d in draws))

    def test_temperature_sharpens_distribution(self):
        logits = np.array([2.0, 1.0], dtype=np.float32)
        cold = SamplingParams(temperature=0.01)
        rng = np.random.default_rng(0)
        draws = [sample(logits, cold, rng=rng) for _ in range(30)]
        self.assertTrue(all(d == 0 for d in draws))


if __name__ == "__main__":
    unittest.main()
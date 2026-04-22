import unittest

from summbench.metrics.composite import combine_hallucination_signals


class CompositeMetricTests(unittest.TestCase):
    def test_formula_matches_notebook_weights(self):
        score = combine_hallucination_signals(
            source_f1=0.80,
            reference_f1=0.60,
            official_selfcheck_score=0.50,
        )
        expected = 0.3 * (1 - 0.80) + 0.3 * (1 - 0.60) + 0.4 * 0.50
        self.assertEqual(score, expected)

    def test_returns_none_if_any_signal_is_missing(self):
        score = combine_hallucination_signals(
            source_f1=0.80,
            reference_f1=None,
            official_selfcheck_score=0.50,
        )
        self.assertIsNone(score)


if __name__ == "__main__":
    unittest.main()

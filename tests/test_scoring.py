import unittest

from src.scoring import ScoringRules, score_tip, tendency


class ScoringTests(unittest.TestCase):
    def test_tendency(self):
        self.assertEqual(tendency(2, 1), 1)
        self.assertEqual(tendency(1, 1), 0)
        self.assertEqual(tendency(0, 2), -1)

    def test_additive_exact_score(self):
        rules = ScoringRules()
        self.assertEqual(score_tip(2, 1, 2, 1, rules), 9.0)

    def test_additive_partial_score(self):
        rules = ScoringRules()
        self.assertEqual(score_tip(2, 1, 3, 2, rules), 3.0)

    def test_highest_only_and_joker(self):
        rules = ScoringRules(combination_mode="highest_only", joker_multiplier=2)
        self.assertEqual(score_tip(2, 1, 2, 1, rules, use_joker=True), 10.0)

    def test_invalid_rules(self):
        with self.assertRaises(ValueError):
            ScoringRules(exact_points=-1)


if __name__ == "__main__":
    unittest.main()


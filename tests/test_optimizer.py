import unittest

import numpy as np

from src.optimizer import expected_points_for_tip, optimize_tip
from src.prediction import MatchPrediction, outcome_probabilities
from src.scoring import ScoringRules


def prediction_from_matrix(matrix):
    matrix = np.asarray(matrix, dtype=float)
    matrix /= matrix.sum()
    home, draw, away = outcome_probabilities(matrix)
    return MatchPrediction(
        home_team="A",
        away_team="B",
        expected_home_goals=1.0,
        expected_away_goals=1.0,
        score_matrix=matrix,
        home_win=home,
        draw=draw,
        away_win=away,
        confidence=0.7,
        confidence_label="hoch",
        data_uncertainty=0.1,
        contributions={},
    )


class OptimizerTests(unittest.TestCase):
    def test_expected_value_for_certain_result(self):
        matrix = np.zeros((3, 3))
        matrix[2, 1] = 1.0
        expected, variance = expected_points_for_tip(
            matrix, 2, 1, ScoringRules()
        )
        self.assertEqual(expected, 9.0)
        self.assertEqual(variance, 0.0)

    def test_optimizer_selects_certain_result(self):
        matrix = np.zeros((3, 3))
        matrix[2, 1] = 1.0
        recommendation = optimize_tip(
            prediction_from_matrix(matrix), ScoringRules(), strategy="safe"
        )
        self.assertEqual(recommendation.score, "2:1")

    def test_all_strategy_modes_return_valid_tip(self):
        matrix = np.array(
            [[0.10, 0.08, 0.03], [0.18, 0.20, 0.06], [0.15, 0.12, 0.08]]
        )
        prediction = prediction_from_matrix(matrix)
        for strategy in ("safe", "value", "risk"):
            recommendation = optimize_tip(
                prediction,
                ScoringRules(),
                strategy=strategy,
                optimizer_config={},
            )
            self.assertGreaterEqual(recommendation.expected_points, 0)

    def test_unknown_strategy_fails(self):
        prediction = prediction_from_matrix([[0.25, 0.25], [0.25, 0.25]])
        with self.assertRaises(ValueError):
            optimize_tip(prediction, ScoringRules(), strategy="unknown")


if __name__ == "__main__":
    unittest.main()


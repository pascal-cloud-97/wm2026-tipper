import unittest

import numpy as np

from src.prediction import (
    MatchPrediction,
    outcome_probabilities,
    poisson_score_matrix,
    predict_match,
)


class PredictionTests(unittest.TestCase):
    def test_poisson_matrix_is_probability_distribution(self):
        matrix = poisson_score_matrix(1.6, 1.1, max_goals=6)
        self.assertEqual(matrix.shape, (7, 7))
        self.assertAlmostEqual(float(matrix.sum()), 1.0, places=12)
        self.assertTrue(np.all(matrix >= 0))

    def test_outcomes_sum_to_one(self):
        matrix = poisson_score_matrix(1.6, 1.1)
        self.assertAlmostEqual(sum(outcome_probabilities(matrix)), 1.0, places=12)

    def test_rating_edge_increases_home_expectation(self):
        config = {
            "max_goals": 6,
            "base_home_goals": 1.4,
            "base_away_goals": 1.2,
            "min_expected_goals": 0.2,
            "max_expected_goals": 4.5,
            "weights": {"rating": 0.5},
        }
        prediction = predict_match(
            {
                "home_team": "A",
                "away_team": "B",
                "rating_diff": 300,
                "data_uncertainty": 0.2,
            },
            config,
        )
        self.assertGreater(
            prediction.expected_home_goals, prediction.expected_away_goals
        )

    def test_fifa_points_are_blended_with_elo(self):
        config = {
            "max_goals": 6,
            "base_home_goals": 1.4,
            "base_away_goals": 1.2,
            "min_expected_goals": 0.2,
            "max_expected_goals": 4.5,
            "fifa_blend_weight": 0.5,
            "weights": {"rating": 0.5},
        }
        elo_only = predict_match(
            {
                "home_team": "A",
                "away_team": "B",
                "rating_diff": 200,
                "data_uncertainty": 0.2,
            },
            config,
        )
        blended = predict_match(
            {
                "home_team": "A",
                "away_team": "B",
                "rating_diff": 200,
                "fifa_points_diff": -200,
                "data_uncertainty": 0.2,
            },
            config,
        )
        self.assertLess(
            blended.expected_home_goals / blended.expected_away_goals,
            elo_only.expected_home_goals / elo_only.expected_away_goals,
        )

    def test_invalid_expected_goals(self):
        with self.assertRaises(ValueError):
            poisson_score_matrix(0, 1.0)

    def test_market_and_availability_can_shift_prediction(self):
        config = {
            "max_goals": 6,
            "base_home_goals": 1.4,
            "base_away_goals": 1.2,
            "min_expected_goals": 0.2,
            "max_expected_goals": 4.5,
            "weights": {"availability": 0.2, "market": 0.2},
        }
        neutral = predict_match(
            {"home_team": "A", "away_team": "B", "data_uncertainty": 0.5},
            config,
        )
        shifted = predict_match(
            {
                "home_team": "A",
                "away_team": "B",
                "availability_edge": -1.0,
                "market_log_odds_edge": -1.0,
                "data_uncertainty": 0.2,
            },
            config,
        )
        self.assertLess(
            shifted.expected_home_goals / shifted.expected_away_goals,
            neutral.expected_home_goals / neutral.expected_away_goals,
        )

    def test_enabled_calibration_changes_outcomes_but_keeps_distribution(self):
        config = {
            "max_goals": 6,
            "base_home_goals": 1.4,
            "base_away_goals": 1.2,
            "min_expected_goals": 0.2,
            "max_expected_goals": 4.5,
            "weights": {},
            "calibration": {
                "enabled": True,
                "inverse_temperature": 0.8,
                "home_bias": -0.05,
                "draw_bias": 0.10,
                "away_bias": -0.05,
            },
        }
        prediction = predict_match(
            {"home_team": "A", "away_team": "B", "data_uncertainty": 0.2},
            config,
        )
        self.assertAlmostEqual(
            prediction.home_win + prediction.draw + prediction.away_win,
            1.0,
        )
        self.assertNotAlmostEqual(
            prediction.draw,
            prediction.uncalibrated_draw,
        )
        self.assertAlmostEqual(float(prediction.score_matrix.sum()), 1.0)

    def test_representative_score_matches_most_likely_outcome(self):
        matrix = np.array(
            [
                [0.05, 0.08, 0.04],
                [0.20, 0.25, 0.08],
                [0.18, 0.08, 0.04],
            ]
        )
        matrix /= matrix.sum()
        home, draw, away = outcome_probabilities(matrix)
        prediction = MatchPrediction(
            home_team="Heim",
            away_team="Auswärts",
            expected_home_goals=1.4,
            expected_away_goals=1.1,
            score_matrix=matrix,
            home_win=home,
            draw=draw,
            away_win=away,
            confidence=0.6,
            confidence_label="mittel",
            data_uncertainty=0.2,
            contributions={},
        )
        outcome_code, label, probability = prediction.most_likely_outcome()
        score, score_probability = prediction.representative_score()
        self.assertEqual(outcome_code, "1")
        self.assertEqual(label, "Heim")
        self.assertGreater(probability, draw)
        home_goals, away_goals = map(int, score.split(":"))
        self.assertGreater(home_goals, away_goals)
        self.assertGreater(score_probability, 0)

    def test_modal_score_can_differ_from_likely_outcome_score(self):
        matrix = np.array(
            [
                [0.06, 0.07, 0.03],
                [0.18, 0.24, 0.07],
                [0.17, 0.10, 0.08],
            ]
        )
        matrix /= matrix.sum()
        home, draw, away = outcome_probabilities(matrix)
        prediction = MatchPrediction(
            home_team="Heim",
            away_team="Auswärts",
            expected_home_goals=1.5,
            expected_away_goals=1.1,
            score_matrix=matrix,
            home_win=home,
            draw=draw,
            away_win=away,
            confidence=0.6,
            confidence_label="mittel",
            data_uncertainty=0.2,
            contributions={},
        )
        self.assertEqual(prediction.modal_score()[0], "1:1")
        self.assertEqual(prediction.representative_score()[0], "1:0")


if __name__ == "__main__":
    unittest.main()

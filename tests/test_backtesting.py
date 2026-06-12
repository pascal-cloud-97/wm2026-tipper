import unittest

import numpy as np
import pandas as pd

from src.backtesting import fit_and_validate_calibrator, run_backtest


class BacktestingTests(unittest.TestCase):
    def setUp(self):
        rows = []
        teams = ["A", "B", "C", "D"]
        for index in range(100):
            home = teams[index % 4]
            away = teams[(index + 1 + (index // 4) % 2) % 4]
            rows.append(
                {
                    "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=index * 14),
                    "home_team": home,
                    "away_team": away,
                    "home_goals": [2, 1, 0][index % 3],
                    "away_goals": [0, 1, 2][index % 3],
                    "neutral": True,
                }
            )
        self.history = pd.DataFrame(rows)
        self.teams = pd.DataFrame(
            {
                "team_id": teams,
                "team_name": teams,
            }
        )
        self.config = {
            "max_goals": 6,
            "base_home_goals": 1.4,
            "base_away_goals": 1.2,
            "min_expected_goals": 0.2,
            "max_expected_goals": 4.5,
            "weights": {
                "rating": 0.3,
                "form": 0.2,
                "attack_defense": 0.25,
                "head_to_head": 0.05,
            },
        }

    def test_backtest_returns_finite_metrics_without_current_ratings(self):
        report = run_backtest(
            self.history,
            self.config,
            known_teams=self.teams,
            start_date="2023-06-01",
            min_prior_matches=2,
            max_matches=20,
            calibration_bins=5,
        )
        self.assertEqual(report.matches, 20)
        self.assertTrue(np.isfinite(report.brier_score))
        self.assertTrue(np.isfinite(report.log_loss))
        self.assertTrue(np.isfinite(report.baseline_brier_score))
        self.assertTrue(np.isfinite(report.baseline_log_loss))
        self.assertGreaterEqual(report.accuracy, 0.0)
        self.assertLessEqual(report.accuracy, 1.0)
        self.assertEqual(
            int(report.calibration["observations"].sum()),
            report.matches * 3,
        )

    def test_backtest_rejects_empty_history(self):
        with self.assertRaises(ValueError):
            run_backtest(pd.DataFrame(), self.config)

    def test_calibrator_uses_temporal_holdout(self):
        report = run_backtest(
            self.history,
            self.config,
            known_teams=self.teams,
            start_date="2023-03-01",
            min_prior_matches=1,
            max_matches=None,
        )
        validation = fit_and_validate_calibrator(
            report.predictions,
            validation_start="2025-01-01",
        )
        self.assertGreaterEqual(validation.training_matches, 30)
        self.assertGreaterEqual(validation.validation_matches, 30)
        self.assertTrue(np.isfinite(validation.calibrated_log_loss))


if __name__ == "__main__":
    unittest.main()

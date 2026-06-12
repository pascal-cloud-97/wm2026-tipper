import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml

from src.data_loader import load_dataset
from src.feature_engineering import build_match_features
from src.history_updater import sync_completed_matches
from src.prediction import MatchPrediction, predict_all
from src.simulation import (
    _actual_or_sampled_score,
    _assign_third_placed,
    simulate_world_cup,
)


ROOT = Path(__file__).resolve().parents[1]


class WorldCupSimulationTests(unittest.TestCase):
    def test_third_place_slots_receive_all_qualified_groups(self):
        qualified = {
            group: f"TEAM_{group}" for group in "ABCDEFGH"
        }
        assignment = _assign_third_placed(qualified)
        self.assertEqual(len(assignment), 8)
        self.assertEqual(set(assignment.values()), set(qualified.values()))

    def test_official_dataset_simulates_to_one_champion(self):
        with (ROOT / "config.yaml").open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        bundle = load_dataset(ROOT / "data" / "world_cup_2026")
        features = build_match_features(
            bundle.matches,
            bundle.teams,
            bundle.historical_results,
            bundle.ratings,
        )
        predictions = predict_all(features, config["model"])
        result = simulate_world_cup(
            bundle.matches,
            predictions,
            bundle.teams,
            runs=100,
            seed=2026,
        )
        self.assertEqual(len(result.team_probabilities), 48)
        self.assertAlmostEqual(
            result.team_probabilities["champion_probability"].sum(),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            result.team_probabilities["final_probability"].sum(),
            2.0,
            places=12,
        )

    def test_completed_match_uses_actual_score(self):
        prediction = MatchPrediction(
            home_team="A",
            away_team="B",
            expected_home_goals=1,
            expected_away_goals=1,
            score_matrix=np.full((2, 2), 0.25),
            home_win=0.25,
            draw=0.5,
            away_win=0.25,
            confidence=0.5,
            confidence_label="mittel",
            data_uncertainty=0.5,
            contributions={},
        )
        score = _actual_or_sampled_score(
            {
                "status": "completed",
                "actual_home_goals": 3,
                "actual_away_goals": 1,
            },
            prediction,
            np.random.default_rng(1),
        )
        self.assertEqual(score, (3, 1))

    def test_history_sync_marks_world_cup_match_completed(self):
        history = pd.DataFrame(
            [
                {
                    "date": "2026-06-11",
                    "home_team": "MEX",
                    "away_team": "RSA",
                    "home_goals": 2,
                    "away_goals": 0,
                    "competition": "FIFA World Cup",
                }
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": "2026-06-11T13:00:00",
                    "home_team": "MEX",
                    "away_team": "RSA",
                    "stage": "Group",
                    "status": "scheduled",
                }
            ]
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "matches.csv"
            matches.to_csv(path, index=False)
            count = sync_completed_matches(history, path)
            synced = pd.read_csv(path)
        self.assertEqual(count, 1)
        self.assertEqual(synced.iloc[0]["status"], "completed")
        self.assertEqual(int(synced.iloc[0]["actual_home_goals"]), 2)


if __name__ == "__main__":
    unittest.main()

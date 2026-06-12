import unittest

import pandas as pd

from src.data_loader import DataBundle
from src.feature_engineering import build_match_features
from src.matchday import apply_match_results


class MatchdayTests(unittest.TestCase):
    def setUp(self):
        self.bundle = DataBundle(
            teams=pd.DataFrame(
                [
                    {
                        "team_id": "A",
                        "team_name": "A",
                        "country": "Country A",
                        "rating": 1500,
                    },
                    {
                        "team_id": "B",
                        "team_name": "B",
                        "country": "Country B",
                        "rating": 1500,
                    },
                    {
                        "team_id": "C",
                        "team_name": "C",
                        "country": "Country C",
                        "rating": 1500,
                    },
                ]
            ),
            matches=pd.DataFrame(
                [
                    {
                        "match_id": "M1",
                        "date": pd.Timestamp("2026-06-11 18:00:00"),
                        "home_team": "A",
                        "away_team": "B",
                        "group": "A",
                        "stage": "Group",
                        "venue_country": "Country A",
                        "status": "scheduled",
                        "actual_home_goals": pd.NA,
                        "actual_away_goals": pd.NA,
                    },
                    {
                        "match_id": "M2",
                        "date": pd.Timestamp("2026-06-15 18:00:00"),
                        "home_team": "A",
                        "away_team": "C",
                        "group": "A",
                        "stage": "Group",
                        "venue_country": "Neutral",
                        "status": "scheduled",
                        "actual_home_goals": pd.NA,
                        "actual_away_goals": pd.NA,
                    },
                ]
            ),
            historical_results=pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2026-06-01"),
                        "home_team": "C",
                        "away_team": "B",
                        "home_goals": 1,
                        "away_goals": 1,
                    }
                ]
            ),
            ratings=pd.DataFrame(),
            tips=pd.DataFrame(),
        )

    def test_result_updates_match_and_next_match_form(self):
        results = pd.DataFrame(
            [
                {
                    "result_id": 1,
                    "match_id": "M1",
                    "recorded_at": pd.Timestamp("2026-06-11 20:00:00"),
                    "home_goals": 2,
                    "away_goals": 0,
                    "source": "Official",
                }
            ]
        )
        updated = apply_match_results(self.bundle, results)
        first = updated.matches.set_index("match_id").loc["M1"]
        self.assertEqual(first["status"], "completed")
        self.assertEqual(int(first["actual_home_goals"]), 2)
        self.assertEqual(len(updated.historical_results), 2)

        features = build_match_features(
            updated.matches[updated.matches["match_id"] == "M2"],
            updated.teams,
            updated.historical_results,
        ).iloc[0]
        self.assertEqual(int(features["home_matches_5"]), 1)
        self.assertEqual(float(features["home_goals_for_5"]), 2.0)

    def test_correction_replaces_fixture_in_history_without_duplicate(self):
        results = pd.DataFrame(
            [
                {
                    "result_id": 1,
                    "match_id": "M1",
                    "recorded_at": pd.Timestamp("2026-06-11 20:00:00"),
                    "home_goals": 1,
                    "away_goals": 0,
                    "source": "Initial",
                },
                {
                    "result_id": 2,
                    "match_id": "M1",
                    "recorded_at": pd.Timestamp("2026-06-11 20:05:00"),
                    "home_goals": 2,
                    "away_goals": 0,
                    "source": "Correction",
                },
            ]
        )
        updated = apply_match_results(self.bundle, results)
        fixture = updated.historical_results[
            (updated.historical_results["home_team"] == "A")
            & (updated.historical_results["away_team"] == "B")
        ]
        self.assertEqual(len(fixture), 1)
        self.assertEqual(int(fixture.iloc[0]["home_goals"]), 2)
        self.assertEqual(fixture.iloc[0]["source"], "Correction")


if __name__ == "__main__":
    unittest.main()

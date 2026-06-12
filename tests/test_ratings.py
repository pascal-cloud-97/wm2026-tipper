import unittest

import pandas as pd

from src.ratings import current_elo_ratings, elo_snapshots, teams_with_current_elo


class RatingsTests(unittest.TestCase):
    def test_snapshot_uses_rating_before_current_result(self):
        history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 3,
                    "away_goals": 0,
                    "neutral": True,
                },
                {
                    "date": pd.Timestamp("2024-02-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 0,
                    "away_goals": 1,
                    "neutral": True,
                },
            ]
        )
        snapshots, ratings = elo_snapshots(history)
        self.assertEqual(snapshots.iloc[0]["home_rating"], 1500.0)
        self.assertEqual(snapshots.iloc[0]["away_rating"], 1500.0)
        self.assertGreater(
            snapshots.iloc[1]["home_rating"],
            snapshots.iloc[1]["away_rating"],
        )
        self.assertIn("A", ratings)
        self.assertIn("B", ratings)

    def test_current_ratings_return_latest_state(self):
        history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 2,
                    "away_goals": 0,
                    "neutral": True,
                }
            ]
        )
        ratings = current_elo_ratings(history)
        lookup = ratings.set_index("team_id")["rating"]
        self.assertGreater(lookup["A"], lookup["B"])
        self.assertGreater(ratings["as_of"].min(), history["date"].max())

    def test_teams_receive_current_elo_but_keep_fallback(self):
        history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 2,
                    "away_goals": 0,
                    "neutral": True,
                }
            ]
        )
        teams = pd.DataFrame(
            [
                {"team_id": "A", "rating": 1000},
                {"team_id": "B", "rating": 1000},
                {"team_id": "C", "rating": 1600},
            ]
        )
        updated = teams_with_current_elo(teams, history).set_index("team_id")
        self.assertGreater(updated.loc["A", "rating"], updated.loc["B", "rating"])
        self.assertEqual(updated.loc["C", "rating"], 1600)


if __name__ == "__main__":
    unittest.main()

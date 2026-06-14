import unittest

import pandas as pd

from src.feature_engineering import build_match_features
from src.form_analysis import team_form_curve


class FeatureEngineeringTests(unittest.TestCase):
    def test_future_results_do_not_leak_into_features(self):
        teams = pd.DataFrame(
            [
                {"team_id": "A", "team_name": "A", "rating": 1700},
                {"team_id": "B", "team_name": "B", "rating": 1600},
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": pd.Timestamp("2026-06-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Group",
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-07-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 9,
                    "away_goals": 0,
                }
            ]
        )
        result = build_match_features(matches, teams, history)
        self.assertEqual(int(result.iloc[0]["home_matches_5"]), 0)
        self.assertTrue(pd.isna(result.iloc[0]["home_goals_for_5"]))

    def test_time_sensitive_signals_are_used_before_kickoff(self):
        teams = pd.DataFrame(
            [
                {"team_id": "A", "team_name": "A", "rating": 1700},
                {"team_id": "B", "team_name": "B", "rating": 1700},
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": pd.Timestamp("2026-06-12 18:00:00"),
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Group",
                }
            ]
        )
        history = pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "team_id": "A",
                    "player_name": "Key player",
                    "status": "out",
                    "impact": 1.0,
                    "as_of": pd.Timestamp("2026-06-11"),
                    "source": "Test",
                },
                {
                    "team_id": "B",
                    "player_name": "Player",
                    "status": "available",
                    "impact": 0.8,
                    "as_of": pd.Timestamp("2026-06-11"),
                    "source": "Test",
                },
            ]
        )
        lineup_rows = []
        for team_id, rating in (("A", 70), ("B", 80)):
            for player in range(11):
                lineup_rows.append(
                    {
                        "match_id": "M1",
                        "team_id": team_id,
                        "player_name": f"{team_id}{player}",
                        "is_starting": True,
                        "expected_minutes": 90,
                        "player_rating": rating,
                        "as_of": pd.Timestamp("2026-06-12 12:00:00"),
                        "source": "Test",
                    }
                )
        odds = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "bookmaker": "Book",
                    "collected_at": pd.Timestamp("2026-06-12 12:00:00"),
                    "home_odds": 4.0,
                    "draw_odds": 3.2,
                    "away_odds": 1.9,
                    "source": "Test",
                }
            ]
        )
        result = build_match_features(
            matches,
            teams,
            history,
            availability=availability,
            lineups=pd.DataFrame(lineup_rows),
            odds=odds,
        ).iloc[0]
        self.assertLess(result["availability_edge"], 0)
        self.assertLess(result["lineup_strength_diff"], 0)
        self.assertLess(result["market_log_odds_edge"], 0)
        self.assertAlmostEqual(
            result["market_home_probability"]
            + result["market_draw_probability"]
            + result["market_away_probability"],
            1.0,
            places=12,
        )

    def test_swisslos_snapshot_keeps_raw_odds_and_removes_margin(self):
        teams = pd.DataFrame(
            [
                {"team_id": "A", "team_name": "A", "rating": 1700},
                {"team_id": "B", "team_name": "B", "rating": 1700},
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": pd.Timestamp("2026-06-12 18:00:00"),
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Group",
                }
            ]
        )
        history = pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
            ]
        )
        odds = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "bookmaker": "Swisslos",
                    "collected_at": pd.Timestamp("2026-06-12 12:00:00"),
                    "home_odds": 2.0,
                    "draw_odds": 3.5,
                    "away_odds": 4.0,
                    "source": "Swisslos Sporttip",
                }
            ]
        )
        result = build_match_features(
            matches, teams, history, odds=odds
        ).iloc[0]
        self.assertEqual(result["swisslos_home_odds"], 2.0)
        self.assertGreater(result["swisslos_margin"], 0)
        self.assertAlmostEqual(
            result["swisslos_home_probability"]
            + result["swisslos_draw_probability"]
            + result["swisslos_away_probability"],
            1.0,
            places=12,
        )

    def test_baseline_goals_use_only_results_before_kickoff(self):
        teams = pd.DataFrame(
            [
                {"team_id": "A", "team_name": "A"},
                {"team_id": "B", "team_name": "B"},
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": pd.Timestamp("2024-01-10"),
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Backtest",
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2024-01-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 1,
                    "away_goals": 1,
                },
                {
                    "date": pd.Timestamp("2024-02-01"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 9,
                    "away_goals": 9,
                },
            ]
        )
        result = build_match_features(matches, teams, history).iloc[0]
        self.assertAlmostEqual(result["baseline_goals"], 1.0)

    def test_utc_kickoff_allows_quote_after_local_venue_time(self):
        teams = pd.DataFrame(
            [
                {"team_id": "A", "team_name": "A", "rating": 1700},
                {"team_id": "B", "team_name": "B", "rating": 1700},
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": pd.Timestamp("2026-06-11 20:00:00"),
                    "kickoff_utc": pd.Timestamp("2026-06-12 02:00:00"),
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Group",
                }
            ]
        )
        history = pd.DataFrame(
            columns=[
                "date",
                "home_team",
                "away_team",
                "home_goals",
                "away_goals",
            ]
        )
        odds = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "bookmaker": "Swisslos",
                    "collected_at": pd.Timestamp("2026-06-11 21:45:00"),
                    "home_odds": 2.6,
                    "draw_odds": 3.0,
                    "away_odds": 3.0,
                    "source": "Swisslos Sporttip",
                }
            ]
        )

        result = build_match_features(
            matches, teams, history, odds=odds
        ).iloc[0]

        self.assertEqual(result["swisslos_home_odds"], 2.6)

    def test_cards_and_match_scoped_suspensions_affect_next_match_only(self):
        teams = pd.DataFrame(
            [
                {"team_id": "A", "team_name": "A", "rating": 1700},
                {"team_id": "B", "team_name": "B", "rating": 1700},
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "match_id": "NEXT",
                    "date": pd.Timestamp("2026-06-18"),
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Group",
                },
                {
                    "match_id": "LATER",
                    "date": pd.Timestamp("2026-06-24"),
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Group",
                },
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp("2026-06-11"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": 2,
                    "away_goals": 0,
                    "home_yellow_cards": 1,
                    "away_yellow_cards": 2,
                    "home_red_cards": 0,
                    "away_red_cards": 2,
                }
            ]
        )
        availability = pd.DataFrame(
            [
                {
                    "team_id": "B",
                    "player_name": "Suspended player",
                    "status": "suspended",
                    "impact": 0.65,
                    "as_of": pd.Timestamp("2026-06-11"),
                    "match_id": "NEXT",
                    "source": "Match report",
                }
            ]
        )

        result = build_match_features(
            matches,
            teams,
            history,
            availability=availability,
        ).set_index("match_id")

        self.assertGreater(result.loc["NEXT", "discipline_edge"], 0)
        self.assertGreater(result.loc["NEXT", "availability_edge"], 0)
        self.assertTrue(pd.isna(result.loc["LATER", "availability_edge"]))

    def test_form_curve_and_trend_use_only_prior_results(self):
        history = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(f"2026-06-0{day}"),
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": goals,
                    "away_goals": 1,
                }
                for day, goals in enumerate([0, 1, 2, 3, 4], start=1)
            ]
        )
        curve = team_form_curve(
            history,
            "A",
            before=pd.Timestamp("2026-06-06"),
            window=5,
        )
        self.assertEqual(len(curve), 5)
        self.assertEqual(curve.iloc[-1]["result"], "S")
        self.assertGreater(
            curve.iloc[-1]["rolling_points_5"],
            curve.iloc[0]["rolling_points_5"],
        )


if __name__ == "__main__":
    unittest.main()

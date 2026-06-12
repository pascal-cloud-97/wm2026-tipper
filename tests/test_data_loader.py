import unittest

import pandas as pd

from src.data_loader import DataValidationError, validate_dataframe


class DataLoaderTests(unittest.TestCase):
    def test_valid_matches_are_parsed(self):
        frame = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": "2026-06-12T18:00:00Z",
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Group",
                }
            ]
        )
        result = validate_dataframe(frame, "matches")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result["date"]))

    def test_missing_required_column_fails(self):
        frame = pd.DataFrame([{"team_id": "A"}])
        with self.assertRaises(DataValidationError):
            validate_dataframe(frame, "teams")

    def test_negative_goals_fail(self):
        frame = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "home_team": "A",
                    "away_team": "B",
                    "home_goals": -1,
                    "away_goals": 0,
                }
            ]
        )
        with self.assertRaises(DataValidationError):
            validate_dataframe(frame, "historical_results")

    def test_duplicate_match_id_fails(self):
        frame = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": "2026-01-01",
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Group",
                },
                {
                    "match_id": "M1",
                    "date": "2026-01-02",
                    "home_team": "C",
                    "away_team": "D",
                    "stage": "Group",
                },
            ]
        )
        with self.assertRaises(DataValidationError):
            validate_dataframe(frame, "matches")

    def test_duplicate_official_match_number_fails(self):
        frame = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "official_match_number": 1,
                    "date": "2026-06-11",
                    "home_team": "A",
                    "away_team": "B",
                    "stage": "Group",
                },
                {
                    "match_id": "M2",
                    "official_match_number": 1,
                    "date": "2026-06-12",
                    "home_team": "C",
                    "away_team": "D",
                    "stage": "Group",
                },
            ]
        )
        with self.assertRaises(DataValidationError):
            validate_dataframe(frame, "matches")

    def test_availability_requires_valid_status_and_impact(self):
        valid = pd.DataFrame(
            [
                {
                    "team_id": "A",
                    "player_name": "Player",
                    "status": "doubtful",
                    "impact": 0.8,
                    "as_of": "2026-06-10T10:00:00",
                    "source": "Test source",
                }
            ]
        )
        result = validate_dataframe(valid, "availability")
        self.assertEqual(result.iloc[0]["status"], "doubtful")
        invalid = valid.copy()
        invalid.loc[0, "impact"] = 1.5
        with self.assertRaises(DataValidationError):
            validate_dataframe(invalid, "availability")

    def test_odds_must_be_decimal_prices_above_one(self):
        odds = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "bookmaker": "Book",
                    "collected_at": "2026-06-10T10:00:00",
                    "home_odds": 1.0,
                    "draw_odds": 3.5,
                    "away_odds": 4.0,
                    "source": "Test source",
                }
            ]
        )
        with self.assertRaises(DataValidationError):
            validate_dataframe(odds, "odds")

    def test_outright_odds_validate_champion_market(self):
        frame = pd.DataFrame(
            [
                {
                    "team_id": "ESP",
                    "bookmaker": "Swisslos",
                    "market": "champion",
                    "collected_at": "2026-06-10T10:00:00",
                    "decimal_odds": 8.5,
                    "source": "Swisslos",
                }
            ]
        )
        result = validate_dataframe(frame, "outright_odds")
        self.assertEqual(result.iloc[0]["market"], "champion")
        invalid = frame.copy()
        invalid.loc[0, "market"] = "top_scorer"
        with self.assertRaises(DataValidationError):
            validate_dataframe(invalid, "outright_odds")


if __name__ == "__main__":
    unittest.main()

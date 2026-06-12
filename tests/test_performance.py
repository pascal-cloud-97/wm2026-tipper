import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.performance import calculate_betting_performance
from src.storage import (
    load_match_odds,
    load_match_results,
    load_outright_odds,
    load_paper_bets,
    save_match_odds,
    save_match_result,
    save_outright_odds,
    save_paper_bets,
    settle_paper_bets,
)


class PaperBettingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "test.sqlite"
        self.summary = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "home_team": "A",
                    "away_team": "B",
                    "bet_decision": "WETTEN",
                    "bet_outcome": "1",
                    "bet_odds": 2.0,
                    "swisslos_collected_at": pd.Timestamp("2026-06-10 10:00:00"),
                    "bet_model_probability": 0.60,
                    "bet_implied_probability": 0.50,
                    "bet_edge": 0.10,
                    "bet_expected_return": 0.20,
                    "bet_stake": 2.0,
                },
                {
                    "match_id": "M2",
                    "home_team": "C",
                    "away_team": "D",
                    "bet_decision": "KEINE WETTE",
                    "bet_outcome": "X",
                    "bet_odds": 4.0,
                    "swisslos_collected_at": pd.Timestamp("2026-06-10 10:00:00"),
                    "bet_model_probability": 0.30,
                    "bet_implied_probability": 0.25,
                    "bet_edge": 0.05,
                    "bet_expected_return": 0.20,
                    "bet_stake": 0.0,
                },
            ]
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_save_ignores_no_bet_and_duplicate(self):
        self.assertEqual(save_paper_bets(self.database, self.summary), 1)
        self.assertEqual(save_paper_bets(self.database, self.summary), 0)
        ledger = load_paper_bets(self.database)
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger.iloc[0]["status"], "pending")

    def test_settlement_and_performance(self):
        save_paper_bets(self.database, self.summary)
        matches = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": pd.Timestamp("2026-06-11 18:00:00"),
                    "actual_home_goals": 2,
                    "actual_away_goals": 1,
                }
            ]
        )
        self.assertEqual(settle_paper_bets(self.database, matches), 1)
        ledger = load_paper_bets(self.database)
        odds = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "bookmaker": "Swisslos",
                    "collected_at": pd.Timestamp("2026-06-11 17:00:00"),
                    "home_odds": 1.8,
                    "draw_odds": 3.5,
                    "away_odds": 4.5,
                }
            ]
        )
        performance = calculate_betting_performance(
            ledger,
            starting_bankroll=100.0,
            odds=odds,
            matches=matches,
        )
        self.assertEqual(performance.wins, 1)
        self.assertAlmostEqual(performance.total_profit, 2.0)
        self.assertAlmostEqual(performance.roi, 1.0)
        self.assertAlmostEqual(performance.current_bankroll, 102.0)
        self.assertAlmostEqual(performance.pending_stake, 0.0)
        self.assertAlmostEqual(performance.available_bankroll, 102.0)
        self.assertAlmostEqual(
            performance.average_closing_line_value,
            2.0 / 1.8 - 1.0,
        )

    def test_market_snapshots_persist_without_overwriting_history(self):
        match_odds = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "bookmaker": "Swisslos",
                    "collected_at": pd.Timestamp("2026-06-10 10:00:00"),
                    "home_odds": 2.0,
                    "draw_odds": 3.5,
                    "away_odds": 4.0,
                    "source": "Swisslos",
                },
                {
                    "match_id": "M1",
                    "bookmaker": "Swisslos",
                    "collected_at": pd.Timestamp("2026-06-10 12:00:00"),
                    "home_odds": 1.9,
                    "draw_odds": 3.6,
                    "away_odds": 4.2,
                    "source": "Swisslos",
                },
            ]
        )
        outright_odds = pd.DataFrame(
            [
                {
                    "team_id": "ESP",
                    "bookmaker": "Swisslos",
                    "market": "champion",
                    "collected_at": pd.Timestamp("2026-06-10 12:00:00"),
                    "decimal_odds": 8.0,
                    "source": "Swisslos",
                }
            ]
        )
        self.assertEqual(save_match_odds(self.database, match_odds), 2)
        self.assertEqual(save_match_odds(self.database, match_odds), 0)
        self.assertEqual(save_outright_odds(self.database, outright_odds), 1)
        self.assertEqual(len(load_match_odds(self.database)), 2)
        self.assertEqual(len(load_outright_odds(self.database)), 1)

    def test_match_result_snapshots_keep_corrections_and_return_latest(self):
        save_match_result(
            self.database,
            "M1",
            1,
            0,
            "First source",
            pd.Timestamp("2026-06-11 20:00:00"),
        )
        save_match_result(
            self.database,
            "M1",
            2,
            0,
            "Corrected source",
            pd.Timestamp("2026-06-11 20:05:00"),
        )
        history = load_match_results(self.database, latest_only=False)
        latest = load_match_results(self.database)
        self.assertEqual(len(history), 2)
        self.assertEqual(len(latest), 1)
        self.assertEqual(int(latest.iloc[0]["home_goals"]), 2)
        self.assertEqual(latest.iloc[0]["source"], "Corrected source")


if __name__ == "__main__":
    unittest.main()

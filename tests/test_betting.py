import unittest

import numpy as np

import pandas as pd

from src.betting import (
    allocate_match_portfolio,
    assess_bet,
    assess_market,
    evaluate_outright_market,
)
from src.prediction import MatchPrediction


CONFIG = {
    "fractional_kelly": 0.25,
    "max_stake_fraction": 0.02,
    "min_edge": 0.04,
    "min_expected_return": 0.03,
    "max_odds_age_hours": 24.0,
    "min_confidence": 0.45,
    "max_data_uncertainty": 0.55,
}


def prediction(home=0.55, draw=0.25, away=0.20):
    matrix = np.full((2, 2), 0.25)
    return MatchPrediction(
        home_team="A",
        away_team="B",
        expected_home_goals=1.4,
        expected_away_goals=1.0,
        score_matrix=matrix,
        home_win=home,
        draw=draw,
        away_win=away,
        confidence=0.8,
        confidence_label="hoch",
        data_uncertainty=0.1,
        contributions={},
    )


class BettingTests(unittest.TestCase):
    def test_positive_value_uses_capped_fractional_kelly(self):
        result = assess_bet(
            outcome="1",
            model_probability=0.60,
            decimal_odds=2.0,
            implied_probability=0.50,
            confidence=0.8,
            data_uncertainty=0.1,
            bankroll=100.0,
            config=CONFIG,
            odds_age_hours=1.0,
        )
        self.assertEqual(result.decision, "WETTEN")
        self.assertAlmostEqual(result.expected_return, 0.20)
        self.assertAlmostEqual(result.full_kelly_fraction, 0.20)
        self.assertAlmostEqual(result.recommended_fraction, 0.02)
        self.assertAlmostEqual(result.stake, 2.0)

    def test_negative_expected_return_means_no_bet(self):
        result = assess_bet(
            outcome="1",
            model_probability=0.45,
            decimal_odds=2.0,
            implied_probability=0.50,
            confidence=0.8,
            data_uncertainty=0.1,
            bankroll=100.0,
            config=CONFIG,
            odds_age_hours=1.0,
        )
        self.assertEqual(result.decision, "KEINE WETTE")
        self.assertEqual(result.stake, 0.0)

    def test_missing_or_stale_quote_means_no_bet(self):
        missing = assess_bet(
            outcome="X",
            model_probability=0.30,
            decimal_odds=None,
            implied_probability=None,
            confidence=0.8,
            data_uncertainty=0.1,
            bankroll=100.0,
            config=CONFIG,
        )
        stale = assess_bet(
            outcome="X",
            model_probability=0.30,
            decimal_odds=4.0,
            implied_probability=0.25,
            confidence=0.8,
            data_uncertainty=0.1,
            bankroll=100.0,
            config=CONFIG,
            odds_age_hours=25.0,
        )
        self.assertEqual(missing.decision, "KEINE WETTE")
        self.assertEqual(stale.decision, "KEINE WETTE")
        self.assertEqual(stale.stake, 0.0)

    def test_market_selects_value_not_most_likely_outcome(self):
        result = assess_market(
            prediction(),
            decimal_odds={"1": 1.70, "X": 5.0, "2": 4.0},
            implied_probabilities={"1": 0.57, "X": 0.19, "2": 0.24},
            bankroll=100.0,
            config=CONFIG,
            odds_age_hours=1.0,
        )
        self.assertEqual(result.decision, "WETTEN")
        self.assertEqual(result.outcome, "X")
        self.assertAlmostEqual(result.expected_return, 0.25)

    def test_closed_market_never_recommends_a_bet(self):
        result = assess_market(
            prediction(),
            decimal_odds={"1": 2.0, "X": 5.0, "2": 6.0},
            implied_probabilities={"1": 0.50, "X": 0.20, "2": 0.1667},
            bankroll=100.0,
            config=CONFIG,
            odds_age_hours=1.0,
            market_open=False,
        )
        self.assertEqual(result.decision, "KEINE WETTE")
        self.assertEqual(result.stake, 0.0)

    def test_outright_market_requires_sufficient_coverage(self):
        probabilities = pd.DataFrame(
            [
                {"team_id": "A", "team": "A", "champion_probability": 0.60},
                {"team_id": "B", "team": "B", "champion_probability": 0.40},
            ]
        )
        partial_odds = pd.DataFrame(
            [
                {
                    "team_id": "A",
                    "bookmaker": "Swisslos",
                    "market": "champion",
                    "collected_at": pd.Timestamp("2026-06-10 10:00:00"),
                    "decimal_odds": 2.0,
                    "source": "Swisslos",
                }
            ]
        )
        result = evaluate_outright_market(
            probabilities,
            partial_odds,
            bankroll=100.0,
            config={**CONFIG, "outright_min_market_coverage": 0.80},
            as_of=pd.Timestamp("2026-06-10 12:00:00"),
        )
        self.assertTrue((result["decision"] == "KEINE WETTE").all())

    def test_complete_outright_market_can_find_value(self):
        probabilities = pd.DataFrame(
            [
                {"team_id": "A", "team": "A", "champion_probability": 0.60},
                {"team_id": "B", "team": "B", "champion_probability": 0.40},
            ]
        )
        odds = pd.DataFrame(
            [
                {
                    "team_id": "A",
                    "bookmaker": "Swisslos",
                    "market": "champion",
                    "collected_at": pd.Timestamp("2026-06-10 10:00:00"),
                    "decimal_odds": 2.0,
                    "source": "Swisslos",
                },
                {
                    "team_id": "B",
                    "bookmaker": "Swisslos",
                    "market": "champion",
                    "collected_at": pd.Timestamp("2026-06-10 10:00:00"),
                    "decimal_odds": 2.5,
                    "source": "Swisslos",
                },
            ]
        )
        result = evaluate_outright_market(
            probabilities,
            odds,
            bankroll=100.0,
            config={**CONFIG, "outright_min_market_coverage": 0.80},
            as_of=pd.Timestamp("2026-06-10 12:00:00"),
        )
        team_a = result[result["team_id"] == "A"].iloc[0]
        self.assertEqual(team_a["decision"], "WETTEN")
        self.assertGreater(team_a["expected_return"], 0)

    def test_portfolio_respects_daily_and_total_limits(self):
        recommendations = pd.DataFrame(
            [
                {
                    "match_id": f"M{index}",
                    "date": "2026-06-11 18:00",
                    "bet_decision": "WETTEN",
                    "bet_outcome": "1",
                    "bet_stake": 2.0,
                    "bet_expected_return": 0.20 - index * 0.01,
                    "bet_edge": 0.10,
                    "confidence": 0.8,
                    "data_uncertainty": 0.1,
                    "bet_reason": "Value",
                }
                for index in range(5)
            ]
        )
        result = allocate_match_portfolio(
            recommendations,
            bankroll=100.0,
            config={
                "max_total_exposure_fraction": 0.10,
                "max_daily_exposure_fraction": 0.05,
                "minimum_portfolio_stake": 0.50,
                "max_bets_per_day": 3,
            },
        )
        self.assertAlmostEqual(result["bet_stake"].sum(), 5.0)
        self.assertEqual(int((result["bet_decision"] == "WETTEN").sum()), 3)
        self.assertAlmostEqual(result["portfolio_remaining_capacity"].iloc[0], 5.0)

    def test_portfolio_accounts_for_pending_and_duplicate_bets(self):
        recommendations = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "date": "2026-06-11 18:00",
                    "bet_decision": "WETTEN",
                    "bet_outcome": "1",
                    "bet_stake": 2.0,
                    "bet_expected_return": 0.20,
                    "bet_edge": 0.10,
                    "confidence": 0.8,
                    "data_uncertainty": 0.1,
                    "bet_reason": "Value",
                },
                {
                    "match_id": "M2",
                    "date": "2026-06-12 18:00",
                    "bet_decision": "WETTEN",
                    "bet_outcome": "X",
                    "bet_stake": 2.0,
                    "bet_expected_return": 0.18,
                    "bet_edge": 0.08,
                    "confidence": 0.8,
                    "data_uncertainty": 0.1,
                    "bet_reason": "Value",
                },
            ]
        )
        pending = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "outcome": "1",
                    "stake": 4.0,
                    "status": "pending",
                }
            ]
        )
        matches = pd.DataFrame(
            [
                {"match_id": "M1", "date": pd.Timestamp("2026-06-11")},
                {"match_id": "M2", "date": pd.Timestamp("2026-06-12")},
            ]
        )
        result = allocate_match_portfolio(
            recommendations,
            bankroll=50.0,
            config={
                "max_total_exposure_fraction": 0.10,
                "max_daily_exposure_fraction": 0.10,
                "minimum_portfolio_stake": 0.50,
                "max_bets_per_day": 3,
            },
            pending_bets=pending,
            matches=matches,
        )
        duplicate = result[result["match_id"] == "M1"].iloc[0]
        second = result[result["match_id"] == "M2"].iloc[0]
        self.assertEqual(duplicate["bet_decision"], "KEINE WETTE")
        self.assertIn("bereits offen", duplicate["bet_reason"])
        self.assertAlmostEqual(second["bet_stake"], 1.0)
        self.assertAlmostEqual(result["portfolio_pending_stake"].iloc[0], 4.0)


if __name__ == "__main__":
    unittest.main()

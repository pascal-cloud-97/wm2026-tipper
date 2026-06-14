from __future__ import annotations

from dataclasses import dataclass
from math import lgamma

import numpy as np
import pandas as pd

from .calibration import OutcomeCalibrator, calibrate_score_matrix
from .feature_engineering import impute_feature


@dataclass(frozen=True)
class MatchPrediction:
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    score_matrix: np.ndarray
    home_win: float
    draw: float
    away_win: float
    confidence: float
    confidence_label: str
    data_uncertainty: float
    contributions: dict[str, float]
    uncalibrated_home_win: float | None = None
    uncalibrated_draw: float | None = None
    uncalibrated_away_win: float | None = None

    def score_probabilities(self) -> pd.DataFrame:
        rows = [
            {
                "home_goals": home,
                "away_goals": away,
                "score": f"{home}:{away}",
                "probability": float(self.score_matrix[home, away]),
            }
            for home in range(self.score_matrix.shape[0])
            for away in range(self.score_matrix.shape[1])
        ]
        return pd.DataFrame(rows).sort_values("probability", ascending=False)

    def most_likely_outcome(self) -> tuple[str, str, float]:
        outcomes = [
            ("1", self.home_team, self.home_win),
            ("X", "Unentschieden", self.draw),
            ("2", self.away_team, self.away_win),
        ]
        code, label, probability = max(outcomes, key=lambda item: item[2])
        return code, label, float(probability)

    def representative_score(self) -> tuple[str, float]:
        """Most likely score conditional on the most likely 1/X/2 outcome."""
        outcome_code, _, _ = self.most_likely_outcome()
        scores = self.score_probabilities()
        if outcome_code == "1":
            candidates = scores[scores["home_goals"] > scores["away_goals"]]
        elif outcome_code == "2":
            candidates = scores[scores["home_goals"] < scores["away_goals"]]
        else:
            candidates = scores[scores["home_goals"] == scores["away_goals"]]
        row = candidates.iloc[0]
        return str(row["score"]), float(row["probability"])

    def modal_score(self) -> tuple[str, float]:
        row = self.score_probabilities().iloc[0]
        return str(row["score"]), float(row["probability"])


def poisson_score_matrix(
    expected_home_goals: float,
    expected_away_goals: float,
    max_goals: int = 6,
) -> np.ndarray:
    if expected_home_goals <= 0 or expected_away_goals <= 0:
        raise ValueError("Erwartete Tore müssen größer als null sein.")
    if max_goals < 1:
        raise ValueError("max_goals muss mindestens 1 sein.")
    goals = np.arange(max_goals + 1)
    home = np.exp(
        goals * np.log(expected_home_goals)
        - expected_home_goals
        - np.array([lgamma(int(goal) + 1) for goal in goals])
    )
    away = np.exp(
        goals * np.log(expected_away_goals)
        - expected_away_goals
        - np.array([lgamma(int(goal) + 1) for goal in goals])
    )
    matrix = np.outer(home, away)
    # The model explicitly searches 0..max_goals. Renormalization makes that
    # finite state space a proper probability distribution.
    return matrix / matrix.sum()


def outcome_probabilities(matrix: np.ndarray) -> tuple[float, float, float]:
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    return home_win, draw, away_win


def _confidence(
    matrix: np.ndarray,
    outcomes: tuple[float, float, float],
    uncertainty: float,
) -> tuple[float, str]:
    flat = matrix.ravel()
    entropy = -float(np.sum(flat * np.log(flat + 1e-15)))
    normalized_entropy = entropy / np.log(len(flat))
    outcome_certainty = max(outcomes)
    data_quality = 1.0 - np.clip(uncertainty, 0.0, 1.0)
    confidence = np.clip(
        0.42 * data_quality
        + 0.38 * outcome_certainty
        + 0.20 * (1.0 - normalized_entropy),
        0.0,
        1.0,
    )
    label = "hoch" if confidence >= 0.68 else "mittel" if confidence >= 0.48 else "niedrig"
    return float(confidence), label


def predict_match(feature_row: pd.Series | dict, model_config: dict) -> MatchPrediction:
    row = dict(feature_row)
    weights = model_config.get("weights", {})
    elo_rating_diff = row.get("rating_diff")
    if elo_rating_diff is None or not np.isfinite(float(elo_rating_diff)):
        elo_rating_diff = np.nan
    else:
        elo_rating_diff = float(elo_rating_diff)
    fifa_points_diff = row.get("fifa_points_diff")
    fifa_rank_diff = row.get("fifa_rank_diff")
    if fifa_points_diff is not None and np.isfinite(float(fifa_points_diff)):
        fifa_rating_diff = float(fifa_points_diff)
    elif fifa_rank_diff is not None and np.isfinite(float(fifa_rank_diff)):
        fifa_rating_diff = float(fifa_rank_diff) * float(
            model_config.get("fifa_rank_to_rating", 8.0)
        )
    else:
        fifa_rating_diff = np.nan
    fifa_blend_weight = float(
        np.clip(model_config.get("fifa_blend_weight", 0.25), 0.0, 1.0)
    )
    if np.isfinite(elo_rating_diff) and np.isfinite(fifa_rating_diff):
        rating_diff = (
            (1.0 - fifa_blend_weight) * elo_rating_diff
            + fifa_blend_weight * fifa_rating_diff
        )
    elif np.isfinite(elo_rating_diff):
        rating_diff = elo_rating_diff
    else:
        rating_diff = impute_feature(fifa_rating_diff)
    form_diff = impute_feature(row.get("form_diff"))
    form_trend_diff = impute_feature(row.get("form_trend_diff"))
    h2h_diff = impute_feature(row.get("h2h_goal_diff"))
    travel_diff = impute_feature(row.get("travel_diff_1000km"))
    availability_edge = impute_feature(row.get("availability_edge"))
    discipline_edge = impute_feature(row.get("discipline_edge"))
    lineup_diff = impute_feature(row.get("lineup_strength_diff"))
    market_edge = impute_feature(row.get("market_log_odds_edge"))

    home_attack = impute_feature(row.get("home_attack_strength_10"), 1.0)
    away_attack = impute_feature(row.get("away_attack_strength_10"), 1.0)
    home_defense = impute_feature(row.get("home_defense_strength_10"), 1.0)
    away_defense = impute_feature(row.get("away_defense_strength_10"), 1.0)
    attack_defense_edge = np.log(
        max(0.2, home_attack * away_defense)
        / max(0.2, away_attack * home_defense)
    )
    context_edge = (
        impute_feature(row.get("home_host_advantage"))
        - impute_feature(row.get("away_host_advantage"))
        + 0.5
        * (
            impute_feature(row.get("home_continent_advantage"))
            - impute_feature(row.get("away_continent_advantage"))
        )
    )

    contributions = {
        "Rating/FIFA": weights.get("rating", 0.34) * rating_diff / 400.0,
        "Form": weights.get("form", 0.22) * form_diff / 3.0,
        "Formkurve": weights.get("form_trend", 0.06)
        * form_trend_diff
        / 3.0,
        "Angriff/Verteidigung": weights.get("attack_defense", 0.25)
        * attack_defense_edge,
        "Austragungsort": weights.get("home_context", 0.10) * context_edge,
        "Reise": weights.get("travel", 0.04) * travel_diff,
        "Direktduelle": weights.get("head_to_head", 0.05) * h2h_diff,
        "Verfügbarkeit": weights.get("availability", 0.08) * availability_edge,
        "Disziplin/Karten": weights.get("discipline", 0.04)
        * discipline_edge
        / 4.0,
        "Aufstellung": weights.get("lineup", 0.08) * lineup_diff / 10.0,
        "Marktquoten": weights.get("market", 0.12) * market_edge / 2.0,
    }
    total_edge = float(sum(contributions.values()))
    home_base = float(model_config.get("base_home_goals", 1.45))
    away_base = float(model_config.get("base_away_goals", 1.20))
    min_goals = float(model_config.get("min_expected_goals", 0.2))
    max_goals = float(model_config.get("max_expected_goals", 4.5))
    expected_home = float(np.clip(home_base * np.exp(total_edge / 2), min_goals, max_goals))
    expected_away = float(np.clip(away_base * np.exp(-total_edge / 2), min_goals, max_goals))

    matrix = poisson_score_matrix(
        expected_home,
        expected_away,
        max_goals=int(model_config.get("max_goals", 6)),
    )
    raw_outcomes = outcome_probabilities(matrix)
    calibration_config = model_config.get("calibration", {})
    if calibration_config.get("enabled", False):
        matrix = calibrate_score_matrix(
            matrix,
            OutcomeCalibrator.from_dict(calibration_config),
        )
    outcomes = outcome_probabilities(matrix)
    uncertainty = float(np.clip(row.get("data_uncertainty", 1.0), 0.0, 1.0))
    confidence, confidence_label = _confidence(matrix, outcomes, uncertainty)
    return MatchPrediction(
        home_team=str(row.get("home_team_name", row.get("home_team", "Heim"))),
        away_team=str(row.get("away_team_name", row.get("away_team", "Auswärts"))),
        expected_home_goals=expected_home,
        expected_away_goals=expected_away,
        score_matrix=matrix,
        home_win=outcomes[0],
        draw=outcomes[1],
        away_win=outcomes[2],
        confidence=confidence,
        confidence_label=confidence_label,
        data_uncertainty=uncertainty,
        contributions=contributions,
        uncalibrated_home_win=raw_outcomes[0],
        uncalibrated_draw=raw_outcomes[1],
        uncalibrated_away_win=raw_outcomes[2],
    )


def predict_all(features: pd.DataFrame, model_config: dict) -> dict[str, MatchPrediction]:
    return {
        str(row["match_id"]): predict_match(row, model_config)
        for _, row in features.iterrows()
    }

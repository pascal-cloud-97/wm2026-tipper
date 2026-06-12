from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .prediction import MatchPrediction
from .scoring import ScoringRules, score_tip, tendency


@dataclass(frozen=True)
class TipRecommendation:
    tip_home: int
    tip_away: int
    expected_points: float
    standard_deviation: float
    exact_probability: float
    objective: float
    strategy: str
    classification: str
    alternatives: pd.DataFrame

    @property
    def score(self) -> str:
        return f"{self.tip_home}:{self.tip_away}"


def expected_points_for_tip(
    score_matrix: np.ndarray,
    tip_home: int,
    tip_away: int,
    rules: ScoringRules,
    use_joker: bool = False,
) -> tuple[float, float]:
    points = np.zeros_like(score_matrix, dtype=float)
    for actual_home in range(score_matrix.shape[0]):
        for actual_away in range(score_matrix.shape[1]):
            points[actual_home, actual_away] = score_tip(
                tip_home,
                tip_away,
                actual_home,
                actual_away,
                rules,
                use_joker=use_joker,
            )
    expected = float(np.sum(score_matrix * points))
    variance = float(np.sum(score_matrix * (points - expected) ** 2))
    return expected, variance


def evaluate_all_tips(
    prediction: MatchPrediction,
    rules: ScoringRules,
    max_tip_goals: int | None = None,
    use_joker: bool = False,
) -> pd.DataFrame:
    limit = (
        max_tip_goals
        if max_tip_goals is not None
        else prediction.score_matrix.shape[0] - 1
    )
    rows = []
    for home in range(limit + 1):
        for away in range(limit + 1):
            expected, variance = expected_points_for_tip(
                prediction.score_matrix, home, away, rules, use_joker
            )
            exact_probability = (
                float(prediction.score_matrix[home, away])
                if home < prediction.score_matrix.shape[0]
                and away < prediction.score_matrix.shape[1]
                else 0.0
            )
            tip_outcome = tendency(home, away)
            outcome_probability = {
                1: prediction.home_win,
                0: prediction.draw,
                -1: prediction.away_win,
            }[tip_outcome]
            rows.append(
                {
                    "tip_home": home,
                    "tip_away": away,
                    "tip": f"{home}:{away}",
                    "expected_points": expected,
                    "variance": variance,
                    "standard_deviation": np.sqrt(variance),
                    "exact_probability": exact_probability,
                    "outcome_probability": outcome_probability,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["expected_points", "exact_probability"], ascending=False
    )


def optimize_tip(
    prediction: MatchPrediction,
    rules: ScoringRules,
    strategy: str = "safe",
    optimizer_config: dict | None = None,
    use_joker: bool = False,
) -> TipRecommendation:
    config = optimizer_config or {}
    candidates = evaluate_all_tips(prediction, rules, use_joker=use_joker).copy()
    best_ev = float(candidates["expected_points"].max())

    if strategy == "safe":
        candidates["objective"] = candidates["expected_points"]
    elif strategy == "value":
        ratio = float(config.get("value_min_ev_ratio", 0.92))
        candidates = candidates[candidates["expected_points"] >= best_ev * ratio].copy()
        novelty = 1.0 - candidates["exact_probability"]
        weight = float(config.get("value_novelty_weight", 0.18))
        candidates["objective"] = candidates["expected_points"] * (1 + weight * novelty)
    elif strategy == "risk":
        ratio = float(config.get("risk_min_ev_ratio", 0.75))
        candidates = candidates[candidates["expected_points"] >= best_ev * ratio].copy()
        weight = float(config.get("risk_upside_weight", 0.25))
        candidates["objective"] = (
            candidates["expected_points"] + weight * candidates["standard_deviation"]
        )
    else:
        raise ValueError("strategy muss safe, value oder risk sein.")

    candidates = candidates.sort_values(
        ["objective", "expected_points", "standard_deviation"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    winner = candidates.iloc[0]
    score_probabilities = prediction.score_probabilities().reset_index(drop=True)
    probability_rank = score_probabilities.index[
        (score_probabilities["home_goals"] == int(winner["tip_home"]))
        & (score_probabilities["away_goals"] == int(winner["tip_away"]))
    ]
    rank = int(probability_rank[0] + 1) if len(probability_rank) else 999
    classification = (
        "Mainstream" if rank <= 3 else "Value" if rank <= 10 else "Risiko"
    )
    return TipRecommendation(
        tip_home=int(winner["tip_home"]),
        tip_away=int(winner["tip_away"]),
        expected_points=float(winner["expected_points"]),
        standard_deviation=float(winner["standard_deviation"]),
        exact_probability=float(winner["exact_probability"]),
        objective=float(winner["objective"]),
        strategy=strategy,
        classification=classification,
        alternatives=candidates.head(10),
    )


def optimize_all(
    predictions: dict[str, MatchPrediction],
    rules: ScoringRules,
    strategy: str,
    optimizer_config: dict | None = None,
    use_joker: bool = False,
) -> dict[str, TipRecommendation]:
    return {
        match_id: optimize_tip(
            prediction,
            rules,
            strategy=strategy,
            optimizer_config=optimizer_config,
            use_joker=use_joker,
        )
        for match_id, prediction in predictions.items()
    }

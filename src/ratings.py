from __future__ import annotations

import math

import numpy as np
import pandas as pd


def elo_snapshots(
    historical_results: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 24.0,
    home_advantage: float = 55.0,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if historical_results.empty:
        return pd.DataFrame(), {}
    ordered = historical_results.copy()
    ordered["_source_index"] = ordered.index
    ordered = ordered.sort_values(["date", "_source_index"], kind="stable")
    ratings: dict[str, float] = {}
    rows = []

    for row in ordered.to_dict("records"):
        home = str(row["home_team"])
        away = str(row["away_team"])
        home_rating = ratings.get(home, initial_rating)
        away_rating = ratings.get(away, initial_rating)
        neutral = bool(row.get("neutral", False))
        adjusted_home = home_rating + (0.0 if neutral else home_advantage)
        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away_rating - adjusted_home) / 400.0)
        )
        goal_difference = int(row["home_goals"]) - int(row["away_goals"])
        actual_home = 1.0 if goal_difference > 0 else 0.5 if goal_difference == 0 else 0.0
        margin_multiplier = 1.0 + 0.35 * math.log1p(abs(goal_difference))
        change = k_factor * margin_multiplier * (actual_home - expected_home)
        rows.append(
            {
                "history_index": int(row["_source_index"]),
                "date": row["date"],
                "home_team": home,
                "away_team": away,
                "home_rating": float(home_rating),
                "away_rating": float(away_rating),
            }
        )
        ratings[home] = float(home_rating + change)
        ratings[away] = float(away_rating - change)
    return pd.DataFrame(rows), ratings


def current_elo_ratings(
    historical_results: pd.DataFrame,
    initial_rating: float = 1500.0,
    k_factor: float = 24.0,
    home_advantage: float = 55.0,
) -> pd.DataFrame:
    _, ratings = elo_snapshots(
        historical_results,
        initial_rating=initial_rating,
        k_factor=k_factor,
        home_advantage=home_advantage,
    )
    if not ratings:
        return pd.DataFrame(
            columns=["team_id", "as_of", "rating", "source"]
        )
    as_of = pd.Timestamp(historical_results["date"].max()) + pd.Timedelta(seconds=1)
    return pd.DataFrame(
        [
            {
                "team_id": team_id,
                "as_of": as_of,
                "rating": float(rating),
                "source": "Rolling Elo from historical senior internationals",
            }
            for team_id, rating in sorted(ratings.items())
            if np.isfinite(rating)
        ]
    )


def teams_with_current_elo(
    teams: pd.DataFrame,
    historical_results: pd.DataFrame,
) -> pd.DataFrame:
    updated = teams.copy()
    current = current_elo_ratings(historical_results)
    if current.empty:
        return updated
    rating_lookup = current.set_index("team_id")["rating"]
    mapped = updated["team_id"].astype(str).map(rating_lookup)
    if "rating" not in updated:
        updated["rating"] = mapped
    else:
        updated["rating"] = mapped.fillna(updated["rating"])
    return updated

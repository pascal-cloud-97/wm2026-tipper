from __future__ import annotations

import numpy as np
import pandas as pd


def team_form_curve(
    historical_results: pd.DataFrame,
    team_id: str,
    before: pd.Timestamp,
    window: int = 10,
) -> pd.DataFrame:
    if historical_results.empty:
        return pd.DataFrame()
    home = pd.DataFrame(
        {
            "date": historical_results["date"],
            "team": historical_results["home_team"].astype(str),
            "opponent": historical_results["away_team"].astype(str),
            "goals_for": historical_results["home_goals"],
            "goals_against": historical_results["away_goals"],
        }
    )
    away = pd.DataFrame(
        {
            "date": historical_results["date"],
            "team": historical_results["away_team"].astype(str),
            "opponent": historical_results["home_team"].astype(str),
            "goals_for": historical_results["away_goals"],
            "goals_against": historical_results["home_goals"],
        }
    )
    rows = pd.concat([home, away], ignore_index=True).sort_values("date")
    rows = rows[
        (rows["team"] == str(team_id)) & (rows["date"] < pd.Timestamp(before))
    ].tail(window)
    if rows.empty:
        return pd.DataFrame()
    curve = rows.copy()
    curve["points"] = np.select(
        [
            curve["goals_for"] > curve["goals_against"],
            curve["goals_for"] == curve["goals_against"],
        ],
        [3, 1],
        default=0,
    )
    curve["result"] = np.select(
        [
            curve["goals_for"] > curve["goals_against"],
            curve["goals_for"] == curve["goals_against"],
        ],
        ["S", "U"],
        default="N",
    )
    curve["score"] = (
        curve["goals_for"].astype(int).astype(str)
        + ":"
        + curve["goals_against"].astype(int).astype(str)
    )
    curve["rolling_points_5"] = curve["points"].rolling(
        5, min_periods=1
    ).mean()
    return curve.reset_index(drop=True)

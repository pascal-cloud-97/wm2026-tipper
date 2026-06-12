from __future__ import annotations

import pandas as pd

from .data_loader import DataBundle


def _neutral_match(match: pd.Series, teams: pd.DataFrame) -> bool:
    if "venue_country" not in match or pd.isna(match.get("venue_country")):
        return True
    countries = (
        teams.assign(team_id=teams["team_id"].astype(str))
        .set_index("team_id")
        .get("country", pd.Series(dtype=object))
        .to_dict()
    )
    venue = str(match["venue_country"])
    home_country = countries.get(str(match["home_team"]))
    away_country = countries.get(str(match["away_team"]))
    return venue not in {str(home_country), str(away_country)}


def apply_match_results(
    bundle: DataBundle,
    results: pd.DataFrame,
) -> DataBundle:
    if results is None or results.empty:
        return bundle
    matches = bundle.matches.copy()
    matches["match_id"] = matches["match_id"].astype(str)
    latest = (
        results.copy()
        .assign(match_id=lambda frame: frame["match_id"].astype(str))
        .sort_values(["recorded_at", "result_id"])
        .groupby("match_id", as_index=False)
        .tail(1)
    )
    known = set(matches["match_id"])
    latest = latest[latest["match_id"].isin(known)]
    if latest.empty:
        return bundle

    for result in latest.to_dict("records"):
        mask = matches["match_id"] == str(result["match_id"])
        matches.loc[mask, "actual_home_goals"] = int(result["home_goals"])
        matches.loc[mask, "actual_away_goals"] = int(result["away_goals"])
        matches.loc[mask, "status"] = "completed"

    history = bundle.historical_results.copy()
    history["date"] = pd.to_datetime(history["date"])
    additions = []
    for result in latest.to_dict("records"):
        match = matches[matches["match_id"] == str(result["match_id"])].iloc[0]
        match_date = pd.Timestamp(match["date"])
        same_fixture = (
            (history["date"].dt.normalize() == match_date.normalize())
            & (history["home_team"].astype(str) == str(match["home_team"]))
            & (history["away_team"].astype(str) == str(match["away_team"]))
        )
        history = history.loc[~same_fixture].copy()
        additions.append(
            {
                "date": match_date,
                "home_team": str(match["home_team"]),
                "away_team": str(match["away_team"]),
                "home_goals": int(result["home_goals"]),
                "away_goals": int(result["away_goals"]),
                "neutral": _neutral_match(match, bundle.teams),
                "competition": "FIFA World Cup",
                "source": str(result["source"]),
            }
        )
    if additions:
        history = (
            pd.concat([history, pd.DataFrame(additions)], ignore_index=True)
            .sort_values("date")
            .reset_index(drop=True)
        )
    bundle.matches = matches
    bundle.historical_results = history
    return bundle

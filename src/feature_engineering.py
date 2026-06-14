from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _perspective_history(history: pd.DataFrame) -> pd.DataFrame:
    def numeric(column: str) -> pd.Series:
        if column not in history:
            return pd.Series(np.nan, index=history.index, dtype=float)
        return pd.to_numeric(history[column], errors="coerce")

    home = pd.DataFrame(
        {
            "date": history["date"],
            "team": history["home_team"].astype(str),
            "opponent": history["away_team"].astype(str),
            "goals_for": history["home_goals"],
            "goals_against": history["away_goals"],
            "is_home": True,
            "yellow_cards": numeric("home_yellow_cards"),
            "red_cards": numeric("home_red_cards"),
        }
    )
    away = pd.DataFrame(
        {
            "date": history["date"],
            "team": history["away_team"].astype(str),
            "opponent": history["home_team"].astype(str),
            "goals_for": history["away_goals"],
            "goals_against": history["home_goals"],
            "is_home": False,
            "yellow_cards": numeric("away_yellow_cards"),
            "red_cards": numeric("away_red_cards"),
        }
    )
    long = pd.concat([home, away], ignore_index=True)
    long["goal_diff"] = long["goals_for"] - long["goals_against"]
    long["points"] = np.select(
        [long["goal_diff"] > 0, long["goal_diff"] == 0], [3, 1], default=0
    )
    return long.sort_values("date")


def _team_discipline(
    long_history: pd.DataFrame,
    team_id: str,
    before: pd.Timestamp,
    window: int = 5,
) -> dict[str, float]:
    rows = long_history[
        (long_history["team"] == str(team_id))
        & (long_history["date"] < before)
        & long_history["yellow_cards"].notna()
        & long_history["red_cards"].notna()
    ].tail(window)
    if rows.empty:
        return {
            "yellow_cards_5": np.nan,
            "red_cards_5": np.nan,
            "discipline_burden_5": np.nan,
            "discipline_matches_5": 0,
        }
    weights = np.linspace(0.65, 1.0, len(rows))
    yellow = float(np.average(rows["yellow_cards"], weights=weights))
    red = float(np.average(rows["red_cards"], weights=weights))
    return {
        "yellow_cards_5": yellow,
        "red_cards_5": red,
        "discipline_burden_5": yellow + 4.0 * red,
        "discipline_matches_5": int(len(rows)),
    }


def _team_form(
    long_history: pd.DataFrame,
    team_id: str,
    before: pd.Timestamp,
    window: int,
    baseline_goals: float,
) -> dict[str, float]:
    rows = long_history[
        (long_history["team"] == str(team_id)) & (long_history["date"] < before)
    ].tail(window)
    if rows.empty:
        return {
            f"form_points_{window}": np.nan,
            f"goals_for_{window}": np.nan,
            f"goals_against_{window}": np.nan,
            f"goal_diff_{window}": np.nan,
            f"attack_strength_{window}": np.nan,
            f"defense_strength_{window}": np.nan,
            f"matches_{window}": 0,
            f"form_trend_{window}": np.nan,
        }
    weights = np.linspace(0.65, 1.0, len(rows))
    trend = (
        float(np.polyfit(np.arange(len(rows)), rows["points"], 1)[0])
        if len(rows) >= 2
        else np.nan
    )
    return {
        f"form_points_{window}": float(np.average(rows["points"], weights=weights)),
        f"goals_for_{window}": float(
            np.average(rows["goals_for"], weights=weights)
        ),
        f"goals_against_{window}": float(
            np.average(rows["goals_against"], weights=weights)
        ),
        f"goal_diff_{window}": float(
            np.average(rows["goal_diff"], weights=weights)
        ),
        f"attack_strength_{window}": float(
            np.average(rows["goals_for"], weights=weights) / baseline_goals
        ),
        f"defense_strength_{window}": float(
            np.average(rows["goals_against"], weights=weights) / baseline_goals
        ),
        f"matches_{window}": int(len(rows)),
        f"form_trend_{window}": trend,
    }


def team_form_curve(
    historical_results: pd.DataFrame,
    team_id: str,
    before: pd.Timestamp,
    window: int = 10,
) -> pd.DataFrame:
    if historical_results.empty:
        return pd.DataFrame()
    rows = _perspective_history(historical_results)
    rows = rows[
        (rows["team"] == str(team_id)) & (rows["date"] < pd.Timestamp(before))
    ].tail(window)
    if rows.empty:
        return pd.DataFrame()
    curve = rows[
        ["date", "team", "opponent", "goals_for", "goals_against", "points"]
    ].copy()
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


def _h2h(
    long_history: pd.DataFrame,
    home_team: str,
    away_team: str,
    before: pd.Timestamp,
) -> tuple[float, int]:
    rows = long_history[
        (long_history["team"] == str(home_team))
        & (long_history["opponent"] == str(away_team))
        & (long_history["date"] < before)
    ].tail(10)
    if rows.empty:
        return np.nan, 0
    return float(rows["goal_diff"].mean()), int(len(rows))


def _latest_ratings(
    teams: pd.DataFrame, ratings: pd.DataFrame | None
) -> pd.DataFrame:
    base_columns = [column for column in teams.columns if column != "rating"]
    base = teams[base_columns].copy()
    if ratings is None or ratings.empty:
        base["rating"] = pd.to_numeric(
            teams.get("rating", pd.Series(index=teams.index, dtype=float)),
            errors="coerce",
        )
        return base
    latest = (
        ratings.sort_values("as_of")
        .groupby("team_id", as_index=False)
        .tail(1)[["team_id", "rating"]]
    )
    return base.merge(latest, on="team_id", how="left")


def _availability_burden(
    availability: pd.DataFrame | None,
    team_id: str,
    match_id: str,
    kickoff: pd.Timestamp,
) -> tuple[float, int, float]:
    if availability is None or availability.empty:
        return np.nan, 0, np.nan
    rows = availability[
        (availability["team_id"].astype(str) == str(team_id))
        & (availability["as_of"] <= kickoff)
    ].copy()
    if "match_id" in rows:
        scope = rows["match_id"].fillna("").astype(str).str.strip()
        rows = rows[(scope == "") | (scope == str(match_id))]
    if rows.empty:
        return np.nan, 0, np.nan
    rows = rows.sort_values("as_of").groupby("player_name", as_index=False).tail(1)
    status_weight = {
        "out": 1.0,
        "suspended": 1.0,
        "doubtful": 0.65,
        "questionable": 0.35,
        "available": 0.0,
    }
    burden = sum(
        float(row["impact"]) * status_weight.get(str(row["status"]).lower(), 0.0)
        for row in rows.to_dict("records")
    )
    latest = rows["as_of"].max()
    age_days = max(0.0, (kickoff - latest).total_seconds() / 86400)
    return float(burden), int(len(rows)), float(age_days)


def _lineup_strength(
    lineups: pd.DataFrame | None,
    match_id: str,
    team_id: str,
    kickoff: pd.Timestamp,
) -> tuple[float, int, float]:
    if lineups is None or lineups.empty:
        return np.nan, 0, np.nan
    rows = lineups[
        (lineups["match_id"].astype(str) == str(match_id))
        & (lineups["team_id"].astype(str) == str(team_id))
        & (lineups["as_of"] <= kickoff)
    ].copy()
    if rows.empty:
        return np.nan, 0, np.nan
    latest_snapshot = rows["as_of"].max()
    rows = rows[rows["as_of"] == latest_snapshot]
    starters = rows[rows["is_starting"]]
    if starters.empty:
        return np.nan, 0, float(
            max(0.0, (kickoff - latest_snapshot).total_seconds() / 3600)
        )
    if "expected_minutes" in starters and starters["expected_minutes"].notna().any():
        weights = starters["expected_minutes"].fillna(90).clip(lower=1)
    else:
        weights = np.ones(len(starters))
    strength = float(np.average(starters["player_rating"], weights=weights))
    age_hours = max(0.0, (kickoff - latest_snapshot).total_seconds() / 3600)
    return strength, int(len(starters)), float(age_hours)


def _market_probabilities(
    odds: pd.DataFrame | None,
    match_id: str,
    kickoff: pd.Timestamp,
) -> tuple[float, float, float, int, float]:
    if odds is None or odds.empty:
        return np.nan, np.nan, np.nan, 0, np.nan
    rows = odds[
        (odds["match_id"].astype(str) == str(match_id))
        & (odds["collected_at"] <= kickoff)
    ].copy()
    if rows.empty:
        return np.nan, np.nan, np.nan, 0, np.nan
    latest = rows["collected_at"].max()
    rows = rows[rows["collected_at"] == latest]
    implied = pd.DataFrame(
        {
            "home": 1.0 / rows["home_odds"],
            "draw": 1.0 / rows["draw_odds"],
            "away": 1.0 / rows["away_odds"],
        }
    )
    normalized = implied.div(implied.sum(axis=1), axis=0)
    consensus = normalized.mean()
    age_hours = max(0.0, (kickoff - latest).total_seconds() / 3600)
    return (
        float(consensus["home"]),
        float(consensus["draw"]),
        float(consensus["away"]),
        int(len(rows)),
        float(age_hours),
    )


def _bookmaker_snapshot(
    odds: pd.DataFrame | None,
    match_id: str,
    kickoff: pd.Timestamp,
    bookmaker: str,
) -> dict[str, float]:
    empty = {
        "home_odds": np.nan,
        "draw_odds": np.nan,
        "away_odds": np.nan,
        "home_probability": np.nan,
        "draw_probability": np.nan,
        "away_probability": np.nan,
        "margin": np.nan,
        "age_hours": np.nan,
        "collected_at": pd.NaT,
    }
    if odds is None or odds.empty:
        return empty
    rows = odds[
        (odds["match_id"].astype(str) == str(match_id))
        & (odds["bookmaker"].astype(str).str.lower() == bookmaker.lower())
        & (odds["collected_at"] <= kickoff)
    ].copy()
    if rows.empty:
        return empty
    row = rows.sort_values("collected_at").iloc[-1]
    raw = np.array(
        [row["home_odds"], row["draw_odds"], row["away_odds"]],
        dtype=float,
    )
    implied = 1.0 / raw
    margin = float(implied.sum() - 1.0)
    normalized = implied / implied.sum()
    age_hours = max(
        0.0,
        (kickoff - row["collected_at"]).total_seconds() / 3600,
    )
    return {
        "home_odds": float(raw[0]),
        "draw_odds": float(raw[1]),
        "away_odds": float(raw[2]),
        "home_probability": float(normalized[0]),
        "draw_probability": float(normalized[1]),
        "away_probability": float(normalized[2]),
        "margin": margin,
        "age_hours": float(age_hours),
        "collected_at": row["collected_at"],
    }


def build_match_features(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    historical_results: pd.DataFrame,
    ratings: pd.DataFrame | None = None,
    availability: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    odds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build transparent pre-match features using history before kickoff only."""
    if historical_results.empty:
        long_history = pd.DataFrame(
            columns=[
                "date",
                "team",
                "opponent",
                "goals_for",
                "goals_against",
                "goal_diff",
                "points",
                "yellow_cards",
                "red_cards",
            ]
        )
    else:
        long_history = _perspective_history(historical_results)

    team_frame = _latest_ratings(teams, ratings).copy()
    team_frame["team_id"] = team_frame["team_id"].astype(str)
    team_lookup = team_frame.set_index("team_id").to_dict("index")
    records: list[dict] = []

    for match in matches.to_dict("records"):
        kickoff_value = match.get("kickoff_utc")
        kickoff = (
            pd.Timestamp(kickoff_value)
            if pd.notna(kickoff_value)
            else pd.Timestamp(match["date"])
        )
        prior_results = historical_results[historical_results["date"] < kickoff]
        baseline_goals = (
            max(
                0.4,
                float(
                    prior_results[["home_goals", "away_goals"]]
                    .stack()
                    .mean()
                ),
            )
            if not prior_results.empty
            else 1.35
        )
        home_id = str(match["home_team"])
        away_id = str(match["away_team"])
        home_meta = team_lookup.get(home_id, {})
        away_meta = team_lookup.get(away_id, {})
        row = dict(match)
        row["home_team_name"] = home_meta.get("team_name", home_id)
        row["away_team_name"] = away_meta.get("team_name", away_id)
        explicit_home_rating = match.get("home_rating", np.nan)
        explicit_away_rating = match.get("away_rating", np.nan)
        row["home_rating"] = (
            explicit_home_rating
            if pd.notna(explicit_home_rating)
            else home_meta.get("rating", np.nan)
        )
        row["away_rating"] = (
            explicit_away_rating
            if pd.notna(explicit_away_rating)
            else away_meta.get("rating", np.nan)
        )
        row["rating_diff"] = (
            row["home_rating"] - row["away_rating"]
            if pd.notna(row["home_rating"]) and pd.notna(row["away_rating"])
            else np.nan
        )
        home_rank = home_meta.get("fifa_rank", np.nan)
        away_rank = away_meta.get("fifa_rank", np.nan)
        row["fifa_rank_diff"] = (
            away_rank - home_rank
            if pd.notna(home_rank) and pd.notna(away_rank)
            else np.nan
        )
        home_fifa_points = home_meta.get("fifa_points", np.nan)
        away_fifa_points = away_meta.get("fifa_points", np.nan)
        row["fifa_points_diff"] = (
            home_fifa_points - away_fifa_points
            if pd.notna(home_fifa_points) and pd.notna(away_fifa_points)
            else np.nan
        )

        for window in (5, 10):
            for prefix, team_id in (("home", home_id), ("away", away_id)):
                form = _team_form(
                    long_history, team_id, kickoff, window, baseline_goals
                )
                row.update({f"{prefix}_{key}": value for key, value in form.items()})

        for prefix, team_id in (("home", home_id), ("away", away_id)):
            discipline = _team_discipline(long_history, team_id, kickoff)
            row.update(
                {f"{prefix}_{key}": value for key, value in discipline.items()}
            )
        row["discipline_edge"] = (
            row["away_discipline_burden_5"]
            - row["home_discipline_burden_5"]
            if pd.notna(row["home_discipline_burden_5"])
            and pd.notna(row["away_discipline_burden_5"])
            else np.nan
        )

        row["form_diff"] = (
            row["home_form_points_5"] - row["away_form_points_5"]
            if pd.notna(row["home_form_points_5"])
            and pd.notna(row["away_form_points_5"])
            else np.nan
        )
        row["form_trend_diff"] = (
            row["home_form_trend_5"] - row["away_form_trend_5"]
            if pd.notna(row["home_form_trend_5"])
            and pd.notna(row["away_form_trend_5"])
            else np.nan
        )
        row["h2h_goal_diff"], row["h2h_matches"] = _h2h(
            long_history, home_id, away_id, kickoff
        )

        venue_continent = match.get("venue_continent")
        row["home_continent_advantage"] = float(
            bool(venue_continent)
            and home_meta.get("continent") == venue_continent
            and away_meta.get("continent") != venue_continent
        )
        row["away_continent_advantage"] = float(
            bool(venue_continent)
            and away_meta.get("continent") == venue_continent
            and home_meta.get("continent") != venue_continent
        )
        venue_country = match.get("venue_country")
        row["home_host_advantage"] = float(
            bool(venue_country) and home_meta.get("country") == venue_country
        )
        row["away_host_advantage"] = float(
            bool(venue_country) and away_meta.get("country") == venue_country
        )
        home_travel = match.get("home_travel_km", np.nan)
        away_travel = match.get("away_travel_km", np.nan)
        row["travel_diff_1000km"] = (
            (away_travel - home_travel) / 1000.0
            if pd.notna(home_travel) and pd.notna(away_travel)
            else np.nan
        )

        home_burden, home_reports, home_availability_age = _availability_burden(
            availability, home_id, str(match["match_id"]), kickoff
        )
        away_burden, away_reports, away_availability_age = _availability_burden(
            availability, away_id, str(match["match_id"]), kickoff
        )
        row["home_availability_burden"] = home_burden
        row["away_availability_burden"] = away_burden
        row["availability_edge"] = (
            (away_burden if pd.notna(away_burden) else 0.0)
            - (home_burden if pd.notna(home_burden) else 0.0)
            if pd.notna(home_burden) or pd.notna(away_burden)
            else np.nan
        )
        row["home_availability_reports"] = home_reports
        row["away_availability_reports"] = away_reports
        row["availability_age_days"] = (
            max(home_availability_age, away_availability_age)
            if pd.notna(home_availability_age)
            and pd.notna(away_availability_age)
            else np.nan
        )

        home_lineup, home_starters, home_lineup_age = _lineup_strength(
            lineups, str(match["match_id"]), home_id, kickoff
        )
        away_lineup, away_starters, away_lineup_age = _lineup_strength(
            lineups, str(match["match_id"]), away_id, kickoff
        )
        row["home_lineup_strength"] = home_lineup
        row["away_lineup_strength"] = away_lineup
        row["lineup_strength_diff"] = (
            home_lineup - away_lineup
            if pd.notna(home_lineup) and pd.notna(away_lineup)
            else np.nan
        )
        row["home_lineup_starters"] = home_starters
        row["away_lineup_starters"] = away_starters
        row["lineup_age_hours"] = (
            max(home_lineup_age, away_lineup_age)
            if pd.notna(home_lineup_age) and pd.notna(away_lineup_age)
            else np.nan
        )

        market_home, market_draw, market_away, books, odds_age = (
            _market_probabilities(odds, str(match["match_id"]), kickoff)
        )
        row["market_home_probability"] = market_home
        row["market_draw_probability"] = market_draw
        row["market_away_probability"] = market_away
        row["market_bookmakers"] = books
        row["odds_age_hours"] = odds_age
        row["market_log_odds_edge"] = (
            float(np.log(max(market_home, 1e-6) / max(market_away, 1e-6)))
            if pd.notna(market_home) and pd.notna(market_away)
            else np.nan
        )
        swisslos = _bookmaker_snapshot(
            odds, str(match["match_id"]), kickoff, "Swisslos"
        )
        row.update(
            {
                f"swisslos_{key}": value
                for key, value in swisslos.items()
            }
        )

        quality_checks = [
            pd.notna(row["rating_diff"]) or pd.notna(row["fifa_rank_diff"]),
            row["home_matches_5"] >= 5,
            row["away_matches_5"] >= 5,
            row["home_matches_10"] >= 8,
            row["away_matches_10"] >= 8,
            row["h2h_matches"] >= 2,
            pd.notna(row["travel_diff_1000km"]),
            home_reports > 0
            and away_reports > 0
            and pd.notna(row["availability_age_days"])
            and row["availability_age_days"] <= 7,
            home_starters >= 7
            and away_starters >= 7
            and pd.notna(row["lineup_age_hours"])
            and row["lineup_age_hours"] <= 48,
            books > 0
            and pd.notna(row["odds_age_hours"])
            and row["odds_age_hours"] <= 72,
        ]
        row["data_uncertainty"] = 1.0 - sum(quality_checks) / len(quality_checks)
        row["baseline_goals"] = baseline_goals
        records.append(row)

    return pd.DataFrame(records)


def impute_feature(value: float, default: float = 0.0) -> float:
    return default if value is None or not math.isfinite(float(value)) else float(value)

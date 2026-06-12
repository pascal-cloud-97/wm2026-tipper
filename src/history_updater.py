from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests


SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master/results.csv"
)
TEAM_IDS = {
    "Mexico": "MEX",
    "South Africa": "RSA",
    "South Korea": "KOR",
    "Czech Republic": "CZE",
    "Canada": "CAN",
    "Bosnia and Herzegovina": "BIH",
    "Qatar": "QAT",
    "Switzerland": "SUI",
    "Brazil": "BRA",
    "Morocco": "MAR",
    "Haiti": "HAI",
    "Scotland": "SCO",
    "United States": "USA",
    "Paraguay": "PAR",
    "Australia": "AUS",
    "Turkey": "TUR",
    "Germany": "GER",
    "Curaçao": "CUW",
    "Ivory Coast": "CIV",
    "Ecuador": "ECU",
    "Netherlands": "NED",
    "Japan": "JPN",
    "Sweden": "SWE",
    "Tunisia": "TUN",
    "Belgium": "BEL",
    "Egypt": "EGY",
    "Iran": "IRN",
    "New Zealand": "NZL",
    "Spain": "ESP",
    "Cape Verde": "CPV",
    "Saudi Arabia": "KSA",
    "Uruguay": "URU",
    "France": "FRA",
    "Senegal": "SEN",
    "Iraq": "IRQ",
    "Norway": "NOR",
    "Argentina": "ARG",
    "Algeria": "ALG",
    "Austria": "AUT",
    "Jordan": "JOR",
    "Portugal": "POR",
    "DR Congo": "COD",
    "Uzbekistan": "UZB",
    "Colombia": "COL",
    "England": "ENG",
    "Croatia": "CRO",
    "Ghana": "GHA",
    "Panama": "PAN",
}


def external_id(name: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
    return f"EXT_{slug}"


def download_history(since: str, as_of: str) -> pd.DataFrame:
    response = requests.get(SOURCE_URL, timeout=90)
    response.raise_for_status()
    source = pd.read_csv(pd.io.common.BytesIO(response.content))
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["home_score"] = pd.to_numeric(source["home_score"], errors="coerce")
    source["away_score"] = pd.to_numeric(source["away_score"], errors="coerce")
    relevant_names = set(TEAM_IDS)
    mask = (
        source["date"].between(pd.Timestamp(since), pd.Timestamp(as_of))
        & source["home_score"].notna()
        & source["away_score"].notna()
        & (
            source["home_team"].isin(relevant_names)
            | source["away_team"].isin(relevant_names)
        )
    )
    selected = source.loc[mask].copy()
    selected["home_team"] = selected["home_team"].map(
        lambda name: TEAM_IDS.get(name, external_id(name))
    )
    selected["away_team"] = selected["away_team"].map(
        lambda name: TEAM_IDS.get(name, external_id(name))
    )
    return pd.DataFrame(
        {
            "date": selected["date"].dt.strftime("%Y-%m-%d"),
            "home_team": selected["home_team"],
            "away_team": selected["away_team"],
            "home_goals": selected["home_score"].astype(int),
            "away_goals": selected["away_score"].astype(int),
            "neutral": selected["neutral"],
            "competition": selected["tournament"],
            "source": "martj42/international_results (CC0)",
        }
    ).sort_values("date")


def write_history(frame: pd.DataFrame, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")


def sync_completed_matches(
    history: pd.DataFrame,
    matches_path: str | Path,
) -> int:
    path = Path(matches_path)
    matches = pd.read_csv(path)
    matches["match_date"] = pd.to_datetime(matches["date"]).dt.strftime("%Y-%m-%d")
    if "actual_home_goals" not in matches:
        matches["actual_home_goals"] = pd.NA
    if "actual_away_goals" not in matches:
        matches["actual_away_goals"] = pd.NA

    lookup = {
        (str(row["date"]), str(row["home_team"]), str(row["away_team"])): (
            int(row["home_goals"]),
            int(row["away_goals"]),
        )
        for row in history.to_dict("records")
        if str(row.get("competition", "")).lower() == "fifa world cup"
    }
    updated = 0
    for index, match in matches.iterrows():
        key = (
            str(match["match_date"]),
            str(match["home_team"]),
            str(match["away_team"]),
        )
        if key not in lookup:
            continue
        home_goals, away_goals = lookup[key]
        matches.at[index, "actual_home_goals"] = home_goals
        matches.at[index, "actual_away_goals"] = away_goals
        matches.at[index, "status"] = "completed"
        updated += 1
    matches = matches.drop(columns=["match_date"])
    matches.to_csv(path, index=False, encoding="utf-8")
    return updated


def update_history(
    output: str | Path,
    since: str,
    as_of: str,
    matches_path: str | Path | None = None,
) -> tuple[pd.DataFrame, int]:
    frame = download_history(since, as_of)
    write_history(frame, output)
    completed = (
        sync_completed_matches(frame, matches_path)
        if matches_path is not None
        else 0
    )
    return frame, completed


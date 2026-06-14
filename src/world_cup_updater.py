from __future__ import annotations

import csv
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/"
    "fifa.world/scoreboard"
)
SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/soccer/"
    "fifa.world/summary"
)
SOURCE_TEMPLATE = "https://www.espn.com/soccer/match/_/gameId/{event_id}"

TEAM_ALIASES = {
    "bosniaherzegovina": "BIH",
    "czechia": "CZE",
    "curacao": "CUW",
    "ivorycoast": "CIV",
    "southkorea": "KOR",
    "turkiye": "TUR",
    "unitedstates": "USA",
}


def _normalize(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", str(value)).encode(
        "ascii", "ignore"
    )
    return re.sub(r"[^a-z0-9]+", "", ascii_value.decode("ascii").lower())


def _team_lookup(teams: pd.DataFrame) -> dict[str, str]:
    lookup = dict(TEAM_ALIASES)
    for row in teams.to_dict("records"):
        for value in (row.get("team_name"), row.get("country")):
            if pd.notna(value):
                lookup[_normalize(str(value))] = str(row["team_id"])
    return lookup


def parse_match_minute(display_value: str) -> int:
    numbers = [int(value) for value in re.findall(r"\d+", str(display_value))]
    if not numbers:
        raise ValueError(f"Ungültige Spielminute: {display_value}")
    return min(130, sum(numbers))


def _request_json(
    session: requests.Session,
    url: str,
    params: dict[str, str],
) -> dict:
    response = session.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_completed_world_cup_matches(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    as_of: str | date,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = session or requests.Session()
    cutoff = pd.Timestamp(as_of).date()
    first_date = pd.to_datetime(matches["kickoff_utc"]).min().date()
    lookup = _team_lookup(teams)
    fixtures = {
        (str(row["home_team"]), str(row["away_team"])): row
        for row in matches.to_dict("records")
    }
    result_rows: list[dict] = []
    event_rows: list[dict] = []

    current = first_date
    while current <= cutoff:
        payload = _request_json(
            client,
            SCOREBOARD_URL,
            {"dates": current.strftime("%Y%m%d")},
        )
        for event in payload.get("events", []):
            if event.get("status", {}).get("type", {}).get("name") != (
                "STATUS_FULL_TIME"
            ):
                continue
            competition = event["competitions"][0]
            competitors = competition.get("competitors", [])
            by_side = {item["homeAway"]: item for item in competitors}
            if "home" not in by_side or "away" not in by_side:
                continue
            home_name = by_side["home"]["team"]["displayName"]
            away_name = by_side["away"]["team"]["displayName"]
            home_id = lookup.get(_normalize(home_name))
            away_id = lookup.get(_normalize(away_name))
            fixture = fixtures.get((str(home_id), str(away_id)))
            if fixture is None:
                continue

            event_id = str(event["id"])
            source = SOURCE_TEMPLATE.format(event_id=event_id)
            summary = _request_json(
                client,
                SUMMARY_URL,
                {"event": event_id},
            )
            player_meta: dict[tuple[str, str], dict] = {}
            espn_team_ids: dict[str, str] = {}
            for roster in summary.get("rosters", []):
                team_name = roster["team"]["displayName"]
                team_id = lookup.get(_normalize(team_name))
                espn_team_id = str(roster["team"]["id"])
                if team_id is None:
                    continue
                espn_team_ids[espn_team_id] = team_id
                for player in roster.get("roster", []):
                    name = player["athlete"]["displayName"]
                    player_meta[(team_id, name)] = {
                        "is_starter": bool(player.get("starter")),
                        "position": player.get("position", {}).get(
                            "displayName", ""
                        ),
                    }

            card_counts = {
                home_id: {"yellow_card": 0, "red_card": 0},
                away_id: {"yellow_card": 0, "red_card": 0},
            }
            for detail in competition.get("details", []):
                event_type = (
                    "red_card"
                    if detail.get("redCard")
                    else "yellow_card"
                    if detail.get("yellowCard")
                    else None
                )
                if event_type is None:
                    continue
                involved = detail.get("athletesInvolved", [])
                if not involved:
                    continue
                player_name = involved[0].get("displayName", "Unbekannt")
                team_id = espn_team_ids.get(str(detail.get("team", {}).get("id")))
                if team_id not in card_counts:
                    continue
                card_counts[team_id][event_type] += 1
                meta = player_meta.get((team_id, player_name), {})
                event_rows.append(
                    {
                        "match_id": str(fixture["match_id"]),
                        "team_id": team_id,
                        "player_name": player_name,
                        "event_type": event_type,
                        "minute": parse_match_minute(
                            detail.get("clock", {}).get("displayValue", "")
                        ),
                        "is_starter": bool(meta.get("is_starter", False)),
                        "position": str(meta.get("position", "")),
                        "source": source,
                    }
                )

            result_rows.append(
                {
                    "match_id": str(fixture["match_id"]),
                    "event_id": event_id,
                    "date": pd.Timestamp(fixture["date"]),
                    "kickoff_utc": pd.Timestamp(fixture["kickoff_utc"]),
                    "home_team": home_id,
                    "away_team": away_id,
                    "home_goals": int(by_side["home"]["score"]),
                    "away_goals": int(by_side["away"]["score"]),
                    "home_yellow_cards": card_counts[home_id]["yellow_card"],
                    "away_yellow_cards": card_counts[away_id]["yellow_card"],
                    "home_red_cards": card_counts[home_id]["red_card"],
                    "away_red_cards": card_counts[away_id]["red_card"],
                    "source": source,
                }
            )
        current += timedelta(days=1)

    results = pd.DataFrame(result_rows)
    events = pd.DataFrame(event_rows)
    if not results.empty:
        results = results.sort_values("kickoff_utc").drop_duplicates(
            "match_id", keep="last"
        )
    if not events.empty:
        events = events.sort_values(["match_id", "minute", "team_id"])
    return results.reset_index(drop=True), events.reset_index(drop=True)


def _neutral_match(match: pd.Series, teams: pd.DataFrame) -> bool:
    countries = teams.set_index("team_id")["country"].astype(str).to_dict()
    venue = str(match.get("venue_country", ""))
    return venue not in {
        countries.get(str(match["home_team"]), ""),
        countries.get(str(match["away_team"]), ""),
    }


def _write_sparse_history(history: pd.DataFrame, path: Path) -> None:
    base_columns = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "neutral",
        "competition",
        "source",
    ]
    card_columns = [
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(base_columns + card_columns)
        for row in history.to_dict("records"):
            values = [row.get(column, "") for column in base_columns]
            cards = [row.get(column, pd.NA) for column in card_columns]
            if any(pd.notna(value) for value in cards):
                values.extend(
                    int(value) if pd.notna(value) else "" for value in cards
                )
            writer.writerow(values)


def update_world_cup_files(
    data_directory: str | Path,
    as_of: str | date,
    session: requests.Session | None = None,
) -> dict[str, int]:
    root = Path(data_directory)
    matches_path = root / "matches.csv"
    history_path = root / "historical_results.csv"
    events_path = root / "match_events.csv"
    availability_path = root / "availability.csv"
    teams = pd.read_csv(root / "teams.csv")
    matches = pd.read_csv(matches_path)
    history = pd.read_csv(history_path)
    availability = pd.read_csv(availability_path)

    results, events = fetch_completed_world_cup_matches(
        matches,
        teams,
        as_of,
        session=session,
    )
    if results.empty:
        return {"completed_matches": 0, "card_events": 0, "suspensions": 0}

    result_lookup = results.set_index("match_id").to_dict("index")
    for index, match in matches.iterrows():
        result = result_lookup.get(str(match["match_id"]))
        if result is None:
            continue
        matches.at[index, "status"] = "completed"
        matches.at[index, "actual_home_goals"] = result["home_goals"]
        matches.at[index, "actual_away_goals"] = result["away_goals"]

    completed_ids = set(results["match_id"].astype(str))
    completed_fixtures = matches[matches["match_id"].astype(str).isin(completed_ids)]
    fixture_keys = {
        (
            pd.Timestamp(row["date"]).normalize(),
            str(row["home_team"]),
            str(row["away_team"]),
        )
        for row in completed_fixtures.to_dict("records")
    }
    history_dates = pd.to_datetime(history["date"], format="mixed")
    keep = [
        (
            timestamp.normalize(),
            str(row["home_team"]),
            str(row["away_team"]),
        )
        not in fixture_keys
        for timestamp, row in zip(history_dates, history.to_dict("records"))
    ]
    history = history.loc[keep].copy()
    history_columns = [
        "date",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "neutral",
        "competition",
        "source",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_red_cards",
        "away_red_cards",
    ]
    for column in history_columns:
        if column not in history:
            history[column] = pd.NA
    additions = []
    match_lookup = matches.set_index("match_id")
    for result in results.to_dict("records"):
        fixture = match_lookup.loc[str(result["match_id"])]
        additions.append(
            {
                "date": pd.Timestamp(fixture["date"]).isoformat(),
                "home_team": result["home_team"],
                "away_team": result["away_team"],
                "home_goals": result["home_goals"],
                "away_goals": result["away_goals"],
                "neutral": _neutral_match(fixture, teams),
                "competition": "FIFA World Cup",
                "source": result["source"],
                "home_yellow_cards": result["home_yellow_cards"],
                "away_yellow_cards": result["away_yellow_cards"],
                "home_red_cards": result["home_red_cards"],
                "away_red_cards": result["away_red_cards"],
            }
        )
    history = pd.concat(
        [history[history_columns], pd.DataFrame(additions)],
        ignore_index=True,
    )
    history["_date"] = pd.to_datetime(history["date"], format="mixed")
    history = history.sort_values("_date", kind="stable").drop(columns="_date")

    existing_events = (
        pd.read_csv(events_path)
        if events_path.exists()
        else pd.DataFrame(columns=events.columns)
    )
    event_keys = ["match_id", "team_id", "player_name", "event_type", "minute"]
    all_events = (
        pd.concat([existing_events, events], ignore_index=True)
        .drop_duplicates(event_keys, keep="last")
        .sort_values(["match_id", "minute", "team_id"])
    )

    source_column = availability.get(
        "source", pd.Series("", index=availability.index)
    )
    espn_mask = source_column.astype(str).str.contains(
        "espn.com/soccer/match", case=False, na=False
    )
    availability = availability.loc[~espn_mask].copy()
    suspensions = []
    ordered_matches = matches.copy()
    ordered_matches["_kickoff"] = pd.to_datetime(ordered_matches["kickoff_utc"])
    result_kickoffs = results.set_index("match_id")["kickoff_utc"].to_dict()
    for card in all_events[
        all_events["event_type"].astype(str) == "red_card"
    ].to_dict("records"):
        source_kickoff = pd.Timestamp(result_kickoffs.get(str(card["match_id"])))
        future = ordered_matches[
            (
                (ordered_matches["home_team"].astype(str) == str(card["team_id"]))
                | (
                    ordered_matches["away_team"].astype(str)
                    == str(card["team_id"])
                )
            )
            & (ordered_matches["_kickoff"] > source_kickoff)
        ].sort_values("_kickoff")
        if future.empty:
            continue
        next_match_id = str(future.iloc[0]["match_id"])
        starter = bool(card.get("is_starter", False))
        suspensions.append(
            {
                "team_id": str(card["team_id"]),
                "player_name": str(card["player_name"]),
                "status": "suspended",
                "impact": 0.65 if starter else 0.35,
                "position": str(card.get("position", "")),
                "as_of": source_kickoff.isoformat(),
                "match_id": next_match_id,
                "impact_basis": (
                    "starter suspension heuristic"
                    if starter
                    else "substitute suspension heuristic"
                ),
                "source": str(card["source"]),
            }
        )
    availability = pd.concat(
        [availability, pd.DataFrame(suspensions)],
        ignore_index=True,
    )

    for column in ("actual_home_goals", "actual_away_goals"):
        matches[column] = pd.to_numeric(
            matches[column], errors="coerce"
        ).astype("Int64")
    matches.drop(columns=["_kickoff"], errors="ignore").to_csv(
        matches_path, index=False, encoding="utf-8"
    )
    _write_sparse_history(history, history_path)
    all_events.to_csv(events_path, index=False, encoding="utf-8")
    availability.to_csv(availability_path, index=False, encoding="utf-8")
    return {
        "completed_matches": int(len(results)),
        "card_events": int(len(events)),
        "suspensions": int(len(suspensions)),
    }

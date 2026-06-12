from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import requests


RANKING_PAGE_URL = "https://inside.fifa.com/fifa-world-ranking/men"
RANKING_API_URL = (
    "https://api.fifa.com/api/v3/fifarankings/rankings/"
    "rankingsbyschedule"
)


def latest_ranking_metadata(html: str) -> tuple[str, str]:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("FIFA-Seite enthält keine lesbaren Ranglistenmetadaten.")
    payload = json.loads(match.group(1))
    ranking = payload["props"]["pageProps"]["pageData"]["ranking"]
    latest = ranking["allAvailableDates"][0]
    return str(latest["id"]), str(latest["date"])


def parse_ranking_payload(payload: dict, as_of: str) -> pd.DataFrame:
    rows = []
    for item in payload.get("Results", []):
        team_names = item.get("TeamName") or []
        name = next(
            (
                value.get("Description")
                for value in team_names
                if value.get("Description")
            ),
            item.get("IdCountry"),
        )
        rows.append(
            {
                "team_id": str(item["IdCountry"]),
                "team_name": str(name),
                "as_of": as_of,
                "fifa_rank": int(item["Rank"]),
                "fifa_points": float(item["TotalPoints"]),
                "rating": float(item["TotalPoints"]),
                "source": "Official FIFA/Coca-Cola Men's World Ranking",
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("Die FIFA-Schnittstelle lieferte keine Ranglistenwerte.")
    if frame["team_id"].duplicated().any():
        raise ValueError("Die FIFA-Rangliste enthält doppelte Ländercodes.")
    return frame.sort_values("fifa_rank").reset_index(drop=True)


def fetch_latest_fifa_rankings(
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, str]:
    client = session or requests.Session()
    headers = {"User-Agent": "WM2026-Tipper/1.0"}
    page = client.get(
        RANKING_PAGE_URL,
        headers=headers,
        timeout=timeout,
    )
    page.raise_for_status()
    schedule_id, as_of = latest_ranking_metadata(page.text)
    response = client.get(
        RANKING_API_URL,
        params={
            "rankingScheduleId": schedule_id,
            "language": "en",
        },
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_ranking_payload(response.json(), as_of), schedule_id


def update_local_fifa_rankings(
    teams_path: str | Path,
    ratings_path: str | Path,
) -> tuple[pd.DataFrame, str]:
    teams_file = Path(teams_path)
    ratings_file = Path(ratings_path)
    rankings, schedule_id = fetch_latest_fifa_rankings()

    teams = pd.read_csv(teams_file)
    missing = sorted(set(teams["team_id"].astype(str)) - set(rankings["team_id"]))
    if missing:
        raise ValueError(
            "In der offiziellen FIFA-Rangliste fehlen WM-Teams: "
            + ", ".join(missing)
        )
    current = rankings.set_index("team_id")
    teams["fifa_rank"] = (
        teams["team_id"].astype(str).map(current["fifa_rank"]).astype(int)
    )
    teams["fifa_points"] = teams["team_id"].astype(str).map(
        current["fifa_points"]
    )
    teams["rating"] = teams["fifa_points"]

    ratings = pd.read_csv(ratings_file)
    if "fifa_points" not in ratings:
        ratings["fifa_points"] = pd.NA
    selected_rankings = rankings[
        rankings["team_id"].isin(set(teams["team_id"].astype(str)))
    ].copy()
    official_rows = selected_rankings[
        ["team_id", "as_of", "rating", "fifa_rank", "fifa_points", "source"]
    ]
    ratings = ratings[
        ratings["source"].astype(str)
        != "Official FIFA/Coca-Cola Men's World Ranking"
    ]
    ratings = pd.concat(
        [ratings.dropna(axis=1, how="all"), official_rows],
        ignore_index=True,
    )
    ratings = (
        ratings.drop_duplicates(["team_id", "as_of", "source"], keep="last")
        .sort_values(["as_of", "team_id"], kind="stable")
        .reset_index(drop=True)
    )

    teams.to_csv(teams_file, index=False, encoding="utf-8")
    ratings.to_csv(ratings_file, index=False, encoding="utf-8")
    return selected_rankings, schedule_id

from __future__ import annotations

import pandas as pd


CITY_TO_VENUE = {
    "VANCOUVER": "BC Place",
    "SEATTLE": "Lumen Field",
    "SAN FRANCISCO BAY AREA": "Levi's Stadium",
    "LOS ANGELES": "SoFi Stadium",
    "GUADALAJARA": "Estadio Akron",
    "MEXICO CITY": "Estadio Azteca",
    "MONTERREY": "Estadio BBVA",
    "HOUSTON": "NRG Stadium",
    "DALLAS": "AT&T Stadium",
    "KANSAS CITY": "Arrowhead Stadium",
    "ATLANTA": "Mercedes-Benz Stadium",
    "MIAMI": "Hard Rock Stadium",
    "TORONTO": "BMO Field",
    "BOSTON": "Gillette Stadium",
    "PHILADELPHIA": "Lincoln Financial Field",
    "NEW YORK NEW JERSEY": "MetLife Stadium",
}

CITY_ET_OFFSETS = {
    "VANCOUVER": -3,
    "SEATTLE": -3,
    "SAN FRANCISCO BAY AREA": -3,
    "LOS ANGELES": -3,
    "GUADALAJARA": -2,
    "MEXICO CITY": -2,
    "MONTERREY": -2,
    "HOUSTON": -1,
    "DALLAS": -1,
    "KANSAS CITY": -1,
    "ATLANTA": 0,
    "MIAMI": 0,
    "TORONTO": 0,
    "BOSTON": 0,
    "PHILADELPHIA": 0,
    "NEW YORK NEW JERSEY": 0,
}


def _pair(home_team: object, away_team: object) -> frozenset[str]:
    return frozenset((str(home_team), str(away_team)))


def _expected_local_timestamp(official: pd.Series) -> pd.Timestamp:
    kickoff = pd.Timestamp(
        f"{official['schedule_date']} {official['time_et']}"
    )
    local_hour = (
        kickoff.hour + CITY_ET_OFFSETS[str(official["host_city"])]
    ) % 24
    return pd.Timestamp(official["schedule_date"]).replace(
        hour=local_hour,
        minute=kickoff.minute,
    )


def _expected_utc_timestamp(official: pd.Series) -> pd.Timestamp:
    kickoff_et = pd.Timestamp(
        f"{official['schedule_date']} {official['time_et']}",
        tz="America/New_York",
    )
    return kickoff_et.tz_convert("UTC").tz_localize(None)


def audit_official_group_schedule(
    matches: pd.DataFrame,
    official_schedule: pd.DataFrame,
) -> list[str]:
    """Compare the local 72-match group schedule with the FIFA source extract."""
    issues: list[str] = []
    local = matches.copy()
    official = official_schedule.copy()
    local["date"] = pd.to_datetime(local["date"], errors="coerce")
    if "kickoff_utc" in local:
        local["kickoff_utc"] = pd.to_datetime(
            local["kickoff_utc"], errors="coerce"
        )
    official["schedule_date"] = pd.to_datetime(
        official["schedule_date"], errors="coerce"
    )

    if len(local) != 72:
        issues.append(f"Lokaler Gruppenspielplan hat {len(local)} statt 72 Spiele.")
    if len(official) != 72:
        issues.append(
            f"Offizieller Vergleichsabzug hat {len(official)} statt 72 Spiele."
        )

    local_by_pair = {
        _pair(row.home_team, row.away_team): row
        for row in local.itertuples(index=False)
    }
    for source in official.itertuples(index=False):
        pair = _pair(source.home_team, source.away_team)
        match = local_by_pair.get(pair)
        label = f"FIFA-Spiel {int(source.official_match_number)}"
        if match is None:
            issues.append(f"{label}: Paarung fehlt.")
            continue
        if str(match.home_team) != str(source.home_team):
            issues.append(f"{label}: Heim-/Auswaertszuordnung weicht ab.")
        if str(match.group) != str(source.group):
            issues.append(f"{label}: Gruppe weicht ab.")
        if int(match.official_match_number) != int(source.official_match_number):
            issues.append(f"{label}: offizielle Spielnummer weicht ab.")

        expected_venue = CITY_TO_VENUE[str(source.host_city)]
        if str(match.venue) != expected_venue:
            issues.append(
                f"{label}: Stadion {match.venue!s} statt {expected_venue}."
            )

        expected_time = _expected_local_timestamp(pd.Series(source._asdict()))
        if pd.isna(match.date) or match.date != expected_time:
            issues.append(
                f"{label}: lokale Anspielzeit {match.date!s} "
                f"statt {expected_time!s}."
            )
        if hasattr(match, "kickoff_utc"):
            expected_utc = _expected_utc_timestamp(pd.Series(source._asdict()))
            if pd.isna(match.kickoff_utc) or match.kickoff_utc != expected_utc:
                issues.append(
                    f"{label}: UTC-Anspielzeit {match.kickoff_utc!s} "
                    f"statt {expected_utc!s}."
                )

    for group, group_matches in local.groupby("group"):
        teams = set(group_matches["home_team"]) | set(group_matches["away_team"])
        pairs = {
            _pair(row.home_team, row.away_team)
            for row in group_matches.itertuples(index=False)
        }
        if len(group_matches) != 6 or len(teams) != 4 or len(pairs) != 6:
            issues.append(
                f"Gruppe {group}: kein vollstaendiges Vierergruppen-Round-Robin."
            )

    return issues

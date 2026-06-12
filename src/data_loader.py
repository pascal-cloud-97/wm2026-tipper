from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TextIO

import pandas as pd


class DataValidationError(ValueError):
    """Raised when imported data does not match the documented contract."""


TABLE_SCHEMAS = {
    "teams": {
        "required": {"team_id", "team_name"},
        "numeric": {
            "rating",
            "fifa_rank",
            "fifa_points",
            "latitude",
            "longitude",
        },
    },
    "matches": {
        "required": {"match_id", "date", "home_team", "away_team", "stage"},
        "numeric": {
            "official_match_number",
            "home_travel_km",
            "away_travel_km",
            "actual_home_goals",
            "actual_away_goals",
        },
        "dates": {"date", "kickoff_utc"},
    },
    "historical_results": {
        "required": {
            "date",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
        },
        "numeric": {"home_goals", "away_goals"},
        "dates": {"date"},
    },
    "ratings": {
        "required": {"team_id", "as_of", "rating"},
        "numeric": {"rating", "fifa_rank", "fifa_points"},
        "dates": {"as_of"},
    },
    "tips": {
        "required": {"match_id", "tip_home", "tip_away"},
        "numeric": {"tip_home", "tip_away"},
    },
    "availability": {
        "required": {
            "team_id",
            "player_name",
            "status",
            "impact",
            "as_of",
            "source",
        },
        "numeric": {"impact"},
        "dates": {"as_of"},
    },
    "lineups": {
        "required": {
            "match_id",
            "team_id",
            "player_name",
            "is_starting",
            "player_rating",
            "as_of",
            "source",
        },
        "numeric": {"player_rating", "expected_minutes"},
        "dates": {"as_of"},
    },
    "odds": {
        "required": {
            "match_id",
            "bookmaker",
            "collected_at",
            "home_odds",
            "draw_odds",
            "away_odds",
            "source",
        },
        "numeric": {"home_odds", "draw_odds", "away_odds"},
        "dates": {"collected_at"},
    },
    "outright_odds": {
        "required": {
            "team_id",
            "bookmaker",
            "market",
            "collected_at",
            "decimal_odds",
            "source",
        },
        "numeric": {"decimal_odds"},
        "dates": {"collected_at"},
    },
}


@dataclass
class DataBundle:
    teams: pd.DataFrame
    matches: pd.DataFrame
    historical_results: pd.DataFrame
    ratings: pd.DataFrame
    tips: pd.DataFrame
    availability: pd.DataFrame = field(default_factory=pd.DataFrame)
    lineups: pd.DataFrame = field(default_factory=pd.DataFrame)
    odds: pd.DataFrame = field(default_factory=pd.DataFrame)
    outright_odds: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_label: str = "Unbekannte Quelle"
    is_demo: bool = False


def _read_json(source: str | Path | BinaryIO | TextIO) -> pd.DataFrame:
    if hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
    else:
        with Path(source).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("records", payload))
    return pd.DataFrame(payload)


def load_table(
    source: str | Path | BinaryIO | TextIO,
    table_name: str,
    filename: str | None = None,
) -> pd.DataFrame:
    """Load and validate one CSV or JSON table."""
    if table_name not in TABLE_SCHEMAS:
        raise DataValidationError(f"Unbekannter Tabellentyp: {table_name}")

    inferred_name = filename or getattr(source, "name", "") or str(source)
    suffix = Path(inferred_name).suffix.lower()
    if suffix == ".json":
        frame = _read_json(source)
    elif suffix == ".csv":
        frame = pd.read_csv(source)
    else:
        raise DataValidationError("Nur CSV- und JSON-Dateien werden unterstützt.")
    return validate_dataframe(frame, table_name)


def load_uploaded_bytes(content: bytes, filename: str, table_name: str) -> pd.DataFrame:
    return load_table(io.BytesIO(content), table_name, filename=filename)


def validate_dataframe(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    schema = TABLE_SCHEMAS[table_name]
    cleaned = frame.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]

    missing = sorted(schema["required"] - set(cleaned.columns))
    if missing:
        raise DataValidationError(
            f"{table_name}: Pflichtspalten fehlen: {', '.join(missing)}"
        )
    null_required = [
        column
        for column in schema["required"]
        if cleaned[column].isna().any()
        or cleaned[column].astype(str).str.strip().eq("").any()
    ]
    if null_required:
        raise DataValidationError(
            f"{table_name}: leere Pflichtwerte in: "
            + ", ".join(sorted(null_required))
        )

    for column in schema.get("numeric", set()) & set(cleaned.columns):
        original_non_null = cleaned[column].notna()
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        invalid = original_non_null & cleaned[column].isna()
        if invalid.any():
            rows = ", ".join(str(index + 2) for index in cleaned.index[invalid][:5])
            raise DataValidationError(
                f"{table_name}.{column}: ungültige Zahl in CSV-Zeile(n) {rows}"
            )

    for column in schema.get("dates", set()) & set(cleaned.columns):
        original_non_null = cleaned[column].notna()
        cleaned[column] = pd.to_datetime(cleaned[column], errors="coerce", utc=True)
        invalid = original_non_null & cleaned[column].isna()
        if invalid.any():
            rows = ", ".join(str(index + 2) for index in cleaned.index[invalid][:5])
            raise DataValidationError(
                f"{table_name}.{column}: ungültiges Datum in CSV-Zeile(n) {rows}"
            )
        cleaned[column] = cleaned[column].dt.tz_convert(None)

    if table_name in {"matches", "historical_results"}:
        same_team = cleaned["home_team"].astype(str) == cleaned["away_team"].astype(str)
        if same_team.any():
            raise DataValidationError(
                f"{table_name}: Heim- und Auswärtsteam dürfen nicht identisch sein."
            )

    if table_name == "matches" and "official_match_number" in cleaned:
        numbers = cleaned["official_match_number"]
        invalid_numbers = numbers.isna() | (numbers < 1) | (numbers % 1 != 0)
        if invalid_numbers.any():
            raise DataValidationError(
                "matches.official_match_number muss eine positive Ganzzahl sein."
            )
        if numbers.duplicated().any():
            duplicates = numbers[numbers.duplicated()].astype(int).tolist()
            raise DataValidationError(
                "matches: doppelte offizielle Spielnummern: "
                + str(duplicates[:5])
            )
        cleaned["official_match_number"] = numbers.astype(int)

    if table_name == "historical_results":
        goals = cleaned[["home_goals", "away_goals"]]
        if (goals < 0).any().any() or (goals % 1 != 0).any().any():
            raise DataValidationError(
                "historical_results: Tore müssen nicht-negative Ganzzahlen sein."
            )
        cleaned[["home_goals", "away_goals"]] = goals.astype(int)

    if table_name == "tips":
        goals = cleaned[["tip_home", "tip_away"]]
        if (goals < 0).any().any() or (goals % 1 != 0).any().any():
            raise DataValidationError(
                "tips: Tipp-Tore müssen nicht-negative Ganzzahlen sein."
            )
        cleaned[["tip_home", "tip_away"]] = goals.astype(int)

    if table_name == "availability":
        allowed = {"out", "doubtful", "questionable", "suspended", "available"}
        cleaned["status"] = cleaned["status"].astype(str).str.lower().str.strip()
        invalid_status = ~cleaned["status"].isin(allowed)
        if invalid_status.any():
            values = sorted(cleaned.loc[invalid_status, "status"].unique())
            raise DataValidationError(
                "availability.status: erlaubt sind "
                + ", ".join(sorted(allowed))
                + f"; gefunden: {values[:5]}"
            )
        if ((cleaned["impact"] < 0) | (cleaned["impact"] > 1)).any():
            raise DataValidationError(
                "availability.impact muss zwischen 0 und 1 liegen."
            )

    if table_name == "lineups":
        truthy = {"1", "true", "yes", "ja"}
        falsy = {"0", "false", "no", "nein"}
        normalized = cleaned["is_starting"].astype(str).str.lower().str.strip()
        invalid = ~normalized.isin(truthy | falsy)
        if invalid.any():
            raise DataValidationError(
                "lineups.is_starting muss true/false oder 1/0 sein."
            )
        cleaned["is_starting"] = normalized.isin(truthy)
        if (
            (cleaned["player_rating"] < 0)
            | (cleaned["player_rating"] > 100)
        ).any():
            raise DataValidationError(
                "lineups.player_rating muss zwischen 0 und 100 liegen."
            )
        if "expected_minutes" in cleaned:
            invalid_minutes = (
                (cleaned["expected_minutes"] < 0)
                | (cleaned["expected_minutes"] > 130)
            )
            if invalid_minutes.any():
                raise DataValidationError(
                    "lineups.expected_minutes muss zwischen 0 und 130 liegen."
                )

    if table_name == "odds":
        prices = cleaned[["home_odds", "draw_odds", "away_odds"]]
        if (prices <= 1.0).any().any():
            raise DataValidationError(
                "odds: Dezimalquoten müssen größer als 1.0 sein."
            )
    if table_name == "outright_odds":
        if (cleaned["decimal_odds"] <= 1.0).any():
            raise DataValidationError(
                "outright_odds: Dezimalquoten müssen größer als 1.0 sein."
            )
        cleaned["market"] = cleaned["market"].astype(str).str.lower().str.strip()
        if (~cleaned["market"].isin({"champion"})).any():
            raise DataValidationError(
                "outright_odds.market unterstützt aktuell nur 'champion'."
            )

    id_column = {
        "teams": "team_id",
        "matches": "match_id",
        "tips": "match_id",
    }.get(table_name)
    if id_column and cleaned[id_column].duplicated().any():
        duplicates = cleaned.loc[cleaned[id_column].duplicated(), id_column].tolist()
        raise DataValidationError(
            f"{table_name}: doppelte IDs in {id_column}: {duplicates[:5]}"
        )

    return cleaned


def load_dataset(directory: str | Path) -> DataBundle:
    root = Path(directory)
    required_files = {
        "teams": root / "teams.csv",
        "matches": root / "matches.csv",
        "historical_results": root / "historical_results.csv",
    }
    optional_files = {
        "ratings": root / "ratings.csv",
        "tips": root / "tips.csv",
        "availability": root / "availability.csv",
        "lineups": root / "lineups.csv",
        "odds": root / "odds.csv",
        "outright_odds": root / "outright_odds.csv",
    }
    missing = [path.name for path in required_files.values() if not path.exists()]
    if missing:
        raise DataValidationError(
            f"Datensatz unvollständig. Dateien fehlen: {', '.join(missing)}"
        )

    loaded = {
        name: load_table(path, name) for name, path in required_files.items()
    }
    for name, path in optional_files.items():
        loaded[name] = (
            load_table(path, name)
            if path.exists()
            else pd.DataFrame(columns=sorted(TABLE_SCHEMAS[name]["required"]))
        )

    return DataBundle(
        **loaded,
        source_label=root.name,
        is_demo="example" in str(root).lower() or "demo" in str(root).lower(),
    )


def validate_references(bundle: DataBundle) -> list[str]:
    """Return cross-table warnings without inventing replacements."""
    warnings: list[str] = []
    team_ids = set(bundle.teams["team_id"].astype(str))
    referenced = set(bundle.matches["home_team"].astype(str)) | set(
        bundle.matches["away_team"].astype(str)
    )
    unknown = sorted(referenced - team_ids)
    if unknown:
        warnings.append(
            "Spiele referenzieren unbekannte team_id-Werte: " + ", ".join(unknown)
        )
    for table_name, frame in (
        ("Verfügbarkeit", bundle.availability),
        ("Aufstellungen", bundle.lineups),
    ):
        if frame.empty:
            continue
        unknown = sorted(set(frame["team_id"].astype(str)) - team_ids)
        if unknown:
            warnings.append(
                f"{table_name} referenziert unbekannte team_id-Werte: "
                + ", ".join(unknown)
            )
    if not bundle.lineups.empty:
        unknown_matches = sorted(
            set(bundle.lineups["match_id"].astype(str))
            - set(bundle.matches["match_id"].astype(str))
        )
        if unknown_matches:
            warnings.append(
                "Aufstellungen referenzieren unbekannte match_id-Werte: "
                + ", ".join(unknown_matches)
            )
    if not bundle.odds.empty:
        unknown_matches = sorted(
            set(bundle.odds["match_id"].astype(str))
            - set(bundle.matches["match_id"].astype(str))
        )
        if unknown_matches:
            warnings.append(
                "Quoten referenzieren unbekannte match_id-Werte: "
                + ", ".join(unknown_matches)
            )
    if not bundle.outright_odds.empty:
        unknown_teams = sorted(
            set(bundle.outright_odds["team_id"].astype(str)) - team_ids
        )
        if unknown_teams:
            warnings.append(
                "Langzeitquoten referenzieren unbekannte team_id-Werte: "
                + ", ".join(unknown_teams)
            )
    if bundle.matches["date"].isna().any():
        warnings.append("Einige Spiele haben kein verwertbares Datum.")
    return warnings

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset  # noqa: E402
from src.storage import save_match_odds, save_outright_odds  # noqa: E402
from src.swisslos_odds import fetch_swisslos_odds  # noqa: E402


def _archive_csv(
    path: Path,
    frame: pd.DataFrame,
    keys: list[str],
) -> None:
    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    incoming = frame.copy()
    incoming["collected_at"] = pd.to_datetime(
        incoming["collected_at"]
    ).dt.floor("s")
    if not existing.empty:
        existing["collected_at"] = pd.to_datetime(
            existing["collected_at"]
        ).dt.floor("s")
    combined = (
        pd.concat([existing, incoming], ignore_index=True)
        if not existing.empty
        else incoming
    )
    combined = combined.drop_duplicates(subset=keys, keep="last")
    combined.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aktuelle offizielle Swisslos-Sporttip-Quoten speichern."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "world_cup_2026",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=ROOT / "data" / "wm2026_tipper.sqlite",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    bundle = load_dataset(args.data_dir)
    snapshot = fetch_swisslos_odds(
        bundle.teams, bundle.matches, timeout=args.timeout
    )
    match_count = save_match_odds(args.database, snapshot.match_odds)
    outright_count = save_outright_odds(
        args.database, snapshot.outright_odds
    )
    _archive_csv(
        args.data_dir / "odds.csv",
        snapshot.match_odds,
        ["match_id", "bookmaker", "collected_at"],
    )
    _archive_csv(
        args.data_dir / "outright_odds.csv",
        snapshot.outright_odds,
        ["team_id", "bookmaker", "market", "collected_at"],
    )
    print(
        f"Swisslos-Stand {snapshot.collected_at.isoformat()}: "
        f"{match_count} Spielquoten und {outright_count} "
        "Weltmeisterquoten neu gespeichert."
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fifa_rankings import update_local_fifa_rankings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aktualisiert die lokale Rangliste aus der offiziellen FIFA-Quelle."
    )
    parser.add_argument(
        "--teams",
        type=Path,
        default=Path("data/world_cup_2026/teams.csv"),
    )
    parser.add_argument(
        "--ratings",
        type=Path,
        default=Path("data/world_cup_2026/ratings.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rankings, schedule_id = update_local_fifa_rankings(
        args.teams,
        args.ratings,
    )
    print(
        f"{len(rankings)} FIFA-Ranglistenwerte vom "
        f"{rankings.iloc[0]['as_of']} geladen ({schedule_id})."
    )


if __name__ == "__main__":
    main()

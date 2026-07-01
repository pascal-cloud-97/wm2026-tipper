from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.world_cup_updater import update_world_cup_files


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update completed World Cup matches and card events."
    )
    parser.add_argument("--as-of", default=date.today().isoformat())
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=Path("data/world_cup_2026"),
    )
    args = parser.parse_args()
    summary = update_world_cup_files(args.data_directory, args.as_of)
    print(
        f"Added {summary.get('added_fixtures', 0)} fixtures. "
        f"Updated {summary['completed_matches']} completed matches, "
        f"{summary['card_events']} card events and "
        f"{summary['suspensions']} suspensions through {args.as_of}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.history_updater import update_history


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Update real men's senior international results."
    )
    parser.add_argument("--since", default="2018-01-01")
    parser.add_argument("--as-of", default="2026-06-10")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/world_cup_2026/historical_results.csv"),
    )
    parser.add_argument(
        "--matches",
        type=Path,
        default=Path("data/world_cup_2026/matches.csv"),
    )
    args = parser.parse_args()
    frame, completed = update_history(
        args.output,
        args.since,
        args.as_of,
        matches_path=args.matches,
    )
    print(
        f"Wrote {len(frame)} matches through {args.as_of} "
        f"to {args.output.resolve()}; synchronized {completed} World Cup results."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

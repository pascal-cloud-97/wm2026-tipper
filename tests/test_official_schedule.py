from pathlib import Path
import unittest

import pandas as pd

from src.schedule_validation import audit_official_group_schedule


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "world_cup_2026"


class OfficialScheduleTests(unittest.TestCase):
    def test_local_schedule_matches_official_fifa_extract(self):
        matches = pd.read_csv(DATA_DIR / "matches.csv")
        official = pd.read_csv(DATA_DIR / "official_schedule_manifest.csv")

        self.assertEqual(audit_official_group_schedule(matches, official), [])

    def test_schedule_audit_detects_wrong_kickoff(self):
        matches = pd.read_csv(DATA_DIR / "matches.csv")
        official = pd.read_csv(DATA_DIR / "official_schedule_manifest.csv")
        matches.loc[matches["official_match_number"] == 29, "date"] = (
            "2026-06-19T21:00:00"
        )

        issues = audit_official_group_schedule(matches, official)

        self.assertTrue(any("FIFA-Spiel 29" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()

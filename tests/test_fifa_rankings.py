import json
import unittest

from src.fifa_rankings import latest_ranking_metadata, parse_ranking_payload


class FifaRankingTests(unittest.TestCase):
    def test_latest_metadata_uses_first_official_date(self):
        next_data = {
            "props": {
                "pageProps": {
                    "pageData": {
                        "ranking": {
                            "allAvailableDates": [
                                {"id": "latest-id", "date": "2026-04-01"},
                                {"id": "older-id", "date": "2026-01-19"},
                            ]
                        }
                    }
                }
            }
        }
        html = (
            '<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(next_data)
            + "</script>"
        )
        self.assertEqual(
            latest_ranking_metadata(html),
            ("latest-id", "2026-04-01"),
        )

    def test_ranking_payload_preserves_official_points(self):
        frame = parse_ranking_payload(
            {
                "Results": [
                    {
                        "IdCountry": "ESP",
                        "Rank": 2,
                        "TotalPoints": 1876.395199,
                        "TeamName": [
                            {"Locale": "en-GB", "Description": "Spain"}
                        ],
                    }
                ]
            },
            "2026-04-01",
        )
        self.assertEqual(frame.iloc[0]["team_id"], "ESP")
        self.assertEqual(int(frame.iloc[0]["fifa_rank"]), 2)
        self.assertAlmostEqual(frame.iloc[0]["fifa_points"], 1876.395199)


if __name__ == "__main__":
    unittest.main()

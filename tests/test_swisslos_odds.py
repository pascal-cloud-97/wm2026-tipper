import unittest

import pandas as pd

from src.swisslos_odds import (
    SwisslosOddsError,
    _decode_message,
    _encode_message,
    _extract_match_odds,
    _extract_outright_odds,
)


class SwisslosOddsTests(unittest.TestCase):
    def setUp(self):
        self.timestamp = pd.Timestamp("2026-06-11 21:45:00")

    def test_binary_message_roundtrip(self):
        payload = {"payload": [{"type": "Test", "body": {"value": 3}}]}
        self.assertEqual(_decode_message(_encode_message(payload)), payload)

    def test_extracts_match_odds_and_maps_curacao_alias(self):
        teams = pd.DataFrame(
            [
                {"team_id": "CUW", "team_name": "Curaçao", "country": "Curaçao"},
                {"team_id": "GER", "team_name": "Deutschland", "country": "Germany"},
            ]
        )
        matches = pd.DataFrame(
            [
                {
                    "match_id": "M1",
                    "home_team": "CUW",
                    "away_team": "GER",
                }
            ]
        )
        store = {
            "Competitor": {
                "c1": {
                    "urn": "c1",
                    "name": "Curacao",
                    "abbreviation": "CUR",
                    "properties": {"AliasNames": ["Curacao"]},
                },
                "c2": {
                    "urn": "c2",
                    "name": "Germany",
                    "abbreviation": "GER",
                    "properties": {"AliasNames": ["Germany"]},
                },
            },
            "Event": {
                "e1": {
                    "urn": "e1",
                    "name": "Curacao : Germany",
                    "type": 0,
                    "eventCompetitors": [
                        {"qualifier": "home", "competitor": "c1"},
                        {"qualifier": "away", "competitor": "c2"},
                    ],
                    "markets": ["m1"],
                }
            },
            "Market": {
                "m1": {
                    "urn": "m1",
                    "type": "asw:markettype:1",
                    "state": 1,
                    "selections": ["s1", "sx", "s2"],
                }
            },
            "Selection": {
                "s1": {
                    "urn": "s1",
                    "type": "asw:selectiontype:1",
                    "state": 1,
                    "odds": 7.0,
                },
                "sx": {
                    "urn": "sx",
                    "type": "asw:selectiontype:2",
                    "state": 1,
                    "odds": 4.2,
                },
                "s2": {
                    "urn": "s2",
                    "type": "asw:selectiontype:3",
                    "state": 1,
                    "odds": 1.4,
                },
            },
        }

        result = _extract_match_odds(
            store, teams, matches, ["e1"], self.timestamp
        ).iloc[0]

        self.assertEqual(result["match_id"], "M1")
        self.assertEqual(result["home_odds"], 7.0)
        self.assertEqual(result["away_odds"], 1.4)

    def test_extracts_champion_odds_via_competitor_alias(self):
        teams = pd.DataFrame(
            [{"team_id": "IRN", "team_name": "Iran", "country": "Iran"}]
        )
        store = {
            "Competitor": {
                "c1": {
                    "urn": "c1",
                    "name": "IR Iran",
                    "abbreviation": "IRI",
                    "properties": {"AliasNames": ["IR Iran", "Iran"]},
                }
            },
            "Market": {
                "m1": {
                    "urn": "m1",
                    "type": "asw:markettype:534:pre:markettext:176991",
                    "state": 1,
                    "selections": ["s1"],
                }
            },
            "Selection": {
                "s1": {
                    "urn": "s1",
                    "type": "st1",
                    "state": 1,
                    "odds": 500.0,
                }
            },
            "SelectionType": {
                "st1": {
                    "urn": "st1",
                    "name": "IR Iran",
                    "translations": {"de": "Iran"},
                }
            },
        }

        result = _extract_outright_odds(
            store, teams, self.timestamp
        ).iloc[0]

        self.assertEqual(result["team_id"], "IRN")
        self.assertEqual(result["decimal_odds"], 500.0)

    def test_unresolved_active_match_is_rejected(self):
        teams = pd.DataFrame(
            [{"team_id": "A", "team_name": "A", "country": "A"}]
        )
        matches = pd.DataFrame(
            [{"match_id": "M1", "home_team": "A", "away_team": "B"}]
        )
        store = {
            "Competitor": {},
            "Event": {
                "e1": {
                    "urn": "e1",
                    "name": "Unknown match",
                    "type": 0,
                    "eventCompetitors": [],
                    "markets": [],
                }
            },
            "Market": {},
            "Selection": {},
        }

        with self.assertRaises(SwisslosOddsError):
            _extract_match_odds(
                store, teams, matches, ["e1"], self.timestamp
            )


if __name__ == "__main__":
    unittest.main()

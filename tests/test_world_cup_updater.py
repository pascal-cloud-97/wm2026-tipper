import unittest

from src.world_cup_updater import parse_match_minute


class WorldCupUpdaterTests(unittest.TestCase):
    def test_parse_regular_and_stoppage_time(self):
        self.assertEqual(parse_match_minute("49'"), 49)
        self.assertEqual(parse_match_minute("90'+3'"), 93)

    def test_invalid_minute_fails(self):
        with self.assertRaises(ValueError):
            parse_match_minute("")


if __name__ == "__main__":
    unittest.main()

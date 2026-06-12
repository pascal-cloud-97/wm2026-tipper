from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .prediction import MatchPrediction, poisson_score_matrix


@dataclass(frozen=True)
class WorldCupSimulationResult:
    team_probabilities: pd.DataFrame
    champion: str
    champion_probability: float
    likely_finalists: tuple[str, str]


def _sample_score(
    prediction: MatchPrediction, rng: np.random.Generator
) -> tuple[int, int]:
    size = prediction.score_matrix.shape[1]
    index = int(
        rng.choice(
            prediction.score_matrix.size,
            p=prediction.score_matrix.ravel(),
        )
    )
    return index // size, index % size


def _actual_or_sampled_score(
    match: dict,
    prediction: MatchPrediction,
    rng: np.random.Generator,
) -> tuple[int, int]:
    home = match.get("actual_home_goals")
    away = match.get("actual_away_goals")
    if (
        str(match.get("status", "")).lower() in {"completed", "played", "finished"}
        and pd.notna(home)
        and pd.notna(away)
    ):
        return int(home), int(away)
    return _sample_score(prediction, rng)


def simulate_tournament(
    matches: pd.DataFrame,
    predictions: dict[str, MatchPrediction],
    runs: int = 5000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Monte Carlo group simulation for the matches currently loaded."""
    if runs < 100:
        raise ValueError("Für eine stabile Simulation sind mindestens 100 Läufe nötig.")
    group_matches = matches[matches["stage"].str.lower() == "group"].copy()
    if group_matches.empty or "group" not in group_matches:
        raise ValueError("Keine Gruppenspiele mit Gruppenzuordnung vorhanden.")

    rng = np.random.default_rng(seed)
    teams_by_group: dict[str, list[str]] = {}
    for group, rows in group_matches.groupby("group"):
        teams_by_group[str(group)] = sorted(
            set(rows["home_team"].astype(str)) | set(rows["away_team"].astype(str))
        )
    counts = {
        (group, team): {"first": 0, "top_two": 0, "points": 0.0}
        for group, teams in teams_by_group.items()
        for team in teams
    }

    for _ in range(runs):
        tables = {
            group: {
                team: {"points": 0, "gd": 0, "gf": 0, "tie": rng.random()}
                for team in teams
            }
            for group, teams in teams_by_group.items()
        }
        for match in group_matches.to_dict("records"):
            match_id = str(match["match_id"])
            if match_id not in predictions:
                continue
            group = str(match["group"])
            home = str(match["home_team"])
            away = str(match["away_team"])
            home_goals, away_goals = _actual_or_sampled_score(
                match, predictions[match_id], rng
            )
            tables[group][home]["gf"] += home_goals
            tables[group][away]["gf"] += away_goals
            tables[group][home]["gd"] += home_goals - away_goals
            tables[group][away]["gd"] += away_goals - home_goals
            if home_goals > away_goals:
                tables[group][home]["points"] += 3
            elif home_goals < away_goals:
                tables[group][away]["points"] += 3
            else:
                tables[group][home]["points"] += 1
                tables[group][away]["points"] += 1

        for group, table in tables.items():
            ranking = sorted(
                table,
                key=lambda team: (
                    table[team]["points"],
                    table[team]["gd"],
                    table[team]["gf"],
                    table[team]["tie"],
                ),
                reverse=True,
            )
            for team, stats in table.items():
                counts[(group, team)]["points"] += stats["points"]
            counts[(group, ranking[0])]["first"] += 1
            for team in ranking[:2]:
                counts[(group, team)]["top_two"] += 1

    rows = [
        {
            "group": group,
            "team_id": team,
            "group_win_probability": values["first"] / runs,
            "top_two_probability": values["top_two"] / runs,
            "expected_group_points": values["points"] / runs,
        }
        for (group, team), values in counts.items()
    ]
    return pd.DataFrame(rows).sort_values(
        ["group", "top_two_probability"], ascending=[True, False]
    )


THIRD_PLACE_SLOTS = {
    74: set("ABCDF"),
    77: set("CDFGH"),
    79: set("CEFHI"),
    80: set("EHIJK"),
    81: set("BEFIJ"),
    82: set("AEHIJ"),
    85: set("EFGIJ"),
    87: set("DEIJL"),
}


def _assign_third_placed(
    qualified_thirds: dict[str, str],
) -> dict[int, str]:
    """Assign best third-placed teams to their eligible round-of-32 slots."""
    slots = sorted(
        THIRD_PLACE_SLOTS,
        key=lambda slot: len(
            set(qualified_thirds) & THIRD_PLACE_SLOTS[slot]
        ),
    )

    def backtrack(
        index: int,
        remaining: set[str],
        assignment: dict[int, str],
    ) -> dict[int, str] | None:
        if index == len(slots):
            return assignment.copy()
        slot = slots[index]
        candidates = sorted(remaining & THIRD_PLACE_SLOTS[slot])
        for group in candidates:
            assignment[slot] = qualified_thirds[group]
            result = backtrack(index + 1, remaining - {group}, assignment)
            if result is not None:
                return result
            assignment.pop(slot, None)
        return None

    result = backtrack(0, set(qualified_thirds), {})
    if result is None:
        raise ValueError(
            "Die qualifizierten Gruppendritten konnten nicht auf den "
            "offiziellen K.-o.-Slotrahmen verteilt werden."
        )
    return result


def _neutral_prediction_matrix(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    cache: dict[tuple[str, str], np.ndarray],
) -> np.ndarray:
    key = (team_a, team_b)
    if key in cache:
        return cache[key]
    edge = (ratings[team_a] - ratings[team_b]) / 500.0
    expected_a = float(np.clip(1.32 * np.exp(edge / 2), 0.25, 4.2))
    expected_b = float(np.clip(1.32 * np.exp(-edge / 2), 0.25, 4.2))
    cache[key] = poisson_score_matrix(expected_a, expected_b, max_goals=6)
    return cache[key]


def _knockout_winner(
    team_a: str,
    team_b: str,
    ratings: dict[str, float],
    rng: np.random.Generator,
    cache: dict[tuple[str, str], np.ndarray],
) -> str:
    matrix = _neutral_prediction_matrix(team_a, team_b, ratings, cache)
    size = matrix.shape[1]
    index = int(rng.choice(matrix.size, p=matrix.ravel()))
    goals_a, goals_b = index // size, index % size
    if goals_a > goals_b:
        return team_a
    if goals_b > goals_a:
        return team_b
    penalty_probability_a = 1.0 / (
        1.0 + np.exp(-(ratings[team_a] - ratings[team_b]) / 600.0)
    )
    return team_a if rng.random() < penalty_probability_a else team_b


def simulate_world_cup(
    matches: pd.DataFrame,
    predictions: dict[str, MatchPrediction],
    teams: pd.DataFrame,
    runs: int = 5000,
    seed: int = 2026,
) -> WorldCupSimulationResult:
    """Simulate the 2026 group stage and official knockout bracket structure."""
    if runs < 100:
        raise ValueError("Für eine stabile Simulation sind mindestens 100 Läufe nötig.")
    group_matches = matches[matches["stage"].str.lower() == "group"].copy()
    groups = {
        str(group): sorted(
            set(rows["home_team"].astype(str))
            | set(rows["away_team"].astype(str))
        )
        for group, rows in group_matches.groupby("group")
    }
    if set(groups) != set("ABCDEFGHIJKL"):
        raise ValueError("Für die WM-Simulation werden die Gruppen A bis L benötigt.")

    ratings = (
        teams.assign(team_id=teams["team_id"].astype(str))
        .set_index("team_id")["rating"]
        .astype(float)
        .to_dict()
    )
    rng = np.random.default_rng(seed)
    stages = ["round_of_32", "round_of_16", "quarterfinal", "semifinal", "final", "champion"]
    counts = {
        team: {stage: 0 for stage in stages}
        for team in teams["team_id"].astype(str)
    }
    group_win_counts = {team: 0 for team in counts}
    final_pair_counts: dict[tuple[str, str], int] = {}
    knockout_cache: dict[tuple[str, str], np.ndarray] = {}

    round_of_16_map = {
        89: (74, 77),
        90: (73, 75),
        91: (76, 78),
        92: (79, 80),
        93: (83, 84),
        94: (81, 82),
        95: (86, 88),
        96: (85, 87),
    }
    quarterfinal_map = {
        97: (89, 90),
        98: (93, 94),
        99: (91, 92),
        100: (95, 96),
    }
    semifinal_map = {101: (97, 98), 102: (99, 100)}

    for _ in range(runs):
        tables = {
            group: {
                team: {"points": 0, "gd": 0, "gf": 0, "tie": rng.random()}
                for team in group_teams
            }
            for group, group_teams in groups.items()
        }
        for match in group_matches.to_dict("records"):
            match_id = str(match["match_id"])
            group = str(match["group"])
            home = str(match["home_team"])
            away = str(match["away_team"])
            home_goals, away_goals = _actual_or_sampled_score(
                match, predictions[match_id], rng
            )
            tables[group][home]["gf"] += home_goals
            tables[group][away]["gf"] += away_goals
            tables[group][home]["gd"] += home_goals - away_goals
            tables[group][away]["gd"] += away_goals - home_goals
            if home_goals > away_goals:
                tables[group][home]["points"] += 3
            elif away_goals > home_goals:
                tables[group][away]["points"] += 3
            else:
                tables[group][home]["points"] += 1
                tables[group][away]["points"] += 1

        ranked: dict[str, list[str]] = {}
        for group, table in tables.items():
            ranked[group] = sorted(
                table,
                key=lambda team: (
                    table[team]["points"],
                    table[team]["gd"],
                    table[team]["gf"],
                    table[team]["tie"],
                ),
                reverse=True,
            )
            group_win_counts[ranked[group][0]] += 1

        third_candidates = [
            (
                group,
                ranked[group][2],
                tables[group][ranked[group][2]],
            )
            for group in sorted(ranked)
        ]
        third_candidates.sort(
            key=lambda item: (
                item[2]["points"],
                item[2]["gd"],
                item[2]["gf"],
                item[2]["tie"],
            ),
            reverse=True,
        )
        qualified_thirds = {
            group: team for group, team, _ in third_candidates[:8]
        }
        third_slots = _assign_third_placed(qualified_thirds)

        r32_fixtures = {
            73: (ranked["A"][1], ranked["B"][1]),
            74: (ranked["E"][0], third_slots[74]),
            75: (ranked["F"][0], ranked["C"][1]),
            76: (ranked["C"][0], ranked["F"][1]),
            77: (ranked["I"][0], third_slots[77]),
            78: (ranked["E"][1], ranked["I"][1]),
            79: (ranked["A"][0], third_slots[79]),
            80: (ranked["L"][0], third_slots[80]),
            81: (ranked["D"][0], third_slots[81]),
            82: (ranked["G"][0], third_slots[82]),
            83: (ranked["K"][1], ranked["L"][1]),
            84: (ranked["H"][0], ranked["J"][1]),
            85: (ranked["B"][0], third_slots[85]),
            86: (ranked["J"][0], ranked["H"][1]),
            87: (ranked["K"][0], third_slots[87]),
            88: (ranked["D"][1], ranked["G"][1]),
        }
        for fixture in r32_fixtures.values():
            for team in fixture:
                counts[team]["round_of_32"] += 1

        winners: dict[int, str] = {}
        for match_number, (team_a, team_b) in r32_fixtures.items():
            winners[match_number] = _knockout_winner(
                team_a, team_b, ratings, rng, knockout_cache
            )
            counts[winners[match_number]]["round_of_16"] += 1

        for match_number, (left, right) in round_of_16_map.items():
            winners[match_number] = _knockout_winner(
                winners[left], winners[right], ratings, rng, knockout_cache
            )
            counts[winners[match_number]]["quarterfinal"] += 1

        for match_number, (left, right) in quarterfinal_map.items():
            winners[match_number] = _knockout_winner(
                winners[left], winners[right], ratings, rng, knockout_cache
            )
            counts[winners[match_number]]["semifinal"] += 1

        for match_number, (left, right) in semifinal_map.items():
            winners[match_number] = _knockout_winner(
                winners[left], winners[right], ratings, rng, knockout_cache
            )
            counts[winners[match_number]]["final"] += 1

        final_pair = tuple(sorted((winners[101], winners[102])))
        final_pair_counts[final_pair] = final_pair_counts.get(final_pair, 0) + 1
        champion = _knockout_winner(*final_pair, ratings, rng, knockout_cache)
        counts[champion]["champion"] += 1

    rows = []
    for team, stage_counts in counts.items():
        row = {
            "team_id": team,
            "group_win_probability": group_win_counts[team] / runs,
        }
        row.update(
            {
                f"{stage}_probability": stage_counts[stage] / runs
                for stage in stages
            }
        )
        rows.append(row)
    probabilities = pd.DataFrame(rows).sort_values(
        "champion_probability", ascending=False
    )
    champion_row = probabilities.iloc[0]
    finalists = max(
        final_pair_counts,
        key=final_pair_counts.get,
    )
    return WorldCupSimulationResult(
        team_probabilities=probabilities,
        champion=str(champion_row["team_id"]),
        champion_probability=float(champion_row["champion_probability"]),
        likely_finalists=(finalists[0], finalists[1]),
    )

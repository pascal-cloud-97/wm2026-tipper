from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ScoringRules:
    exact_points: float = 5.0
    tendency_points: float = 2.0
    goal_difference_points: float = 1.0
    team_goals_points: float = 0.5
    combination_mode: str = "additive"
    bonus_points: float = 0.0
    multiplier: float = 1.0
    joker_multiplier: float = 1.0

    def __post_init__(self) -> None:
        numeric = [
            self.exact_points,
            self.tendency_points,
            self.goal_difference_points,
            self.team_goals_points,
            self.bonus_points,
        ]
        if any(value < 0 for value in numeric):
            raise ValueError("Punktwerte dürfen nicht negativ sein.")
        if self.multiplier <= 0 or self.joker_multiplier <= 0:
            raise ValueError("Multiplikatoren müssen größer als null sein.")
        if self.combination_mode not in {"additive", "highest_only"}:
            raise ValueError("combination_mode muss additive oder highest_only sein.")

    @classmethod
    def from_dict(cls, values: dict) -> "ScoringRules":
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in values.items() if key in allowed})

    def to_dict(self) -> dict:
        return asdict(self)


def tendency(home_goals: int, away_goals: int) -> int:
    return 1 if home_goals > away_goals else -1 if home_goals < away_goals else 0


def score_tip(
    tip_home: int,
    tip_away: int,
    actual_home: int,
    actual_away: int,
    rules: ScoringRules,
    use_joker: bool = False,
) -> float:
    values: list[float] = []
    if tip_home == actual_home and tip_away == actual_away:
        values.append(rules.exact_points)
    if tendency(tip_home, tip_away) == tendency(actual_home, actual_away):
        values.append(rules.tendency_points)
    if tip_home - tip_away == actual_home - actual_away:
        values.append(rules.goal_difference_points)
    if tip_home == actual_home:
        values.append(rules.team_goals_points)
    if tip_away == actual_away:
        values.append(rules.team_goals_points)

    base = sum(values) if rules.combination_mode == "additive" else max(values, default=0.0)
    if base > 0:
        base += rules.bonus_points
    multiplier = rules.multiplier * (rules.joker_multiplier if use_joker else 1.0)
    return float(base * multiplier)


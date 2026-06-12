from __future__ import annotations

import pandas as pd

from .optimizer import TipRecommendation
from .prediction import MatchPrediction


def build_explanation(
    prediction: MatchPrediction,
    recommendation: TipRecommendation,
) -> dict[str, object]:
    alternatives = recommendation.alternatives.copy()
    runner_up = alternatives.iloc[1] if len(alternatives) > 1 else alternatives.iloc[0]
    ev_gap = recommendation.expected_points - float(runner_up["expected_points"])
    sorted_drivers = sorted(
        prediction.contributions.items(), key=lambda item: abs(item[1]), reverse=True
    )
    driver_rows = pd.DataFrame(
        [
            {
                "factor": name,
                "effect": value,
                "direction": (
                    "Heimteam"
                    if value > 0.015
                    else "Auswärtsteam"
                    if value < -0.015
                    else "neutral"
                ),
            }
            for name, value in sorted_drivers
        ]
    )
    strongest = driver_rows.iloc[0]
    why = (
        f"{recommendation.score} liefert im Modus {recommendation.strategy} "
        f"den höchsten Strategie-Score bei {recommendation.expected_points:.2f} "
        "erwarteten Punkten. "
        f"Der stärkste Modelltreiber ist {strongest['factor']} "
        f"({strongest['direction']})."
    )
    return {
        "why": why,
        "ev_gap": ev_gap,
        "runner_up": str(runner_up["tip"]),
        "classification": recommendation.classification,
        "data_warning": (
            f"Datenunsicherheit {prediction.data_uncertainty:.0%}; "
            "fehlende Faktoren werden neutral behandelt."
        ),
        "drivers": driver_rows,
    }


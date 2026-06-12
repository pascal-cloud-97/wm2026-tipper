from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import (
    OutcomeCalibrator,
    fit_outcome_calibrator,
    multiclass_brier_score,
    multiclass_log_loss,
)
from .feature_engineering import build_match_features
from .prediction import predict_all
from .ratings import elo_snapshots


@dataclass(frozen=True)
class BacktestReport:
    predictions: pd.DataFrame
    calibration: pd.DataFrame
    outcome_metrics: pd.DataFrame
    brier_score: float
    log_loss: float
    accuracy: float
    baseline_brier_score: float
    baseline_log_loss: float
    matches: int


@dataclass(frozen=True)
class CalibrationValidation:
    calibrator: OutcomeCalibrator
    training_matches: int
    validation_matches: int
    raw_brier_score: float
    calibrated_brier_score: float
    raw_log_loss: float
    calibrated_log_loss: float


def _backtest_teams(
    historical_results: pd.DataFrame,
    known_teams: pd.DataFrame | None,
) -> pd.DataFrame:
    ids = sorted(
        set(historical_results["home_team"].astype(str))
        | set(historical_results["away_team"].astype(str))
    )
    frame = pd.DataFrame({"team_id": ids})
    if known_teams is None or known_teams.empty:
        frame["team_name"] = frame["team_id"]
        return frame
    available = known_teams.copy()
    available["team_id"] = available["team_id"].astype(str)
    columns = [
        column
        for column in [
            "team_id",
            "team_name",
            "country",
            "continent",
            "latitude",
            "longitude",
            "is_host",
        ]
        if column in available.columns
    ]
    frame = frame.merge(available[columns], on="team_id", how="left")
    frame["team_name"] = frame["team_name"].fillna(frame["team_id"])
    return frame


def _eligible_test_matches(
    history: pd.DataFrame,
    start_date: str | pd.Timestamp,
    min_prior_matches: int,
    max_matches: int | None,
) -> pd.DataFrame:
    ordered = history.sort_values("date").reset_index(drop=True)
    snapshots, _ = elo_snapshots(ordered)
    rating_lookup = snapshots.set_index("history_index").to_dict("index")
    prior_counts: dict[str, int] = {}
    rows = []
    start = pd.Timestamp(start_date)
    for index, row in ordered.iterrows():
        home = str(row["home_team"])
        away = str(row["away_team"])
        if (
            row["date"] >= start
            and prior_counts.get(home, 0) >= min_prior_matches
            and prior_counts.get(away, 0) >= min_prior_matches
        ):
            rows.append(
                {
                    "history_index": index,
                    "match_id": f"BT-{index}",
                    "date": row["date"],
                    "home_team": home,
                    "away_team": away,
                    "stage": "Historical backtest",
                    "actual_home_goals": int(row["home_goals"]),
                    "actual_away_goals": int(row["away_goals"]),
                    "neutral": bool(row.get("neutral", True)),
                    "home_rating": rating_lookup[index]["home_rating"],
                    "away_rating": rating_lookup[index]["away_rating"],
                }
            )
        prior_counts[home] = prior_counts.get(home, 0) + 1
        prior_counts[away] = prior_counts.get(away, 0) + 1
    result = pd.DataFrame(rows)
    if max_matches is not None and len(result) > max_matches:
        result = result.tail(int(max_matches)).reset_index(drop=True)
    return result


def _calibration_table(predictions: pd.DataFrame, bins: int) -> pd.DataFrame:
    probabilities = predictions[["home_win", "draw", "away_win"]].to_numpy().ravel()
    outcomes = predictions[["actual_1", "actual_x", "actual_2"]].to_numpy().ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    labels = pd.IntervalIndex.from_breaks(edges, closed="right")
    bucket = pd.cut(
        probabilities,
        bins=edges,
        include_lowest=True,
        duplicates="drop",
    )
    frame = pd.DataFrame(
        {
            "bucket": bucket,
            "forecast_probability": probabilities,
            "observed": outcomes,
        }
    )
    result = (
        frame.groupby("bucket", observed=True)
        .agg(
            predicted_probability=("forecast_probability", "mean"),
            observed_frequency=("observed", "mean"),
            observations=("observed", "size"),
        )
        .reset_index()
    )
    result["bucket_label"] = result["bucket"].astype(str)
    result["calibration_error"] = (
        result["observed_frequency"] - result["predicted_probability"]
    )
    return result.drop(columns=["bucket"])


def run_backtest(
    historical_results: pd.DataFrame,
    model_config: dict,
    known_teams: pd.DataFrame | None = None,
    start_date: str | pd.Timestamp = "2024-01-01",
    min_prior_matches: int = 5,
    max_matches: int | None = 1000,
    calibration_bins: int = 10,
) -> BacktestReport:
    if historical_results.empty:
        raise ValueError("Für den Backtest fehlen historische Resultate.")
    history = historical_results.copy()
    history["date"] = pd.to_datetime(history["date"])
    test_matches = _eligible_test_matches(
        history,
        start_date=start_date,
        min_prior_matches=int(min_prior_matches),
        max_matches=max_matches,
    )
    if test_matches.empty:
        raise ValueError("Keine Spiele erfüllen den gewählten Backtest-Zeitraum.")

    teams = _backtest_teams(history, known_teams)
    features = build_match_features(
        test_matches,
        teams,
        history,
        ratings=pd.DataFrame(),
        availability=pd.DataFrame(),
        lineups=pd.DataFrame(),
        odds=pd.DataFrame(),
    )
    clean_config = copy.deepcopy(model_config)
    clean_config.setdefault("calibration", {})["enabled"] = False
    weights = clean_config.setdefault("weights", {})
    for key in ("home_context", "travel", "availability", "lineup", "market"):
        weights[key] = 0.0
    predictions = predict_all(features, clean_config)

    rows = []
    for match in test_matches.to_dict("records"):
        prediction = predictions[str(match["match_id"])]
        actual_code = (
            "1"
            if match["actual_home_goals"] > match["actual_away_goals"]
            else "2"
            if match["actual_home_goals"] < match["actual_away_goals"]
            else "X"
        )
        predicted_code, _, predicted_probability = prediction.most_likely_outcome()
        rows.append(
            {
                "match_id": match["match_id"],
                "date": match["date"],
                "home_team": prediction.home_team,
                "away_team": prediction.away_team,
                "actual_result": (
                    f"{match['actual_home_goals']}:{match['actual_away_goals']}"
                ),
                "actual_outcome": actual_code,
                "predicted_outcome": predicted_code,
                "predicted_probability": predicted_probability,
                "home_win": prediction.home_win,
                "draw": prediction.draw,
                "away_win": prediction.away_win,
                "actual_1": float(actual_code == "1"),
                "actual_x": float(actual_code == "X"),
                "actual_2": float(actual_code == "2"),
                "correct": float(predicted_code == actual_code),
            }
        )
    frame = pd.DataFrame(rows)
    probability_matrix = frame[["home_win", "draw", "away_win"]].to_numpy()
    actual_matrix = frame[["actual_1", "actual_x", "actual_2"]].to_numpy()
    brier = float(np.mean(np.sum((probability_matrix - actual_matrix) ** 2, axis=1)))
    actual_probability = np.sum(probability_matrix * actual_matrix, axis=1)
    log_loss = float(-np.mean(np.log(np.clip(actual_probability, 1e-15, 1.0))))
    accuracy = float(frame["correct"].mean())
    training = history[history["date"] < pd.Timestamp(start_date)]
    if training.empty:
        baseline_probabilities = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    else:
        training_outcomes = np.select(
            [
                training["home_goals"] > training["away_goals"],
                training["home_goals"] == training["away_goals"],
            ],
            ["1", "X"],
            default="2",
        )
        frequencies = pd.Series(training_outcomes).value_counts(normalize=True)
        baseline_probabilities = np.array(
            [
                float(frequencies.get("1", 0.0)),
                float(frequencies.get("X", 0.0)),
                float(frequencies.get("2", 0.0)),
            ]
        )
    baseline_matrix = np.tile(baseline_probabilities, (len(frame), 1))
    baseline_brier = float(
        np.mean(np.sum((baseline_matrix - actual_matrix) ** 2, axis=1))
    )
    baseline_actual_probability = np.sum(baseline_matrix * actual_matrix, axis=1)
    baseline_log_loss = float(
        -np.mean(np.log(np.clip(baseline_actual_probability, 1e-15, 1.0)))
    )

    outcome_rows = []
    for code, probability_column, actual_column in (
        ("1", "home_win", "actual_1"),
        ("X", "draw", "actual_x"),
        ("2", "away_win", "actual_2"),
    ):
        outcome_rows.append(
            {
                "outcome": code,
                "mean_predicted_probability": float(frame[probability_column].mean()),
                "observed_frequency": float(frame[actual_column].mean()),
                "calibration_gap": float(
                    frame[actual_column].mean() - frame[probability_column].mean()
                ),
            }
        )

    return BacktestReport(
        predictions=frame,
        calibration=_calibration_table(frame, calibration_bins),
        outcome_metrics=pd.DataFrame(outcome_rows),
        brier_score=brier,
        log_loss=log_loss,
        accuracy=accuracy,
        baseline_brier_score=baseline_brier,
        baseline_log_loss=baseline_log_loss,
        matches=len(frame),
    )


def fit_and_validate_calibrator(
    predictions: pd.DataFrame,
    validation_start: str | pd.Timestamp,
) -> CalibrationValidation:
    split = pd.Timestamp(validation_start)
    ordered = predictions.copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    training = ordered[ordered["date"] < split]
    validation = ordered[ordered["date"] >= split]
    if len(training) < 30 or len(validation) < 30:
        raise ValueError(
            "Kalibrierung benötigt mindestens 30 Trainings- und 30 Validierungsspiele."
        )
    probability_columns = ["home_win", "draw", "away_win"]
    actual_columns = ["actual_1", "actual_x", "actual_2"]
    train_probabilities = training[probability_columns].to_numpy()
    train_actual = training[actual_columns].to_numpy()
    validation_probabilities = validation[probability_columns].to_numpy()
    validation_actual = validation[actual_columns].to_numpy()
    calibrator = fit_outcome_calibrator(train_probabilities, train_actual)
    calibrated = calibrator.transform(validation_probabilities)
    return CalibrationValidation(
        calibrator=calibrator,
        training_matches=len(training),
        validation_matches=len(validation),
        raw_brier_score=multiclass_brier_score(
            validation_probabilities, validation_actual
        ),
        calibrated_brier_score=multiclass_brier_score(calibrated, validation_actual),
        raw_log_loss=multiclass_log_loss(validation_probabilities, validation_actual),
        calibrated_log_loss=multiclass_log_loss(calibrated, validation_actual),
    )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OutcomeCalibrator:
    inverse_temperature: float = 1.0
    home_bias: float = 0.0
    draw_bias: float = 0.0
    away_bias: float = 0.0

    @property
    def biases(self) -> np.ndarray:
        return np.array(
            [self.home_bias, self.draw_bias, self.away_bias],
            dtype=float,
        )

    def transform(self, probabilities: np.ndarray | list[float]) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        one_dimensional = values.ndim == 1
        matrix = np.atleast_2d(values)
        if matrix.shape[1] != 3:
            raise ValueError("Kalibrierung erwartet Wahrscheinlichkeiten für 1/X/2.")
        if np.any(matrix < 0) or np.any(~np.isfinite(matrix)):
            raise ValueError("Wahrscheinlichkeiten müssen endlich und nicht-negativ sein.")
        totals = matrix.sum(axis=1, keepdims=True)
        if np.any(totals <= 0):
            raise ValueError("Wahrscheinlichkeiten müssen eine positive Summe haben.")
        normalized = matrix / totals
        logits = (
            self.inverse_temperature * np.log(np.clip(normalized, 1e-12, 1.0))
            + self.biases
        )
        logits -= logits.max(axis=1, keepdims=True)
        calibrated = np.exp(logits)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        return calibrated[0] if one_dimensional else calibrated

    def to_dict(self) -> dict[str, float]:
        return {
            "inverse_temperature": self.inverse_temperature,
            "home_bias": self.home_bias,
            "draw_bias": self.draw_bias,
            "away_bias": self.away_bias,
        }

    @classmethod
    def from_dict(cls, values: dict | None) -> "OutcomeCalibrator":
        values = values or {}
        return cls(
            inverse_temperature=float(values.get("inverse_temperature", 1.0)),
            home_bias=float(values.get("home_bias", 0.0)),
            draw_bias=float(values.get("draw_bias", 0.0)),
            away_bias=float(values.get("away_bias", 0.0)),
        )


def multiclass_log_loss(
    probabilities: np.ndarray,
    actual: np.ndarray,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(actual, dtype=float)
    return float(
        -np.mean(np.sum(outcomes * np.log(np.clip(values, 1e-15, 1.0)), axis=1))
    )


def multiclass_brier_score(
    probabilities: np.ndarray,
    actual: np.ndarray,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(actual, dtype=float)
    return float(np.mean(np.sum((values - outcomes) ** 2, axis=1)))


def fit_outcome_calibrator(
    probabilities: np.ndarray,
    actual: np.ndarray,
    iterations: int = 1500,
    learning_rate: float = 0.03,
    regularization: float = 0.002,
) -> OutcomeCalibrator:
    values = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(actual, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape != outcomes.shape:
        raise ValueError("Training benötigt gleich große Matrizen mit drei Spalten.")
    if len(values) < 30:
        raise ValueError("Für eine Kalibrierung sind mindestens 30 Spiele nötig.")
    values = values / values.sum(axis=1, keepdims=True)
    log_values = np.log(np.clip(values, 1e-12, 1.0))

    inverse_temperature = 1.0
    biases = np.zeros(3, dtype=float)
    first_moment = np.zeros(4, dtype=float)
    second_moment = np.zeros(4, dtype=float)
    beta1 = 0.9
    beta2 = 0.999

    for step in range(1, int(iterations) + 1):
        logits = inverse_temperature * log_values + biases
        logits -= logits.max(axis=1, keepdims=True)
        calibrated = np.exp(logits)
        calibrated /= calibrated.sum(axis=1, keepdims=True)
        residual = calibrated - outcomes
        gradient_temperature = float(np.mean(np.sum(residual * log_values, axis=1)))
        gradient_temperature += 2 * regularization * (inverse_temperature - 1.0)
        gradient_bias = residual.mean(axis=0) + 2 * regularization * biases
        gradient = np.concatenate([[gradient_temperature], gradient_bias])

        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
        corrected_first = first_moment / (1 - beta1**step)
        corrected_second = second_moment / (1 - beta2**step)
        update = learning_rate * corrected_first / (
            np.sqrt(corrected_second) + 1e-8
        )
        inverse_temperature -= update[0]
        biases -= update[1:]
        inverse_temperature = float(np.clip(inverse_temperature, 0.2, 3.0))
        biases = np.clip(biases - biases.mean(), -2.0, 2.0)

    return OutcomeCalibrator(
        inverse_temperature=inverse_temperature,
        home_bias=float(biases[0]),
        draw_bias=float(biases[1]),
        away_bias=float(biases[2]),
    )


def calibrate_score_matrix(
    score_matrix: np.ndarray,
    calibrator: OutcomeCalibrator,
) -> np.ndarray:
    matrix = np.asarray(score_matrix, dtype=float)
    masks = (
        np.tril(np.ones_like(matrix, dtype=bool), k=-1),
        np.eye(matrix.shape[0], matrix.shape[1], dtype=bool),
        np.triu(np.ones_like(matrix, dtype=bool), k=1),
    )
    raw = np.array([matrix[mask].sum() for mask in masks], dtype=float)
    target = calibrator.transform(raw)
    calibrated = matrix.copy()
    for mask, raw_probability, target_probability in zip(masks, raw, target):
        if raw_probability > 0:
            calibrated[mask] *= target_probability / raw_probability
    return calibrated / calibrated.sum()

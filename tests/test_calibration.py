import unittest

import numpy as np

from src.calibration import (
    OutcomeCalibrator,
    calibrate_score_matrix,
    fit_outcome_calibrator,
    multiclass_log_loss,
)
from src.prediction import outcome_probabilities, poisson_score_matrix


class CalibrationTests(unittest.TestCase):
    def test_transform_returns_valid_probabilities(self):
        calibrator = OutcomeCalibrator(
            inverse_temperature=0.8,
            home_bias=0.1,
            draw_bias=0.0,
            away_bias=-0.1,
        )
        result = calibrator.transform([0.5, 0.3, 0.2])
        self.assertAlmostEqual(float(result.sum()), 1.0)
        self.assertTrue(np.all(result > 0))

    def test_fitted_calibrator_improves_overconfident_predictions(self):
        probabilities = np.tile([0.75, 0.15, 0.10], (300, 1))
        actual = np.zeros((300, 3))
        actual[:150, 0] = 1
        actual[150:240, 1] = 1
        actual[240:, 2] = 1
        raw_loss = multiclass_log_loss(probabilities, actual)
        calibrator = fit_outcome_calibrator(probabilities, actual)
        calibrated_loss = multiclass_log_loss(
            calibrator.transform(probabilities), actual
        )
        self.assertLess(calibrated_loss, raw_loss)

    def test_score_matrix_matches_calibrated_outcomes(self):
        matrix = poisson_score_matrix(1.7, 1.0)
        calibrator = OutcomeCalibrator(
            inverse_temperature=0.7,
            home_bias=-0.05,
            draw_bias=0.10,
            away_bias=-0.05,
        )
        raw_outcomes = np.array(outcome_probabilities(matrix))
        expected = calibrator.transform(raw_outcomes)
        calibrated_matrix = calibrate_score_matrix(matrix, calibrator)
        actual = np.array(outcome_probabilities(calibrated_matrix))
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertAlmostEqual(float(calibrated_matrix.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()

import pandas as pd
import pytest
from audit import INTERSECTION, fit_group_thresholds, apply_group_thresholds


def test_test_distribution_does_not_refit_thresholds():
    calibration = pd.DataFrame({INTERSECTION: ['A'] * 4, 'score': [0.1, 0.2, 0.3, 0.4]})
    thresholds = fit_group_thresholds(calibration, 0.25)
    test = pd.DataFrame({INTERSECTION: ['A'] * 4, 'score': [0.8, 0.85, 0.9, 0.95]})
    assert apply_group_thresholds(test, thresholds, 0.5).sum() == 4
    # Re-fitting to test scores would incorrectly force the selected count back to one.
    assert thresholds['A'] == pytest.approx(0.325)


def test_new_group_uses_frozen_global_threshold():
    test = pd.DataFrame({INTERSECTION: ['new', 'new'], 'score': [0.2, 0.8]})
    assert apply_group_thresholds(test, {'A': 0.9}, 0.5).tolist() == [0, 1]

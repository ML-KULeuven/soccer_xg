import numpy as np
import pytest

import soccer_xg.metrics as metrics_lib


def test_expected_calibration_error():
    np.random.seed(1)
    nsamples = 100
    probs = np.linspace(0, 1, nsamples)
    labels = np.random.rand(nsamples) < probs
    ece = metrics_lib.expected_calibration_error(labels, probs)
    bad_ece = metrics_lib.expected_calibration_error(labels, probs / 2)

    assert ece > 0 and ece < 1
    assert bad_ece > 0 and bad_ece < 1
    assert ece < bad_ece


def test_expected_calibration_error_all_wrong():
    n_bins = 90
    ece = metrics_lib.expected_calibration_error(np.ones(10), np.zeros(10), n_bins=n_bins)
    assert ece == pytest.approx(1.0)

    ece = metrics_lib.expected_calibration_error(np.zeros(10), np.ones(10), n_bins=n_bins)
    assert ece == pytest.approx(1.0)


def test_expected_calibration_error_all_right():
    n_bins = 90
    ece = metrics_lib.expected_calibration_error(np.ones(10), np.ones(10), n_bins=n_bins)
    assert ece == pytest.approx(0.0)

    ece = metrics_lib.expected_calibration_error(np.zeros(10), np.zeros(10), n_bins=n_bins)
    assert ece == pytest.approx(0.0)

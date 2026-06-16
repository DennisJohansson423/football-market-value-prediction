"""Tests for src/models.py — _metrics."""

import numpy as np
import pytest

from src.models import _metrics


class TestMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = _metrics(y, y)
        assert m["rmse_log"] == pytest.approx(0.0, abs=1e-10)
        assert m["mae_log"] == pytest.approx(0.0, abs=1e-10)
        assert m["r2_log"] == pytest.approx(1.0)
        assert m["rmse_eur"] == pytest.approx(0.0, abs=1e-10)
        assert m["mae_eur"] == pytest.approx(0.0, abs=1e-10)

    def test_rmse_log_known_value(self):
        # [0, 2] vs [0, 0] → MSE = (0² + 4²)/2 = 8 → RMSE = √8
        y_true = np.array([0.0, 4.0])
        y_pred = np.array([0.0, 0.0])
        m = _metrics(y_true, y_pred)
        assert m["rmse_log"] == pytest.approx(np.sqrt(8.0))

    def test_mae_log_known_value(self):
        # errors [1, 2, 3] → MAE = 2
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.0, 0.0, 0.0])
        m = _metrics(y_true, y_pred)
        assert m["mae_log"] == pytest.approx(2.0)

    def test_eur_uses_expm1(self):
        # log-space value of log(1+x): expm1(log(1+x)) = x
        y_log = np.log1p(np.array([1_000_000.0, 5_000_000.0]))
        m = _metrics(y_log, y_log)
        assert m["rmse_eur"] == pytest.approx(0.0, abs=1e-3)
        assert m["mae_eur"] == pytest.approx(0.0, abs=1e-3)

    def test_eur_conversion_non_trivial(self):
        # true = log1p(1e6), pred = log1p(0) — EUR error should reflect the full €1M gap
        y_true = np.array([np.log1p(1_000_000.0)])
        y_pred = np.array([0.0])
        m = _metrics(y_true, y_pred)
        assert m["mae_eur"] == pytest.approx(1_000_000.0, rel=1e-4)

    def test_spearman_positive_correlation(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = y + 0.5  # constant offset — perfect rank correlation
        m = _metrics(y, pred)
        assert m["spearman_log"] == pytest.approx(1.0)

    def test_spearman_negative_correlation(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pred = -y
        m = _metrics(y, pred)
        assert m["spearman_log"] == pytest.approx(-1.0)

    def test_returns_all_keys(self):
        y = np.array([1.0, 2.0, 3.0])
        m = _metrics(y, y)
        assert set(m.keys()) == {
            "rmse_log", "mae_log", "r2_log", "spearman_log", "rmse_eur", "mae_eur"
        }

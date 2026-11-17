"""Tests for walk-forward validation.

The property that matters most here is that no fitted parameter ever sees
its own test window. A leak would make the out-of-sample figures meaningless
while still looking plausible.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring import SCORE_COLUMNS                      # noqa: E402
from backtesting.engine import (                        # noqa: E402
    prepare_matrices,
    run_backtest,
    simulate,
)
from validation.walkforward import (                    # noqa: E402
    _factor_matrices,
    _score_matrix,
    _weight_grid,
    apply_weights,
    fit_weights,
    overfitting_gap,
    precompute_zscores,
    summarise,
    walk_forward,
    walk_forward_windows,
)


def synthetic_panel(n_days=900, seed=0, tickers=("A", "B", "C", "D")):
    """A panel carrying the columns the validator needs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n_days)
    frames = []
    for ticker in tickers:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n_days)))
        frames.append(pd.DataFrame({
            "date": dates,
            "ticker": ticker,
            "close": close,
            "return_5d": pd.Series(close).pct_change(5).to_numpy(),
            "trend_signal": rng.integers(0, 2, n_days),
            "volume_ratio": rng.lognormal(0.0, 0.3, n_days),
            "volatility_20d": rng.uniform(0.08, 0.40, n_days),
        }))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Window construction
# --------------------------------------------------------------------------

def test_test_windows_never_overlap():
    """Overlapping test windows would double-count the same out-of-sample
    dates and make the stitched series unusable."""
    dates = pd.bdate_range("2019-01-01", periods=1000)
    windows = walk_forward_windows(dates, train_days=252, test_days=63)
    seen = set()
    for _, test in windows:
        assert not (seen & set(test)), "test windows overlap"
        seen |= set(test)


def test_training_window_always_precedes_its_test_window():
    """The core no-leak property of the split itself."""
    dates = pd.bdate_range("2019-01-01", periods=1000)
    for train, test in walk_forward_windows(dates, 252, 63):
        assert train.max() < test.min()


def test_window_lengths_are_exact():
    dates = pd.bdate_range("2019-01-01", periods=1000)
    for train, test in walk_forward_windows(dates, 252, 63):
        assert len(train) == 252
        assert len(test) == 63


def test_too_little_history_gives_no_windows():
    dates = pd.bdate_range("2019-01-01", periods=100)
    assert walk_forward_windows(dates, 252, 63) == []


def test_walk_forward_refuses_insufficient_history():
    panel = synthetic_panel(n_days=200)
    with pytest.raises(ValueError, match="Not enough history"):
        walk_forward(panel, train_days=504, test_days=126)


# --------------------------------------------------------------------------
# Scoring plumbing
# --------------------------------------------------------------------------

def test_precompute_adds_a_zscore_per_factor():
    frame = precompute_zscores(synthetic_panel(n_days=300))
    for column in SCORE_COLUMNS:
        assert f"z_{column}" in frame.columns


def test_zscores_are_computed_within_each_date():
    """Each date's z-scores must centre on that date's own cross-section."""
    frame = precompute_zscores(synthetic_panel(n_days=300))
    for _, group in frame.groupby("date"):
        values = group["z_return_5d"].dropna()
        if len(values) > 1 and values.std(ddof=0) > 0:
            assert abs(values.mean()) < 1e-9


def test_apply_weights_is_a_linear_combination():
    frame = precompute_zscores(synthetic_panel(n_days=300))
    weights = {"return_5d": 0.5, "trend_signal": 0.2,
               "volume_ratio": 0.2, "volatility_20d": -0.1}
    scored = apply_weights(frame, weights)
    expected = sum(scored[f"z_{c}"] * w for c, w in weights.items())
    pd.testing.assert_series_equal(scored["quant_score"], expected,
                                   check_names=False)


def test_fast_grid_path_matches_run_backtest():
    """The search uses pre-pivoted matrices for speed; it must agree exactly
    with the ordinary path, or every fitted result is on a different engine
    from the one the tests cover."""
    frame = precompute_zscores(synthetic_panel(n_days=400))
    frame = frame.sort_values(["ticker", "date"])
    frame["next_day_return"] = (
        frame.groupby("ticker")["close"].shift(-1) / frame["close"] - 1
    )
    weights = {"return_5d": 0.3, "trend_signal": 0.3,
               "volume_ratio": 0.2, "volatility_20d": -0.2}

    prepared = _factor_matrices(frame, "monthly")
    fast = simulate(
        _score_matrix(prepared, weights), prepared["returns"],
        prepared["dates"], prepared["tickers"], prepared["is_rebalance"],
        top_n=2, cost_bps=5.0, with_holdings=False,
    )
    slow = run_backtest(apply_weights(frame, weights), top_n=2,
                        frequency="monthly", cost_bps=5.0)

    pd.testing.assert_series_equal(fast["net_return"], slow["net_return"])
    pd.testing.assert_series_equal(fast["turnover"], slow["turnover"])


def test_weight_grid_constrains_signs():
    """Momentum, trend and volume positive; volatility negative.

    Letting the optimiser flip these would fit noise instead of testing the
    strategy's own assertions about direction.
    """
    for weights in _weight_grid(0.2):
        assert weights["return_5d"] >= 0
        assert weights["trend_signal"] >= 0
        assert weights["volume_ratio"] >= 0
        assert weights["volatility_20d"] <= 0


def test_weight_grid_excludes_all_zero_weights():
    assert all(
        sum(abs(v) for v in w.values()) > 0 for w in _weight_grid(0.2)
    )


def test_fit_weights_returns_a_member_of_the_grid():
    frame = precompute_zscores(synthetic_panel(n_days=400))
    frame = frame.sort_values(["ticker", "date"])
    frame["next_day_return"] = (
        frame.groupby("ticker")["close"].shift(-1) / frame["close"] - 1
    )
    grid = _weight_grid(0.3)
    weights, sharpe = fit_weights(frame, grid, 2, "monthly", 5.0)
    assert weights in grid
    assert np.isfinite(sharpe)


@pytest.mark.slow
def test_fit_weights_picks_the_best_in_sample_sharpe():
    """Whatever it returns must be at least as good as every alternative."""
    frame = precompute_zscores(synthetic_panel(n_days=400, seed=3))
    frame = frame.sort_values(["ticker", "date"])
    frame["next_day_return"] = (
        frame.groupby("ticker")["close"].shift(-1) / frame["close"] - 1
    )
    grid = _weight_grid(0.3)
    best, best_sharpe = fit_weights(frame, grid, 2, "monthly", 5.0)

    from backtesting.engine import performance
    for weights in grid:
        result = run_backtest(apply_weights(frame, weights), top_n=2,
                              frequency="monthly", cost_bps=5.0)
        if result.empty:
            continue
        sharpe = performance(result["net_return"])["Sharpe Ratio"]
        assert sharpe <= best_sharpe + 1e-9


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_walk_forward_produces_aligned_series():
    panel = synthetic_panel(n_days=900, seed=5)
    result = walk_forward(panel, top_n=2, train_days=252, test_days=126,
                          grid_step=0.3)
    assert len(result["windows"]) >= 2
    assert result["oos_fitted"].index.equals(result["oos_fixed"].index)
    assert not result["oos_fitted"].empty


@pytest.mark.slow
def test_out_of_sample_dates_exclude_the_first_training_window():
    """Nothing before the first test window may appear in the OOS series."""
    panel = synthetic_panel(n_days=900, seed=6)
    result = walk_forward(panel, top_n=2, train_days=252, test_days=126,
                          grid_step=0.3)
    first_test_start = pd.Timestamp(
        result["windows"]["test_start"].iloc[0]
    )
    assert result["oos_fitted"].index.min() >= first_test_start


@pytest.mark.slow
def test_fitted_weights_differ_across_windows():
    """If every window picked identical weights the fit is not responding to
    its training data, and the exercise would be vacuous."""
    panel = synthetic_panel(n_days=1100, seed=9)
    result = walk_forward(panel, top_n=2, train_days=252, test_days=126,
                          grid_step=0.2)
    weight_columns = [c for c in result["windows"].columns
                      if c.startswith("w_")]
    distinct = result["windows"][weight_columns].drop_duplicates()
    assert len(distinct) > 1


@pytest.mark.slow
def test_summarise_covers_all_three_series():
    panel = synthetic_panel(n_days=900, seed=7)
    result = walk_forward(panel, top_n=2, train_days=252, test_days=126,
                          grid_step=0.3)
    table = summarise(result)
    assert set(table.index) == {
        "Fitted weights (OOS)", "Fixed weights (OOS)",
        "Equal weight (no signal)",
    }
    assert table["Sharpe Ratio"].notna().all()


@pytest.mark.slow
def test_overfitting_gap_is_in_sample_minus_out_of_sample():
    panel = synthetic_panel(n_days=900, seed=8)
    result = walk_forward(panel, top_n=2, train_days=252, test_days=126,
                          grid_step=0.3)
    table = overfitting_gap(result)
    expected = table["in_sample_sharpe"] - table["oos_sharpe_fitted"]
    pd.testing.assert_series_equal(table["gap"], expected, check_names=False)


@pytest.mark.slow
def test_in_sample_sharpe_beats_out_of_sample_on_noise():
    """On random data the fit has nothing real to find, so in-sample Sharpe
    is selection luck and should exceed out-of-sample on average.

    This is the test that would catch a leak: without one, the two are
    roughly equal, because the 'test' window was fitted too.
    """
    panel = synthetic_panel(n_days=1400, seed=11, tickers=("A","B","C","D","E"))
    result = walk_forward(panel, top_n=2, train_days=252, test_days=126,
                          grid_step=0.2)
    table = overfitting_gap(result)
    assert table["gap"].mean() > 0


def test_walk_forward_rejects_a_panel_missing_factors():
    panel = synthetic_panel(n_days=900).drop(columns=["volume_ratio"])
    with pytest.raises(ValueError, match="missing columns"):
        walk_forward(panel, train_days=252, test_days=126)

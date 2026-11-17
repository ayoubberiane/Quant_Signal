"""Tests for the QuantSignal feature and ranking code.

Values here are derived by hand where possible, so a regression shows up as
a wrong number rather than only as an exception.

Run from the repository root:  python -m pytest tests/ -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.factors import generate_features          # noqa: E402
from ranking.rank_assets import calculate_score          # noqa: E402
from backtesting.backtest import (                       # noqa: E402
    calculate_forward_returns,
    calculate_metrics,
    calculate_quant_score,
)


def make_frame(close, volume=None, ticker="TEST"):
    """Build a frame in the shape clean_data.py produces."""
    n = len(close)
    close = np.asarray(close, dtype=float)
    volume = np.full(n, 1_000_000.0) if volume is None else np.asarray(
        volume, dtype=float
    )
    return pd.DataFrame({
        "date": pd.bdate_range("2025-01-01", periods=n),
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": volume,
        "vwap": close,
        "ticker": ticker,
    })


# --------------------------------------------------------------------------
# generate_features
# --------------------------------------------------------------------------

def test_daily_return_is_simple_pct_change():
    df = generate_features(make_frame([100.0, 110.0, 99.0]))
    assert df["daily_return"].iloc[1] == pytest.approx(0.10)
    assert df["daily_return"].iloc[2] == pytest.approx(-0.10)
    assert pd.isna(df["daily_return"].iloc[0])


def test_return_5d_spans_five_rows_not_six():
    """pct_change(5) compares row i with row i-5."""
    close = [100.0, 101.0, 102.0, 103.0, 104.0, 150.0]
    df = generate_features(make_frame(close))
    assert df["return_5d"].iloc[5] == pytest.approx(0.50)
    assert df["return_5d"].iloc[:5].isna().all()


def test_moving_averages_need_full_windows():
    df = generate_features(make_frame(np.full(60, 100.0)))
    assert df["ma_20"].iloc[:19].isna().all()
    assert df["ma_20"].iloc[19] == pytest.approx(100.0)
    assert df["ma_50"].iloc[:49].isna().all()
    assert df["ma_50"].iloc[49] == pytest.approx(100.0)


def test_volatility_is_annualised_by_sqrt_252():
    """Recover a known daily sigma from the annualised figure."""
    rng = np.random.default_rng(3)
    daily_sigma = 0.01
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, daily_sigma, 400)))
    df = generate_features(make_frame(close))
    annualised = df["volatility_20d"].iloc[-1]
    assert annualised / np.sqrt(252) == pytest.approx(daily_sigma, rel=0.6)


def test_constant_price_has_zero_volatility():
    df = generate_features(make_frame(np.full(40, 100.0)))
    assert df["volatility_20d"].iloc[-1] == pytest.approx(0.0)


def test_volume_ratio_is_one_for_constant_volume():
    df = generate_features(make_frame(np.full(40, 100.0)))
    assert df["volume_ratio"].iloc[-1] == pytest.approx(1.0)


def test_volume_ratio_detects_a_spike():
    volume = np.concatenate([np.full(39, 1e6), [5e6]])
    df = generate_features(make_frame(np.full(40, 100.0), volume=volume))
    # Spike row is in its own 20-day mean, so the ratio is below the naive 5x.
    assert df["volume_ratio"].iloc[-1] > 3.0


def test_trend_signal_is_one_when_fast_ma_is_above_slow():
    rising = np.linspace(100.0, 200.0, 60)
    df = generate_features(make_frame(rising))
    assert df["trend_signal"].iloc[-1] == 1

    falling = np.linspace(200.0, 100.0, 60)
    df = generate_features(make_frame(falling))
    assert df["trend_signal"].iloc[-1] == 0


def test_trend_signal_is_zero_while_mas_are_nan():
    """`NaN > NaN` is False, so early rows read as 0 rather than missing.

    Worth pinning: a 0 here means 'not yet computable', not 'downtrend',
    and anything consuming trend_signal must drop those rows.
    """
    df = generate_features(make_frame(np.linspace(100.0, 200.0, 60)))
    assert df["trend_signal"].iloc[0] == 0
    assert pd.isna(df["ma_50"].iloc[0])


def test_momentum_signal_tracks_the_sign_of_return_5d():
    close = [100.0, 101.0, 102.0, 103.0, 104.0, 150.0, 90.0]
    df = generate_features(make_frame(close))
    assert df["momentum_signal"].iloc[5] == 1
    assert df["momentum_signal"].iloc[6] == 0


def test_generate_features_does_not_mutate_its_input():
    original = make_frame([100.0, 101.0, 102.0])
    before = original.columns.tolist()
    generate_features(original)
    assert original.columns.tolist() == before


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def test_calculate_score_drops_rows_with_missing_inputs():
    df = generate_features(make_frame(np.linspace(100.0, 150.0, 60)))
    scored = calculate_score(df)
    assert scored["return_5d"].notna().all()
    assert scored["volatility_20d"].notna().all()
    assert len(scored) < len(df)


def test_score_formula_matches_its_documented_weights():
    """Check the arithmetic against the weights written in the comment."""
    df = pd.DataFrame({
        "return_5d": [0.10],
        "trend_signal": [1],
        "volume_ratio": [1.5],
        "volatility_20d": [0.20],
    })
    expected = (0.30 * 0.10) + (0.30 * 1) + (0.20 * 1.5)
    expected += 0.20 * (1 / 1.20)
    assert calculate_score(df)["quant_score"].iloc[0] == pytest.approx(expected)


def test_ranking_and_backtest_scores_agree():
    """rank_assets and backtest each define the score; they must not drift."""
    df = pd.DataFrame({
        "return_5d": [0.05, -0.02],
        "trend_signal": [1, 0],
        "volume_ratio": [1.2, 0.9],
        "volatility_20d": [0.18, 0.33],
    })
    from_ranking = calculate_score(df)["quant_score"].to_numpy()
    from_backtest = calculate_quant_score(df)["quant_score"].to_numpy()
    np.testing.assert_allclose(from_ranking, from_backtest)


def test_calculate_quant_score_rejects_missing_columns():
    with pytest.raises(ValueError):
        calculate_quant_score(pd.DataFrame({"return_5d": [0.01]}))


@pytest.mark.xfail(
    reason=(
        "KNOWN DEFECT: the four terms are summed without standardisation, "
        "so each term's influence is set by its raw scale rather than by "
        "its weight. return_5d is order 0.01, trend_signal is 0 or 1, and "
        "volume_ratio is unbounded above. Momentum's 0.30 weight therefore "
        "moves the ranking far less than volume's 0.20. Fix by z-scoring or "
        "rank-transforming each term cross-sectionally before weighting."
    ),
    strict=True,
)
def test_momentum_outweighs_volume_as_the_weights_intend():
    """Momentum is weighted 0.30 and volume 0.20, so momentum should dominate.

    Asset A has far stronger momentum; asset B only has heavier volume. With
    the intended weighting A outranks B. It does not, because the terms are
    on different scales.
    """
    df = pd.DataFrame({
        "return_5d": [0.08, 0.00],      # A strongly ahead on momentum
        "trend_signal": [0, 0],          # tied
        "volume_ratio": [1.0, 2.0],      # B ahead on volume
        "volatility_20d": [0.20, 0.20],  # tied
    }, index=["A", "B"])
    scored = calculate_score(df)
    assert scored["quant_score"].loc["A"] > scored["quant_score"].loc["B"]


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------

def test_forward_return_looks_forward_exactly_one_day():
    df = pd.DataFrame({
        "ticker": ["X"] * 4,
        "date": pd.bdate_range("2025-01-01", periods=4),
        "close": [100.0, 110.0, 99.0, 99.0],
    })
    out = calculate_forward_returns(df)
    assert out["next_day_return"].iloc[0] == pytest.approx(0.10)
    assert out["next_day_return"].iloc[1] == pytest.approx(-0.10)
    assert pd.isna(out["next_day_return"].iloc[3])


def test_forward_return_does_not_leak_across_tickers():
    """The last row of one ticker must not borrow the next ticker's price."""
    df = pd.DataFrame({
        "ticker": ["A", "A", "B", "B"],
        "date": list(pd.bdate_range("2025-01-01", periods=2)) * 2,
        "close": [100.0, 101.0, 500.0, 505.0],
    })
    out = calculate_forward_returns(df).sort_values(["ticker", "date"])
    last_of_a = out[out["ticker"] == "A"].iloc[-1]
    assert pd.isna(last_of_a["next_day_return"])


def test_metrics_on_a_hand_computed_series():
    returns = pd.Series([0.10, -0.05, 0.02])
    metrics = calculate_metrics(pd.DataFrame({"strategy_return": returns}))
    expected_total = (1.10 * 0.95 * 1.02) - 1
    assert metrics["Total Return"] == pytest.approx(expected_total)
    assert metrics["Win Rate"] == pytest.approx(2 / 3)
    assert metrics["Maximum Drawdown"] == pytest.approx(-0.05)


def test_metrics_drawdown_is_never_positive():
    returns = pd.Series([0.01] * 10)
    metrics = calculate_metrics(pd.DataFrame({"strategy_return": returns}))
    assert metrics["Maximum Drawdown"] <= 0.0


def test_metrics_handle_zero_volatility_without_dividing_by_zero():
    returns = pd.Series([0.0, 0.0, 0.0])
    metrics = calculate_metrics(pd.DataFrame({"strategy_return": returns}))
    assert np.isfinite(metrics["Sharpe Ratio"])
    assert metrics["Sharpe Ratio"] == 0

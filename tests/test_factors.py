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
from scoring import (                                    # noqa: E402
    add_quant_score,
    effective_weights,
    rank_score,
    winsorise,
    zscore,
)
from ranking.rank_assets import calculate_score          # noqa: E402
from backtesting.backtest import (                       # noqa: E402
    calculate_forward_returns,
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


def test_score_is_a_weighted_sum_of_standardised_terms():
    """Check the arithmetic against the weights and the z-scored inputs."""
    df = pd.DataFrame({
        "return_5d": [0.10, -0.02, 0.01],
        "trend_signal": [1, 0, 1],
        "volume_ratio": [1.5, 0.8, 1.0],
        "volatility_20d": [0.20, 0.35, 0.25],
    }, index=["A", "B", "C"])

    scored = calculate_score(df)
    expected = (
        0.30 * scored["z_return_5d"]
        + 0.30 * scored["z_trend_signal"]
        + 0.20 * scored["z_volume_ratio"]
        - 0.20 * scored["z_volatility_20d"]
    )
    pd.testing.assert_series_equal(
        scored["quant_score"], expected, check_names=False
    )


def test_single_asset_scores_zero_not_nan():
    """A cross-section of one has no dispersion, so every z-score is 0."""
    df = pd.DataFrame({
        "return_5d": [0.10],
        "trend_signal": [1],
        "volume_ratio": [1.5],
        "volatility_20d": [0.20],
    })
    scored = calculate_score(df)
    assert scored["quant_score"].iloc[0] == pytest.approx(0.0)
    assert np.isfinite(scored["quant_score"]).all()


def test_ranking_and_backtest_scores_agree():
    """Both call sites share one definition in scoring.py; prove they match."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-06-02")] * 3,
        "return_5d": [0.05, -0.02, 0.01],
        "trend_signal": [1, 0, 1],
        "volume_ratio": [1.2, 0.9, 1.4],
        "volatility_20d": [0.18, 0.33, 0.21],
    })
    from_ranking = calculate_score(df)["quant_score"].to_numpy()
    from_backtest = calculate_quant_score(df)["quant_score"].to_numpy()
    np.testing.assert_allclose(from_ranking, from_backtest)


def test_backtest_score_requires_a_date_column():
    """Without dates there is no per-date cross-section to standardise in."""
    df = pd.DataFrame({
        "return_5d": [0.05, 0.01],
        "trend_signal": [1, 0],
        "volume_ratio": [1.2, 0.9],
        "volatility_20d": [0.18, 0.33],
    })
    with pytest.raises(ValueError, match="date"):
        calculate_quant_score(df)


def test_calculate_quant_score_rejects_missing_columns():
    with pytest.raises(ValueError):
        calculate_quant_score(pd.DataFrame({"return_5d": [0.01]}))


def test_momentum_outweighs_volume_as_the_weights_intend():
    """Momentum is weighted 0.30 and volume 0.20, so momentum should dominate.

    Asset A is ahead on momentum; asset B only on volume. This was the
    regression test for the scaling defect: before the terms were
    standardised, B won, because volume_ratio's raw scale outweighed
    momentum's larger coefficient.
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


# --------------------------------------------------------------------------
# Cross-sectional standardisation (src/scoring.py)
# --------------------------------------------------------------------------

def test_zscore_is_centred_and_unit_scale():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    z = zscore(series, winsorised=False)
    assert z.mean() == pytest.approx(0.0, abs=1e-12)
    assert z.std(ddof=0) == pytest.approx(1.0, abs=1e-12)


def test_zscore_of_a_constant_column_is_zero_not_inf():
    """The case that breaks a naive fix: binary trend_signal, all ones.

    In a broad uptrend every asset reads 1, the cross-sectional standard
    deviation is 0, and dividing by it would give inf or NaN and destroy
    the whole score, not just that term.
    """
    z = zscore(pd.Series([1, 1, 1, 1, 1]))
    assert (z == 0.0).all()
    assert np.isfinite(z).all()


def test_zscore_of_all_zeros_is_zero():
    z = zscore(pd.Series([0, 0, 0]))
    assert (z == 0.0).all()
    assert np.isfinite(z).all()


def test_binary_signal_still_separates_when_it_varies():
    z = zscore(pd.Series([1, 1, 0, 0]))
    assert z.iloc[0] > 0 and z.iloc[2] < 0


def test_winsorise_pulls_in_an_unbounded_volume_spike():
    series = pd.Series([0.9, 1.0, 1.1, 1.0, 0.95, 1.05, 40.0])
    assert winsorise(series).max() < 40.0


def test_rank_score_is_bounded_and_ignores_outlier_size():
    modest = rank_score(pd.Series([1.0, 2.0, 3.0, 4.0]))
    extreme = rank_score(pd.Series([1.0, 2.0, 3.0, 9999.0]))
    pd.testing.assert_series_equal(modest, extreme)
    assert modest.min() >= -1.0 and modest.max() <= 1.0


def test_higher_volatility_lowers_the_score():
    """The volatility weight is negative, so the penalty must bite."""
    df = pd.DataFrame({
        "return_5d": [0.05, 0.05, 0.05],
        "trend_signal": [1, 1, 1],
        "volume_ratio": [1.0, 1.0, 1.0],
        "volatility_20d": [0.10, 0.25, 0.60],
    }, index=["LOW", "MID", "HIGH"])
    scored = add_quant_score(df)
    assert (scored["quant_score"]["LOW"] > scored["quant_score"]["MID"]
            > scored["quant_score"]["HIGH"])


def test_per_date_standardisation_does_not_mix_dates():
    """Each date must be scored against its own cross-section only.

    Both dates hold the same relative pattern at different absolute levels.
    Standardised within date, the z-scores are identical across the two.
    Pooled, the later date's higher level would dominate: look-ahead.
    """
    df = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-02")] * 2
                + [pd.Timestamp("2025-01-03")] * 2,
        "return_5d": [0.01, 0.03, 0.51, 0.53],
        "trend_signal": [0, 1, 0, 1],
        "volume_ratio": [1.0, 1.2, 1.0, 1.2],
        "volatility_20d": [0.20, 0.25, 0.20, 0.25],
    })
    scored = add_quant_score(df, by="date")
    first = scored[scored["date"] == "2025-01-02"]["z_return_5d"].to_numpy()
    second = scored[scored["date"] == "2025-01-03"]["z_return_5d"].to_numpy()
    np.testing.assert_allclose(first, second)


def test_add_quant_score_rejects_an_unknown_method():
    df = pd.DataFrame({
        "return_5d": [0.01], "trend_signal": [1],
        "volume_ratio": [1.0], "volatility_20d": [0.2],
    })
    with pytest.raises(ValueError, match="unknown method"):
        add_quant_score(df, method="not_a_method")


def test_effective_weights_now_track_intended_weights():
    """The regression guard for the original bug.

    Before standardisation, return_5d's 30% weight moved about 3% of the
    ranking spread and volume_ratio's 20% moved about 55%. Each term's
    effective share must now sit close to its intended share.
    """
    rng = np.random.default_rng(17)
    n = 7
    df = pd.DataFrame({
        "return_5d": rng.normal(0.0, 0.02, n),
        "trend_signal": rng.integers(0, 2, n),
        "volume_ratio": rng.lognormal(0.0, 0.4, n),
        "volatility_20d": rng.uniform(0.10, 0.45, n),
    })
    table = effective_weights(df)
    gap = (table["effective"] - table["intended"]).abs().max()
    assert gap < 0.10, f"largest intended/effective gap is {gap:.1%}\n{table}"

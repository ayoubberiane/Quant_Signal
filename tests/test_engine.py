"""Tests for the backtest engine.

The arithmetic that matters here is turnover and cost accounting, because
an error there silently flatters every result the engine produces.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backtesting.engine import (                 # noqa: E402
    buy_and_hold,
    compare_frequencies,
    cost_sensitivity,
    equal_weight_benchmark,
    performance,
    run_backtest,
    target_weights,
)


def panel(dates, data):
    """Build a long panel. data maps ticker -> (scores, next_day_returns)."""
    rows = []
    for ticker, (scores, returns) in data.items():
        for date, score, ret in zip(dates, scores, returns):
            rows.append({"date": date, "ticker": ticker,
                         "quant_score": score, "next_day_return": ret})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Weighting
# --------------------------------------------------------------------------

def test_target_weights_are_equal_and_sum_to_one():
    w = target_weights(pd.Series({"A": 3.0, "B": 2.0, "C": 1.0}), top_n=2)
    assert set(w.index) == {"A", "B"}
    assert w.sum() == pytest.approx(1.0)
    assert w["A"] == pytest.approx(w["B"])


def test_target_weights_handles_fewer_assets_than_requested():
    """Weights must still sum to 1 when the universe is smaller than top_n."""
    w = target_weights(pd.Series({"A": 1.0}), top_n=5)
    assert w.sum() == pytest.approx(1.0)


def test_target_weights_ignores_nan_scores():
    w = target_weights(pd.Series({"A": 1.0, "B": np.nan, "C": 0.5}), top_n=2)
    assert "B" not in w.index


# --------------------------------------------------------------------------
# Turnover and cost
# --------------------------------------------------------------------------

def test_first_rebalance_turnover_is_one_from_cash():
    """Going from all cash into one position trades 100% of the book."""
    dates = pd.bdate_range("2025-01-01", periods=1)
    p = panel(dates, {"A": ([1.0], [0.0]), "B": ([0.0], [0.0])})
    result = run_backtest(p, top_n=1, frequency="daily", cost_bps=0.0)
    assert result["turnover"].iloc[0] == pytest.approx(1.0)


def test_full_switch_between_assets_is_turnover_two():
    """Selling all of A and buying all of B trades twice the book.

    This is the two-way convention: cost is turnover * bps, with no hidden
    factor of two anywhere else in the engine.
    """
    dates = pd.bdate_range("2025-01-01", periods=2)
    p = panel(dates, {"A": ([1.0, 0.0], [0.0, 0.0]),
                      "B": ([0.0, 1.0], [0.0, 0.0])})
    result = run_backtest(p, top_n=1, frequency="daily", cost_bps=0.0)
    assert result["turnover"].iloc[1] == pytest.approx(2.0)


def test_unchanged_holdings_cost_nothing():
    dates = pd.bdate_range("2025-01-01", periods=3)
    p = panel(dates, {"A": ([1.0] * 3, [0.0] * 3),
                      "B": ([0.0] * 3, [0.0] * 3)})
    result = run_backtest(p, top_n=1, frequency="daily", cost_bps=10.0)
    assert result["turnover"].iloc[1:].sum() == pytest.approx(0.0)
    assert result["cost"].iloc[1:].sum() == pytest.approx(0.0)


def test_cost_is_turnover_times_bps():
    dates = pd.bdate_range("2025-01-01", periods=2)
    p = panel(dates, {"A": ([1.0, 0.0], [0.0, 0.0]),
                      "B": ([0.0, 1.0], [0.0, 0.0])})
    result = run_backtest(p, top_n=1, frequency="daily", cost_bps=25.0)
    expected = result["turnover"] * 25.0 / 10_000.0
    pd.testing.assert_series_equal(result["cost"], expected,
                                   check_names=False)


def test_net_return_is_gross_minus_cost():
    dates = pd.bdate_range("2025-01-01", periods=5)
    rng = np.random.default_rng(1)
    p = panel(dates, {
        "A": (rng.normal(size=5), rng.normal(0, 0.01, 5)),
        "B": (rng.normal(size=5), rng.normal(0, 0.01, 5)),
    })
    result = run_backtest(p, top_n=1, frequency="daily", cost_bps=10.0)
    expected = result["gross_return"] - result["cost"]
    pd.testing.assert_series_equal(result["net_return"], expected,
                                   check_names=False)


def test_higher_costs_never_improve_net_return():
    dates = pd.bdate_range("2025-01-01", periods=60)
    rng = np.random.default_rng(4)
    p = panel(dates, {
        t: (rng.normal(size=60), rng.normal(0.0004, 0.012, 60))
        for t in ("A", "B", "C", "D")
    })
    cheap = run_backtest(p, top_n=2, frequency="daily", cost_bps=1.0)
    dear = run_backtest(p, top_n=2, frequency="daily", cost_bps=50.0)
    assert dear["net_return"].sum() < cheap["net_return"].sum()


def test_zero_cost_net_equals_gross():
    dates = pd.bdate_range("2025-01-01", periods=20)
    rng = np.random.default_rng(8)
    p = panel(dates, {t: (rng.normal(size=20), rng.normal(0, 0.01, 20))
                      for t in ("A", "B")})
    result = run_backtest(p, top_n=1, frequency="daily", cost_bps=0.0)
    pd.testing.assert_series_equal(result["net_return"],
                                   result["gross_return"],
                                   check_names=False)


# --------------------------------------------------------------------------
# Rebalance frequency
# --------------------------------------------------------------------------

def test_monthly_rebalancing_trades_less_than_daily():
    """The whole point of the frequency control: less trading, less cost."""
    dates = pd.bdate_range("2025-01-01", periods=120)
    rng = np.random.default_rng(12)
    p = panel(dates, {
        t: (rng.normal(size=120), rng.normal(0.0004, 0.012, 120))
        for t in ("A", "B", "C", "D", "E")
    })
    daily = run_backtest(p, top_n=2, frequency="daily", cost_bps=5.0)
    monthly = run_backtest(p, top_n=2, frequency="monthly", cost_bps=5.0)
    assert monthly["turnover"].sum() < daily["turnover"].sum()
    assert monthly["cost"].sum() < daily["cost"].sum()


def test_monthly_rebalances_only_on_a_few_dates():
    dates = pd.bdate_range("2025-01-01", periods=120)
    rng = np.random.default_rng(13)
    p = panel(dates, {t: (rng.normal(size=120), np.zeros(120))
                      for t in ("A", "B")})
    result = run_backtest(p, top_n=1, frequency="monthly", cost_bps=0.0)
    # Roughly six months of business days.
    assert 3 <= int(result["rebalanced"].sum()) <= 8


def test_unknown_frequency_is_rejected():
    dates = pd.bdate_range("2025-01-01", periods=2)
    p = panel(dates, {"A": ([1.0, 1.0], [0.0, 0.0])})
    with pytest.raises(ValueError, match="unknown frequency"):
        run_backtest(p, frequency="fortnightly")


# --------------------------------------------------------------------------
# Correctness of the simulation
# --------------------------------------------------------------------------

def test_single_holding_return_equals_that_asset_return():
    """With one holding and no cost, the portfolio is that asset."""
    dates = pd.bdate_range("2025-01-01", periods=4)
    returns = [0.01, -0.02, 0.03, 0.00]
    p = panel(dates, {"A": ([1.0] * 4, returns),
                      "B": ([0.0] * 4, [0.0] * 4)})
    result = run_backtest(p, top_n=1, frequency="daily", cost_bps=0.0)
    np.testing.assert_allclose(result["gross_return"].to_numpy(), returns,
                               atol=1e-12)


def test_equal_weight_of_two_assets_is_their_average():
    dates = pd.bdate_range("2025-01-01", periods=1)
    p = panel(dates, {"A": ([2.0], [0.10]), "B": ([1.0], [0.20])})
    result = run_backtest(p, top_n=2, frequency="daily", cost_bps=0.0)
    assert result["gross_return"].iloc[0] == pytest.approx(0.15)


def test_holdings_count_matches_top_n():
    dates = pd.bdate_range("2025-01-01", periods=3)
    rng = np.random.default_rng(21)
    p = panel(dates, {t: (rng.normal(size=3), np.zeros(3))
                      for t in ("A", "B", "C", "D", "E")})
    result = run_backtest(p, top_n=3, frequency="daily", cost_bps=0.0)
    assert (result["n_holdings"] == 3).all()


def test_strategy_picks_the_highest_scoring_asset():
    """B scores highest, so its return is the one earned."""
    dates = pd.bdate_range("2025-01-01", periods=1)
    p = panel(dates, {
        "A": ([0.1], [-0.50]),
        "B": ([9.9], [0.07]),
        "C": ([0.5], [-0.30]),
    })
    result = run_backtest(p, top_n=1, frequency="daily", cost_bps=0.0)
    assert result["gross_return"].iloc[0] == pytest.approx(0.07)
    assert result["holdings"].iloc[0] == "B"


def test_no_lookahead_truncating_the_panel_keeps_earlier_rows_identical():
    """Results up to date t must not change when later dates are appended.

    The strongest available guard: if any step used future information, the
    early rows would move once more data arrived.
    """
    dates = pd.bdate_range("2025-01-01", periods=120)
    rng = np.random.default_rng(31)
    data = {t: (rng.normal(size=120), rng.normal(0.0004, 0.012, 120))
            for t in ("A", "B", "C")}
    full = panel(dates, data)
    cut = full[full["date"] <= dates[59]]

    long_run = run_backtest(full, top_n=2, frequency="monthly", cost_bps=5.0)
    short_run = run_backtest(cut, top_n=2, frequency="monthly", cost_bps=5.0)

    # Monthly rebalancing is anchored to each period's last available date,
    # so truncation mid-month legitimately moves that month's rebalance.
    # Compare only the dates before the final period of the short run.
    boundary = short_run.index[short_run["rebalanced"]].max()
    overlap = short_run.index[short_run.index < boundary]
    pd.testing.assert_series_equal(
        long_run.loc[overlap, "net_return"],
        short_run.loc[overlap, "net_return"],
    )


def test_missing_columns_are_rejected():
    df = pd.DataFrame({"date": [pd.Timestamp("2025-01-02")],
                       "ticker": ["A"]})
    with pytest.raises(ValueError, match="missing columns"):
        run_backtest(df)


def test_empty_panel_returns_empty_frame():
    df = pd.DataFrame(columns=["date", "ticker", "quant_score",
                               "next_day_return"])
    assert run_backtest(df).empty


# --------------------------------------------------------------------------
# Performance statistics
# --------------------------------------------------------------------------

def test_total_return_compounds():
    returns = pd.Series([0.10, -0.05, 0.02])
    stats = performance(returns)
    assert stats["Total Return"] == pytest.approx(1.10 * 0.95 * 1.02 - 1)


def test_drawdown_on_a_hand_built_path():
    """Up 20%, down 50%: peak-to-trough is exactly -50%."""
    returns = pd.Series([0.20, -0.50, 0.10])
    assert performance(returns)["Maximum Drawdown"] == pytest.approx(-0.50)


def test_drawdown_is_never_positive():
    stats = performance(pd.Series([0.01] * 20))
    assert stats["Maximum Drawdown"] <= 0.0


def test_sharpe_is_zero_when_volatility_is_zero():
    stats = performance(pd.Series([0.0, 0.0, 0.0]))
    assert stats["Sharpe Ratio"] == 0.0
    assert np.isfinite(stats["Sharpe Ratio"])


def test_sortino_ignores_upside_volatility():
    """A series with no losses has no downside deviation, so no division."""
    stats = performance(pd.Series([0.01, 0.05, 0.02, 0.09]))
    assert np.isfinite(stats["Sortino Ratio"])


def test_win_rate_counts_positive_periods():
    assert performance(pd.Series([0.01, -0.01, 0.01]))["Win Rate"] == \
        pytest.approx(2 / 3)


def test_performance_on_empty_series_is_all_zero():
    stats = performance(pd.Series(dtype=float))
    assert stats["Total Return"] == 0.0
    assert stats["Observations"] == 0


# --------------------------------------------------------------------------
# Benchmarks and diagnostics
# --------------------------------------------------------------------------

def test_equal_weight_benchmark_is_the_cross_sectional_mean():
    dates = pd.bdate_range("2025-01-01", periods=1)
    p = panel(dates, {"A": ([0.0], [0.10]), "B": ([0.0], [0.20]),
                      "C": ([0.0], [0.30])})
    assert equal_weight_benchmark(p).iloc[0] == pytest.approx(0.20)


def test_buy_and_hold_selects_one_ticker():
    dates = pd.bdate_range("2025-01-01", periods=2)
    p = panel(dates, {"SPY": ([0.0, 0.0], [0.01, 0.02]),
                      "QQQ": ([0.0, 0.0], [0.50, 0.50])})
    series = buy_and_hold(p, "SPY")
    np.testing.assert_allclose(series.to_numpy(), [0.01, 0.02])


def test_buy_and_hold_of_a_missing_ticker_is_empty():
    dates = pd.bdate_range("2025-01-01", periods=1)
    p = panel(dates, {"SPY": ([0.0], [0.01])})
    assert buy_and_hold(p, "NOPE").empty


def test_cost_sensitivity_returns_a_row_per_cost_and_decreases():
    dates = pd.bdate_range("2025-01-01", periods=80)
    rng = np.random.default_rng(44)
    p = panel(dates, {t: (rng.normal(size=80), rng.normal(0.0004, 0.012, 80))
                      for t in ("A", "B", "C")})
    table = cost_sensitivity(p, top_n=1, frequency="daily",
                             sweep=(0.0, 10.0, 50.0))
    assert len(table) == 3
    returns = table["Annualised Return"].to_numpy()
    assert returns[0] >= returns[1] >= returns[2]


def test_compare_frequencies_covers_every_schedule():
    dates = pd.bdate_range("2025-01-01", periods=120)
    rng = np.random.default_rng(55)
    p = panel(dates, {t: (rng.normal(size=120), rng.normal(0.0004, 0.012, 120))
                      for t in ("A", "B", "C")})
    table = compare_frequencies(p, top_n=2, cost_bps=5.0)
    assert set(table.index) == {"daily", "weekly", "monthly"}
    assert table.loc["daily", "Annual Turnover"] > \
        table.loc["monthly", "Annual Turnover"]

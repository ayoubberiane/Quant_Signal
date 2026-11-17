"""Tests for the Chronos factor plumbing and the Yahoo adapter.

The model itself is replaced by a fake, so these run without torch or a
download. What matters is that no forecast can see a price after its own
date, and that the forecast reaches the backtest engine intact.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from forecasting import chronos_signal as cs  # noqa: E402
from ingestion.yahoo import COLUMNS, normalise  # noqa: E402


def closes_frame(n=300, tickers=("AAA", "BBB", "CCC")):
    dates = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(0)
    data = {t: 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
            for t in tickers}
    return pd.DataFrame(data, index=dates)


class RecordingPredict:
    """Fake model: records each context, forecasts a fixed drift per series."""

    def __init__(self, drift_by_last=None):
        self.calls = []

    def __call__(self, contexts, horizon):
        self.calls.append([np.array(c) for c in contexts])
        out = np.empty((len(contexts), horizon, 3))
        for i, context in enumerate(contexts):
            last = context[-1]
            drift = 0.01 * (i + 1)  # series i drifts up by (i+1)%
            path = last * (1 + drift * np.arange(1, horizon + 1) / horizon)
            out[i, :, 0], out[i, :, 1], out[i, :, 2] = path * 0.9, path, path * 1.1
        return out


# --------------------------------------------------------------------------
# No look-ahead
# --------------------------------------------------------------------------

def test_context_stops_at_as_of_date():
    closes = closes_frame()
    as_of = closes.index[200]
    contexts = cs.contexts_as_of(closes, as_of, length=1000, min_length=1)
    for ticker, context in contexts.items():
        assert len(context) == 201
        assert context[-1] == closes.loc[as_of, ticker]


def test_future_prices_cannot_change_a_forecast():
    closes = closes_frame()
    dates = [closes.index[150], closes.index[200]]
    first = cs.expected_returns(closes, RecordingPredict(), dates, horizon=5)

    shocked = closes.copy()
    shocked.iloc[201:] *= 10  # rewrite everything after the last as-of date
    second = cs.expected_returns(shocked, RecordingPredict(), dates, horizon=5)

    pd.testing.assert_frame_equal(first, second)


def test_short_histories_are_left_out():
    closes = closes_frame()
    closes.loc[: closes.index[250], "CCC"] = np.nan
    contexts = cs.contexts_as_of(closes, closes.index[200], min_length=128)
    assert set(contexts) == {"AAA", "BBB"}


def test_context_is_truncated_to_length():
    closes = closes_frame()
    contexts = cs.contexts_as_of(closes, closes.index[-1], length=64)
    assert all(len(c) == 64 for c in contexts.values())


# --------------------------------------------------------------------------
# Forecast arithmetic
# --------------------------------------------------------------------------

def test_expected_return_is_log_of_median_over_last():
    closes = closes_frame()
    as_of = closes.index[-1]
    result = cs.expected_returns(closes, RecordingPredict(), [as_of], horizon=10)
    # The fake drifts series i by (i+1)% over the horizon.
    expected = np.log([1.01, 1.02, 1.03])
    np.testing.assert_allclose(result.loc[as_of].to_numpy(), expected)


def test_one_model_call_per_date():
    closes = closes_frame()
    predict = RecordingPredict()
    cs.expected_returns(closes, predict, closes.index[[150, 200, 250]], 5)
    assert len(predict.calls) == 3
    assert all(len(batch) == 3 for batch in predict.calls)


def test_forecast_bands_shape_and_order():
    closes = closes_frame()
    bands = cs.forecast_bands(closes, RecordingPredict(), horizon=7)
    assert set(bands) == set(closes.columns)
    band = bands["AAA"]
    assert len(band) == 7
    assert band.index.min() > closes.index.max()
    assert (band["p10"] <= band["p50"]).all()
    assert (band["p50"] <= band["p90"]).all()


# --------------------------------------------------------------------------
# Into the backtest panel
# --------------------------------------------------------------------------

def test_forecast_panel_carries_forward_without_looking_ahead():
    dates = pd.bdate_range("2024-01-01", periods=6)
    panel = pd.DataFrame([
        {"date": d, "ticker": t, "quant_score": 99.0, "next_day_return": 0.0}
        for d in dates for t in ("AAA", "BBB")
    ])
    forecasts = pd.DataFrame(
        {"AAA": [0.1, 0.3], "BBB": [0.2, 0.0]},
        index=[dates[1], dates[4]],
    )
    out = cs.forecast_panel(panel, forecasts).set_index(["date", "ticker"])

    assert out.index.get_level_values("date").min() == dates[1]
    assert out.loc[(dates[1], "AAA"), "quant_score"] == 0.1
    assert out.loc[(dates[3], "AAA"), "quant_score"] == 0.1  # carried
    assert out.loc[(dates[4], "AAA"), "quant_score"] == 0.3  # updated
    assert out.loc[(dates[5], "BBB"), "quant_score"] == 0.0
    assert not (out["quant_score"] == 99.0).any()


# --------------------------------------------------------------------------
# Information coefficient
# --------------------------------------------------------------------------

def test_ic_is_one_for_a_perfect_signal_and_minus_one_for_inverse():
    closes = closes_frame(n=60, tickers=("A", "B", "C", "D"))
    horizon = 5
    realised = np.log(closes.shift(-horizon) / closes)
    dates = closes.index[[10, 20, 30]]
    perfect = realised.loc[dates]

    assert cs.information_coefficient(perfect, closes, horizon).eq(1.0).all()
    assert cs.information_coefficient(-perfect, closes, horizon).eq(-1.0).all()


def test_ic_skips_dates_without_a_full_horizon():
    closes = closes_frame(n=60, tickers=("A", "B", "C", "D"))
    signal = pd.DataFrame(1.0, index=closes.index[[-3]], columns=closes.columns)
    signal.iloc[0] = [1, 2, 3, 4]
    assert cs.information_coefficient(signal, closes, horizon=5).empty


# --------------------------------------------------------------------------
# Yahoo adapter
# --------------------------------------------------------------------------

def test_normalise_matches_the_cleaning_contract():
    index = pd.DatetimeIndex(["2025-01-02", "2025-01-03"], name="Date")
    columns = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Open", "Volume"], ["SPY"]],
        names=["Price", "Ticker"],
    )
    raw = pd.DataFrame(
        [[10, 12, 9, 10, 100], [11, 12, 10, 10, 200]],
        index=index, columns=columns, dtype=float,
    )
    out = normalise(raw, "SPY")
    assert list(out.columns) == COLUMNS
    assert (out["ticker"] == "SPY").all()
    assert out["vwap"].iloc[0] == pytest.approx((12 + 9 + 10) / 3)
    assert out["date"].is_monotonic_increasing

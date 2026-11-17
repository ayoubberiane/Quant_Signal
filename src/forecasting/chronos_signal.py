"""Chronos-Bolt price forecasts, and a walk-forward factor built from them.

Chronos-Bolt (amazon/chronos-bolt-small on Hugging Face) is a pretrained
time-series model: it forecasts a series it has never seen, with no
training. Here it is treated as a hypothesis to test, not as an oracle.
Its forecast becomes one more factor and goes through the same backtest,
costs and benchmarks as the hand-built score.

NO LOOK-AHEAD
    A forecast made on date t sees closing prices up to and including t,
    and nothing after. `contexts_as_of` is the only place contexts are cut,
    and the tests pin it. The backtest engine then pays a weight set on t
    the return from t to t+1, as it does for every other score.

THE MODEL IS INJECTED
    Every function takes a `predict` callable rather than loading the model
    itself. The tests pass a fake, so they and CI run without torch or a
    model download. `load_chronos` builds the real one.

        predict(contexts: list[np.ndarray], horizon: int) -> np.ndarray
            shape (n_series, horizon, 3): the 10th, 50th and 90th
            percentile forecast price at each step ahead.
"""

from __future__ import annotations

import threading
from functools import lru_cache
from typing import Callable, Iterable

import numpy as np
import pandas as pd

MODEL_ID = "amazon/chronos-bolt-small"
QUANTILES = (0.1, 0.5, 0.9)

# Chronos-Bolt reads at most 2048 points; 512 trading days is two years,
# plenty for a daily price series and quicker on CPU.
CONTEXT_LENGTH = 512

# Below this many observations a forecast is mostly noise, so the asset is
# left out of that date's cross-section rather than scored on it.
MIN_CONTEXT = 128

# Bolt was trained for horizons up to 64 steps.
MAX_HORIZON = 64

Predict = Callable[[list, int], np.ndarray]


# Loading a transformers model briefly switches torch's default device to
# "meta" for the whole process, so a forecast running in another thread at
# that moment fails. One lock covers loading and every forecast; a web app
# serves several requests at once.
_MODEL_LOCK = threading.Lock()


def load_chronos(model_id: str = MODEL_ID) -> Predict:
    """Download (once) and wrap the Chronos-Bolt pipeline. Thread-safe."""
    with _MODEL_LOCK:
        return _load_chronos(model_id)


@lru_cache(maxsize=1)
def _load_chronos(model_id: str) -> Predict:
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float32
    )

    def predict(contexts: list, horizon: int) -> np.ndarray:
        if not 1 <= horizon <= MAX_HORIZON:
            raise ValueError(f"horizon must be 1..{MAX_HORIZON}, got {horizon}")
        tensors = [torch.tensor(np.asarray(c, dtype=np.float32))
                   for c in contexts]
        with _MODEL_LOCK:
            quantiles, _mean = pipeline.predict_quantiles(
                tensors,
                prediction_length=horizon,
                quantile_levels=list(QUANTILES),
            )
        return quantiles.numpy()

    return predict


def contexts_as_of(
    closes: pd.DataFrame,
    as_of: pd.Timestamp,
    length: int = CONTEXT_LENGTH,
    min_length: int = MIN_CONTEXT,
) -> dict[str, np.ndarray]:
    """Each asset's closing prices up to and including `as_of`.

    closes   wide frame: one row per date, one column per ticker.
    Assets with fewer than `min_length` observations are left out.
    """
    history = closes.loc[:as_of]
    out = {}
    for ticker in history.columns:
        series = history[ticker].dropna()
        if len(series) >= min_length:
            out[ticker] = series.to_numpy()[-length:]
    return out


def forecast_bands(
    closes: pd.DataFrame,
    predict: Predict,
    horizon: int = 20,
    as_of: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Forecast fan per asset: p10, p50, p90 price for each step ahead.

    The index is business days after `as_of`. That is an approximation for
    crypto and FX, which also trade at weekends, and is only used to draw
    the chart.
    """
    as_of = closes.index.max() if as_of is None else pd.Timestamp(as_of)
    contexts = contexts_as_of(closes, as_of)
    if not contexts:
        return {}

    tickers = list(contexts)
    quantiles = predict([contexts[t] for t in tickers], horizon)
    future = pd.bdate_range(as_of + pd.Timedelta(days=1), periods=horizon)

    return {
        ticker: pd.DataFrame(
            quantiles[i], index=future, columns=["p10", "p50", "p90"]
        )
        for i, ticker in enumerate(tickers)
    }


def expected_returns(
    closes: pd.DataFrame,
    predict: Predict,
    dates: Iterable[pd.Timestamp],
    horizon: int = 20,
    progress: Callable[[float, str], None] | None = None,
) -> pd.DataFrame:
    """Walk-forward forecast return per asset on each date.

    Returns a frame indexed by date, one column per ticker, holding
    log(median forecast at the horizon / last close). Each row uses only
    prices up to its own date. One batched model call per date.
    """
    dates = sorted(pd.Timestamp(d) for d in dates)
    rows = {}
    for step, as_of in enumerate(dates):
        contexts = contexts_as_of(closes, as_of)
        if not contexts:
            continue
        tickers = list(contexts)
        quantiles = predict([contexts[t] for t in tickers], horizon)
        last = np.array([contexts[t][-1] for t in tickers])
        median_end = quantiles[:, -1, 1]
        rows[as_of] = pd.Series(np.log(median_end / last), index=tickers)
        if progress is not None:
            progress((step + 1) / len(dates), f"Forecasting {as_of.date()}")

    if not rows:
        return pd.DataFrame(columns=closes.columns, dtype=float)
    return pd.DataFrame(rows).T.reindex(columns=closes.columns).sort_index()


def forecast_panel(panel: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Replace `quant_score` in a backtest panel with the forecast return.

    Each forecast is carried forward until the next one, per asset. That
    uses only information already known, and keeps the daily rows the
    engine needs to compound returns between rebalances. Dates before the
    first forecast are dropped.
    """
    long = (forecasts.rename_axis("date").reset_index()
            .melt(id_vars="date", var_name="ticker", value_name="forecast"))
    out = panel.drop(columns=["quant_score"], errors="ignore").merge(
        long, on=["date", "ticker"], how="left"
    )
    out = out.sort_values(["ticker", "date"])
    out["quant_score"] = out.groupby("ticker")["forecast"].ffill()
    first = forecasts.index.min()
    return out[out["date"] >= first].drop(columns=["forecast"])


def information_coefficient(
    signal: pd.DataFrame, closes: pd.DataFrame, horizon: int
) -> pd.Series:
    """Rank correlation, per date, between a signal and what happened next.

    signal   date x ticker. Higher should mean a better next `horizon` days.
    Compared against the realised log return from each date over the next
    `horizon` rows of `closes`. Dates without a full horizon ahead are
    dropped, so this never scores against prices it cannot see.

    An IC averaging near zero means the signal ordered the assets no better
    than chance.
    """
    realised = np.log(closes.shift(-horizon) / closes)
    values = {}
    for date, row in signal.iterrows():
        if date not in realised.index:
            continue
        pair = pd.concat([row, realised.loc[date]], axis=1).dropna()
        if len(pair) >= 3:
            values[date] = pair.iloc[:, 0].corr(pair.iloc[:, 1],
                                                method="spearman")
    return pd.Series(values, dtype=float, name="IC")

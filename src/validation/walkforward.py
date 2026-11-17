"""Walk-forward out-of-sample validation of the score weights.

WHAT THIS ANSWERS
    The four score weights were chosen by hand and never tested out of
    sample. Two separate questions follow from that, and this module keeps
    them apart:

    1. Do the fixed, hand-chosen weights work out of sample?
    2. Does *fitting* the weights to history beat leaving them fixed?

    The second question is the one that catches overfitting. Fitting four
    weights on a handful of assets can reproduce almost any past, so a
    strong in-sample result is nearly guaranteed and means nothing. The
    gap between in-sample and out-of-sample performance is the measurement
    that matters, and a large gap is the finding, not a failure.

METHOD
    The sample is cut into consecutive windows. In each one, weights are
    fitted on the training segment only and then applied, unchanged, to the
    test segment that follows it. Test segments never overlap, so the
    concatenated out-of-sample returns form a single series that no fitted
    parameter ever saw.

    Three series come out, over identical dates:
      fitted  - weights refitted in each training window
      fixed   - the hand-chosen weights, never fitted
      equal   - equal weight across the universe, no signal at all

    If `fitted` does not beat `fixed`, fitting is adding nothing. If
    neither beats `equal`, the signal has not demonstrated anything.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoring import SCORE_COLUMNS, WEIGHTS, zscore  # noqa: E402
from backtesting.engine import (  # noqa: E402
    DEFAULT_COST_BPS,
    _rebalance_dates,
    equal_weight_benchmark,
    performance,
    run_backtest,
    simulate,
)

# Training and test lengths in business days. Two years of training leaves
# enough history for the factors to vary across regimes; six months of test
# keeps the out-of-sample series long enough to be worth reading.
TRAIN_DAYS = 504
TEST_DAYS = 126


def _weight_grid(step: float = 0.1) -> list[dict[str, float]]:
    """Candidate weight sets on a coarse grid.

    Deliberately coarse. A fine grid would fit the training data more
    closely and so overstate in-sample performance without improving what
    happens out of sample; it would also invite reading precision into
    weights that carry none.

    Signs are constrained rather than searched: momentum, trend and volume
    enter positively and volatility negatively. These are the directions the
    strategy asserts, so letting the optimiser flip them would be fitting
    noise rather than testing the idea.
    """
    levels = [round(x * step, 4) for x in range(0, int(0.6 / step) + 1)]
    grid = []
    for mom, trend, volume, vol in itertools.product(levels, repeat=4):
        total = mom + trend + volume + vol
        if total <= 0:
            continue
        grid.append({
            "return_5d": mom,
            "trend_signal": trend,
            "volume_ratio": volume,
            "volatility_20d": -vol,
        })
    return grid


def precompute_zscores(panel: pd.DataFrame) -> pd.DataFrame:
    """Add per-date z-scores once, so scoring a weight set is just a sum.

    The z-scores do not depend on the weights, so computing them once makes
    a grid search over weights cheap. Standardisation stays within each
    date: pooling dates would leak a later distribution into an earlier
    score.
    """
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=SCORE_COLUMNS)
    for column in SCORE_COLUMNS:
        frame[f"z_{column}"] = frame.groupby("date")[column].transform(zscore)
    return frame


def apply_weights(frame: pd.DataFrame, weights: dict[str, float]
                  ) -> pd.DataFrame:
    """Score a pre-z-scored frame with one weight set."""
    out = frame.copy()
    out["quant_score"] = sum(
        out[f"z_{column}"] * weight for column, weight in weights.items()
    )
    return out


def walk_forward_windows(
    dates: pd.DatetimeIndex,
    train_days: int = TRAIN_DAYS,
    test_days: int = TEST_DAYS,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Consecutive (train, test) date splits with non-overlapping tests."""
    dates = pd.DatetimeIndex(sorted(dates))
    windows = []
    start = 0
    while start + train_days + test_days <= len(dates):
        train = dates[start:start + train_days]
        test = dates[start + train_days:start + train_days + test_days]
        windows.append((train, test))
        start += test_days
    return windows


def _factor_matrices(frame: pd.DataFrame, frequency: str):
    """Pivot each z-score column once, for reuse across every weight set.

    A weight set then only needs a linear combination of these matrices,
    which turns the search from thousands of panel reshapes into thousands
    of small matrix sums. This is what makes the search finish in seconds
    rather than an hour.
    """
    usable = frame.dropna(subset=["next_day_return"]).copy()
    if usable.empty:
        return None

    layers = {}
    index = columns = None
    for column in SCORE_COLUMNS:
        wide = usable.pivot_table(index="date", columns="ticker",
                                  values=f"z_{column}").sort_index()
        layers[column] = wide
        index, columns = wide.index, wide.columns

    returns = (usable.pivot_table(index="date", columns="ticker",
                                  values="next_day_return")
               .sort_index().reindex(index=index, columns=columns))

    rebalance_on = _rebalance_dates(index, frequency)
    return {
        "layers": {k: v.to_numpy(dtype=float) for k, v in layers.items()},
        "returns": np.nan_to_num(returns.to_numpy(dtype=float), nan=0.0),
        "dates": index,
        "tickers": np.asarray(columns),
        "is_rebalance": np.fromiter((d in rebalance_on for d in index),
                                    dtype=bool, count=len(index)),
    }


def _score_matrix(prepared: dict, weights: dict[str, float]) -> np.ndarray:
    """Weighted sum of the pre-pivoted z-score layers."""
    total = None
    for column, weight in weights.items():
        layer = prepared["layers"][column]
        term = layer * weight
        total = term if total is None else total + term
    return total


def fit_weights(
    frame: pd.DataFrame,
    grid: list[dict[str, float]],
    top_n: int,
    frequency: str,
    cost_bps: float,
) -> tuple[dict[str, float], float]:
    """Pick the weight set with the best in-sample net Sharpe."""
    prepared = _factor_matrices(frame, frequency)
    if prepared is None:
        return dict(WEIGHTS), float("-inf")

    best_weights, best_sharpe = None, -np.inf
    for weights in grid:
        result = simulate(
            _score_matrix(prepared, weights),
            prepared["returns"],
            prepared["dates"],
            prepared["tickers"],
            prepared["is_rebalance"],
            top_n=top_n,
            cost_bps=cost_bps,
            with_holdings=False,
        )
        sharpe = performance(result["net_return"])["Sharpe Ratio"]
        if sharpe > best_sharpe:
            best_weights, best_sharpe = weights, sharpe
    return (best_weights or dict(WEIGHTS)), best_sharpe


def walk_forward(
    panel: pd.DataFrame,
    top_n: int = 3,
    frequency: str = "monthly",
    cost_bps: float = DEFAULT_COST_BPS,
    train_days: int = TRAIN_DAYS,
    test_days: int = TEST_DAYS,
    grid_step: float = 0.1,
) -> dict:
    """Run the validation. Returns per-window records and stitched series."""
    required = {"date", "ticker", "close", *SCORE_COLUMNS}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")

    frame = precompute_zscores(panel)
    frame = frame.sort_values(["ticker", "date"])
    frame["next_day_return"] = (
        frame.groupby("ticker")["close"].shift(-1) / frame["close"] - 1
    )

    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    windows = walk_forward_windows(dates, train_days, test_days)
    if not windows:
        raise ValueError(
            f"Not enough history: {len(dates)} dates, but a window needs "
            f"{train_days + test_days}. Download a longer range or reduce "
            f"train_days/test_days."
        )

    grid = _weight_grid(grid_step)
    records, fitted_parts, fixed_parts = [], [], []

    for index, (train, test) in enumerate(windows, start=1):
        train_frame = frame[frame["date"].isin(train)]
        test_frame = frame[frame["date"].isin(test)]

        weights, in_sample_sharpe = fit_weights(
            train_frame, grid, top_n, frequency, cost_bps
        )

        fitted_test = run_backtest(
            apply_weights(test_frame, weights),
            top_n=top_n, frequency=frequency, cost_bps=cost_bps,
        )
        fixed_test = run_backtest(
            apply_weights(test_frame, WEIGHTS),
            top_n=top_n, frequency=frequency, cost_bps=cost_bps,
        )
        if fitted_test.empty or fixed_test.empty:
            continue

        fitted_parts.append(fitted_test["net_return"])
        fixed_parts.append(fixed_test["net_return"])

        records.append({
            "window": index,
            "train_start": train.min().date(),
            "train_end": train.max().date(),
            "test_start": test.min().date(),
            "test_end": test.max().date(),
            "in_sample_sharpe": in_sample_sharpe,
            "oos_sharpe_fitted": performance(
                fitted_test["net_return"])["Sharpe Ratio"],
            "oos_sharpe_fixed": performance(
                fixed_test["net_return"])["Sharpe Ratio"],
            **{f"w_{k}": round(v, 3) for k, v in weights.items()},
        })

    fitted = pd.concat(fitted_parts).sort_index()
    fixed = pd.concat(fixed_parts).sort_index()

    equal = equal_weight_benchmark(frame)
    equal = equal.reindex(fitted.index).dropna()

    return {
        "windows": pd.DataFrame(records).set_index("window"),
        "oos_fitted": fitted,
        "oos_fixed": fixed,
        "oos_equal_weight": equal,
        "grid_size": len(grid),
    }


def summarise(result: dict) -> pd.DataFrame:
    """Out-of-sample performance of the three series, side by side."""
    return pd.DataFrame({
        "Fitted weights (OOS)": performance(result["oos_fitted"]),
        "Fixed weights (OOS)": performance(result["oos_fixed"]),
        "Equal weight (no signal)": performance(result["oos_equal_weight"]),
    }).T


def overfitting_gap(result: dict) -> pd.DataFrame:
    """In-sample minus out-of-sample Sharpe, per window and on average.

    A positive mean gap is the cost of fitting: performance that existed in
    the training data and did not survive into the future. It is expected to
    be positive. Its size is the point.
    """
    windows = result["windows"]
    table = windows[["in_sample_sharpe", "oos_sharpe_fitted",
                     "oos_sharpe_fixed"]].copy()
    table["gap"] = table["in_sample_sharpe"] - table["oos_sharpe_fitted"]
    table["fitted_minus_fixed"] = (
        table["oos_sharpe_fitted"] - table["oos_sharpe_fixed"]
    )
    return table

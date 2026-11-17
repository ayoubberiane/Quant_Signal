"""Cross-sectional standardisation and the QuantSignal composite score.

Shared by ranking/rank_assets.py and backtesting/backtest.py so that the
score has exactly one definition. It previously had two, written out
separately in each file, which could drift apart silently.

WHY STANDARDISATION IS NEEDED
    The score sums four terms that live on different scales:

        return_5d        order 0.01      (a 1% move is 0.01)
        trend_signal     0 or 1
        volume_ratio     order 1, unbounded above
        volatility_20d   order 0.2

    Summing them raw means each term's influence on the ranking comes from
    its units rather than from its weight. Measured across 7 assets, a
    nominal 30% weight on return_5d moved the ranking about 3%, while a
    nominal 20% weight on volume_ratio moved it about 55%: the ranking was
    effectively sorting on volume spikes.

    Standardising each term cross-sectionally puts them on a common scale,
    so the weights mean what they say.

CROSS-SECTIONAL, NOT TIME-SERIES
    Standardisation happens across assets on a single date. Using a z-score
    computed along an asset's own history would leak information across
    time and say nothing about how assets compare today.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

# Weights for the composite score. The volatility weight is negative, so
# higher volatility lowers the score; this replaces the old 1/(1+vol)
# transform, which was monotonic but compressed almost all its variation
# away (it contributed roughly 1.6% of the ranking spread rather than 20%).
WEIGHTS: dict[str, float] = {
    "return_5d": 0.30,
    "trend_signal": 0.30,
    "volume_ratio": 0.20,
    "volatility_20d": -0.20,
}

SCORE_COLUMNS: list[str] = list(WEIGHTS.keys())

# Clip to these quantiles before z-scoring. volume_ratio has no upper bound,
# so one spike would otherwise stretch the mean and standard deviation and
# flatten the differences between every other asset.
WINSOR_LIMITS: tuple[float, float] = (0.05, 0.95)


def winsorise(
    series: pd.Series, limits: tuple[float, float] = WINSOR_LIMITS
) -> pd.Series:
    """Clip to the given quantiles, leaving the ordering inside them intact."""
    valid = series.dropna()
    if len(valid) < 3:
        return series
    lower, upper = valid.quantile(limits[0]), valid.quantile(limits[1])
    return series.clip(lower=lower, upper=upper)


def zscore(series: pd.Series, winsorised: bool = True) -> pd.Series:
    """Cross-sectional z-score, safe on degenerate inputs.

    Returns zeros rather than NaN or inf when the cross-section carries no
    information. That matters for `trend_signal`, which is binary: in a
    broad uptrend every asset reads 1, the standard deviation is 0, and a
    naive z-score would be inf or NaN and poison the whole score.
    """
    data = winsorise(series) if winsorised else series
    valid = data.dropna()
    if len(valid) < 2:
        return pd.Series(0.0, index=series.index, dtype=float)
    sigma = valid.std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (data - valid.mean()) / sigma


def rank_score(series: pd.Series) -> pd.Series:
    """Percentile rank mapped to roughly [-1, 1].

    An alternative to z-scoring that ignores the shape of the distribution
    entirely. Useful as a robustness check: with a cross-section of 7 assets,
    winsorising at the 5th and 95th percentiles barely clips anything, so a
    z-score is still somewhat outlier-sensitive. If the two methods disagree
    about the ordering, that ordering rests on a few extreme observations.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (series.rank(pct=True, na_option="keep") - 0.5) * 2.0


_METHODS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "zscore": zscore,
    "rank": rank_score,
}


def add_quant_score(
    frame: pd.DataFrame,
    weights: dict[str, float] | None = None,
    method: str = "zscore",
    by: str | None = None,
) -> pd.DataFrame:
    """Add standardised `z_*` columns and `quant_score`.

    frame   one row per asset. With `by` set, one row per asset per date.
    by      None standardises over the whole frame, which is correct when
            the frame already holds a single date. Pass "date" for a panel
            spanning many dates, so that each date is standardised on its
            own and no information crosses between dates.

    The weights are not renormalised: only their relative sizes affect the
    ordering, so their absolute scale carries no meaning.
    """
    weights = weights or WEIGHTS
    if method not in _METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {sorted(_METHODS)}"
        )
    transform = _METHODS[method]

    missing = [column for column in weights if column not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing columns required for quant score: {missing}"
        )

    out = frame.copy()
    for column in weights:
        if by is None:
            out[f"z_{column}"] = transform(out[column])
        else:
            out[f"z_{column}"] = out.groupby(by)[column].transform(transform)

    out["quant_score"] = sum(
        out[f"z_{column}"] * weight for column, weight in weights.items()
    )
    return out


def effective_weights(
    frame: pd.DataFrame, weights: dict[str, float] | None = None
) -> pd.DataFrame:
    """Each term's share of the total cross-sectional spread of the score.

    A diagnostic for the bug this module exists to fix. Once the terms are
    standardised, the effective share of each should sit close to its
    weight's share of the total absolute weight. A large gap means some term
    is still dominating on scale rather than on intent.
    """
    weights = weights or WEIGHTS
    scored = add_quant_score(frame, weights)

    spreads, intended = {}, {}
    total_weight = sum(abs(w) for w in weights.values())
    for column, weight in weights.items():
        contribution = scored[f"z_{column}"] * weight
        spreads[column] = contribution.max() - contribution.min()
        intended[column] = abs(weight) / total_weight

    total_spread = sum(spreads.values())
    return pd.DataFrame({
        "intended": pd.Series(intended),
        "effective": pd.Series({
            column: (spread / total_spread if total_spread else 0.0)
            for column, spread in spreads.items()
        }),
    })

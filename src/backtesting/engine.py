"""Portfolio simulation with transaction costs and benchmarks.

WHY THIS MODULE EXISTS
    The first backtest held one asset, rebalanced every day, at zero cost.
    That is close to the most cost-sensitive configuration possible: daily
    turnover on a single position. A result from it says almost nothing
    about whether a strategy works, because the costs it omits are the
    largest term in the calculation.

TIMING
    Weights are formed from the score on date t and earn the return from t
    to t+1. Nothing on date t uses information after t. `no_lookahead` in
    the tests pins this.

WEIGHT DRIFT
    Between rebalance dates the portfolio is held, so weights drift with
    the assets' returns rather than being silently reset. Resetting them
    every day would understate turnover and so understate cost.

TURNOVER AND COST
    Turnover is two-way: sum of |target weight - drifted weight| across
    assets. A full switch from one asset to another is therefore a turnover
    of 2.0, and costs `2.0 * cost_bps / 10_000`. Reported turnover figures
    use the same convention, so a strategy's cost is always
    `turnover * cost_bps / 10_000` with no hidden factor of two.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Rebalance schedules, as pandas period aliases on the date index.
REBALANCE_RULES = {
    "daily": None,
    "weekly": "W",
    "monthly": "M",
}

# Round-trip cost assumptions, in basis points of traded notional. Liquid US
# ETFs sit near the low end; crypto and thin instruments are far higher.
DEFAULT_COST_BPS = 5.0
COST_SWEEP_BPS = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0)

TRADING_DAYS = 252


def _rebalance_dates(dates: pd.DatetimeIndex, frequency: str) -> set:
    """The dates on which target weights are reset."""
    if frequency not in REBALANCE_RULES:
        raise ValueError(
            f"unknown frequency {frequency!r}; "
            f"expected one of {sorted(REBALANCE_RULES)}"
        )
    rule = REBALANCE_RULES[frequency]
    if rule is None:
        return set(dates)
    # Last available trading date within each period.
    series = pd.Series(dates, index=dates)
    return set(series.groupby(dates.to_period(rule)).max())


def target_weights(scores: pd.Series, top_n: int) -> pd.Series:
    """Equal weight across the top `top_n` assets by score.

    Equal weight rather than score-proportional: the score is an ordering,
    not a magnitude, so treating a score of 1.4 as 1.4x the conviction of
    1.0 would read meaning into it that it does not carry.
    """
    ranked = scores.dropna().sort_values(ascending=False)
    chosen = ranked.head(top_n)
    if chosen.empty:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(chosen), index=chosen.index)


def run_backtest(
    panel: pd.DataFrame,
    top_n: int = 3,
    frequency: str = "monthly",
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Simulate the strategy. Returns one row per date.

    panel must hold columns: date, ticker, quant_score, next_day_return.
    `next_day_return` on date t is the return from t to t+1, so a weight
    set on t is paid that return.
    """
    required = {"date", "ticker", "quant_score", "next_day_return"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing columns: {sorted(missing)}")

    frame = panel.dropna(subset=["quant_score", "next_day_return"]).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.empty:
        return pd.DataFrame()

    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    rebalance_on = _rebalance_dates(dates, frequency)

    scores = frame.pivot_table(index="date", columns="ticker",
                               values="quant_score")
    returns = frame.pivot_table(index="date", columns="ticker",
                                values="next_day_return")

    weights = pd.Series(0.0, index=scores.columns)
    rows = []

    for date in dates:
        turnover = 0.0
        rebalanced = date in rebalance_on

        if rebalanced:
            target = target_weights(scores.loc[date], top_n)
            target = target.reindex(scores.columns).fillna(0.0)
            turnover = float((target - weights).abs().sum())
            weights = target

        period_returns = returns.loc[date].fillna(0.0)
        gross = float((weights * period_returns).sum())
        cost = turnover * cost_bps / 10_000.0

        rows.append({
            "date": date,
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
            "n_holdings": int((weights.abs() > 1e-12).sum()),
            "rebalanced": rebalanced,
            "holdings": ", ".join(
                sorted(weights[weights.abs() > 1e-12].index)
            ),
        })

        # Hold to the next date: weights drift with realised returns.
        grown = weights * (1.0 + period_returns)
        total = grown.sum()
        weights = grown / total if total > 0 else grown

    return pd.DataFrame(rows).set_index("date")


def buy_and_hold(panel: pd.DataFrame, ticker: str) -> pd.Series:
    """Single-asset benchmark, on the same dates as the strategy."""
    series = panel[panel["ticker"] == ticker]
    if series.empty:
        return pd.Series(dtype=float)
    return (series.set_index("date")["next_day_return"]
            .sort_index().rename(f"{ticker} buy & hold"))


def equal_weight_benchmark(panel: pd.DataFrame) -> pd.Series:
    """Equal weight across every asset, rebalanced daily, costs ignored.

    The benchmark a signal has to beat to have shown anything: it uses no
    signal at all. Costs are omitted deliberately, which makes it a
    slightly generous comparison rather than a flattering one.
    """
    wide = panel.pivot_table(index="date", columns="ticker",
                             values="next_day_return")
    return wide.mean(axis=1).rename("Equal weight")


def performance(returns: pd.Series, periods: int = TRADING_DAYS) -> dict:
    """Standard performance statistics for a return series."""
    series = pd.Series(returns).dropna()
    if series.empty:
        return {key: 0.0 for key in (
            "Total Return", "Annualised Return", "Annualised Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Maximum Drawdown",
            "Calmar Ratio", "Win Rate", "Observations",
        )}

    curve = (1.0 + series).cumprod()
    total = float(curve.iloc[-1] - 1.0)

    years = len(series) / periods
    annualised = ((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else 0.0

    volatility = float(series.std(ddof=1) * np.sqrt(periods))
    sharpe = annualised / volatility if volatility > 0 else 0.0

    # Sortino uses downside deviation only, so upside volatility is not
    # treated as risk.
    downside = series[series < 0]
    downside_dev = float(downside.std(ddof=1) * np.sqrt(periods)) \
        if len(downside) > 1 else 0.0
    sortino = annualised / downside_dev if downside_dev > 0 else 0.0

    drawdown = float((curve / curve.cummax() - 1.0).min())
    calmar = annualised / abs(drawdown) if drawdown < 0 else 0.0

    return {
        "Total Return": total,
        "Annualised Return": annualised,
        "Annualised Volatility": volatility,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Maximum Drawdown": drawdown,
        "Calmar Ratio": calmar,
        "Win Rate": float((series > 0).mean()),
        "Observations": int(len(series)),
    }


def cost_sensitivity(
    panel: pd.DataFrame,
    top_n: int = 3,
    frequency: str = "monthly",
    sweep: tuple[float, ...] = COST_SWEEP_BPS,
) -> pd.DataFrame:
    """Net performance across a range of cost assumptions.

    The most informative single table in a backtest. A strategy whose Sharpe
    collapses between 0 and 10bp is a cost artefact, not a signal, and the
    break-even cost says how much slippage the edge can absorb before it
    disappears.
    """
    rows = {}
    for bps in sweep:
        result = run_backtest(panel, top_n=top_n, frequency=frequency,
                              cost_bps=bps)
        if result.empty:
            continue
        stats = performance(result["net_return"])
        stats["Annual Turnover"] = float(
            result["turnover"].sum() / (len(result) / TRADING_DAYS)
        )
        rows[f"{bps:g} bp"] = stats
    return pd.DataFrame(rows).T


def compare_frequencies(
    panel: pd.DataFrame,
    top_n: int = 3,
    cost_bps: float = DEFAULT_COST_BPS,
) -> pd.DataFrame:
    """Net performance by rebalance frequency at a fixed cost.

    Trading more often acts on fresher signals but pays more in costs. This
    shows which side wins, rather than assuming either.
    """
    rows = {}
    for frequency in REBALANCE_RULES:
        result = run_backtest(panel, top_n=top_n, frequency=frequency,
                              cost_bps=cost_bps)
        if result.empty:
            continue
        stats = performance(result["net_return"])
        stats["Annual Turnover"] = float(
            result["turnover"].sum() / (len(result) / TRADING_DAYS)
        )
        stats["Cost Drag"] = float(result["cost"].sum())
        rows[frequency] = stats
    return pd.DataFrame(rows).T

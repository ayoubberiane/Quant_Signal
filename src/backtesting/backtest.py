"""QuantSignal backtest.

    python src/backtesting/backtest.py
    python src/backtesting/backtest.py --top-n 2 --frequency weekly --cost-bps 10

Reports net-of-cost performance against benchmarks, plus how the result
behaves as the cost assumption changes. A gross result on its own is not
reportable: the costs it leaves out are usually the largest term.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoring import WEIGHTS, add_quant_score  # noqa: E402
from backtesting.engine import (  # noqa: E402
    DEFAULT_COST_BPS,
    buy_and_hold,
    compare_frequencies,
    cost_sensitivity,
    equal_weight_benchmark,
    performance,
    run_backtest,
)

# Benchmark asset, if present in the data.
BENCHMARK_TICKER = "SPY"


def load_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load every *_features.csv found.

    Discovered rather than hardcoded: the previous fixed list held five
    equity tickers, so TLT and GLD were silently dropped and the backtest
    ran equity-only while the ranking covered everything.
    """
    paths = sorted(data_dir.glob("*_features.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No *_features.csv found in {data_dir}. "
            "Run the ingestion, cleaning and feature steps first."
        )

    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["ticker"] = path.stem.replace("_features", "")
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(panel)} observations across "
          f"{panel['ticker'].nunique()} assets")
    print(f"Assets: {sorted(panel['ticker'].unique())}")
    print(f"Dates:  {panel['date'].min().date()} to "
          f"{panel['date'].max().date()}")
    return panel


def calculate_quant_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score every observation, standardised within its own date.

    Standardising per date keeps each day's cross-section self-contained.
    Pooling the panel would let a later date's distribution inform an
    earlier date's score, which is look-ahead.

    The definition lives in src/scoring.py and is shared with
    ranking/rank_assets.py so the two cannot drift apart.
    """
    df = df.copy()

    missing = [c for c in WEIGHTS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns required for quant score: {missing}"
        )
    if "date" not in df.columns:
        raise ValueError(
            "calculate_quant_score needs a 'date' column: the score is "
            "standardised within each date's cross-section."
        )

    df = df.dropna(subset=["return_5d", "volatility_20d", "volume_ratio"])
    return add_quant_score(df, weights=WEIGHTS, by="date")


def calculate_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Next-day return per asset: the return from t to t+1.

    Grouped by ticker, so the last row of one asset never borrows the next
    asset's price.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["next_day_return"] = (
        df.groupby("ticker")["close"].shift(-1) / df["close"] - 1
    )
    return df


def _show(title: str, table: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    formatters = {}
    for column in table.columns:
        if "Ratio" in column or column == "Observations":
            formatters[column] = lambda x: f"{x:.3f}"
        elif "Turnover" in column:
            formatters[column] = lambda x: f"{x:.2f}x"
        else:
            formatters[column] = lambda x: f"{x:.2%}"
    print(table.to_string(formatters=formatters))


def main() -> int:
    parser = argparse.ArgumentParser(description="QuantSignal backtest")
    parser.add_argument("--top-n", type=int, default=3,
                        help="number of assets held (default 3)")
    parser.add_argument("--frequency", default="monthly",
                        choices=["daily", "weekly", "monthly"],
                        help="rebalance frequency (default monthly)")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS,
                        help=f"round-trip cost in basis points "
                             f"(default {DEFAULT_COST_BPS:g})")
    args = parser.parse_args()

    panel = load_features()

    print("\nScoring and computing forward returns...")
    panel = calculate_quant_score(panel)
    panel = calculate_forward_returns(panel)

    print(f"\nRunning backtest: top {args.top_n}, {args.frequency} "
          f"rebalance, {args.cost_bps:g}bp costs")
    result = run_backtest(panel, top_n=args.top_n,
                          frequency=args.frequency,
                          cost_bps=args.cost_bps)
    if result.empty:
        print("No scoreable dates. Check that the feature files are populated.")
        return 1

    # Strategy against the benchmarks it has to beat.
    rows = {
        "Strategy (net)": performance(result["net_return"]),
        "Strategy (gross)": performance(result["gross_return"]),
        "Equal weight": performance(equal_weight_benchmark(panel)),
    }
    bench = buy_and_hold(panel, BENCHMARK_TICKER)
    if not bench.empty:
        rows[f"{BENCHMARK_TICKER} buy & hold"] = performance(bench)

    _show("PERFORMANCE", pd.DataFrame(rows).T)

    gross = performance(result["gross_return"])["Annualised Return"]
    net = performance(result["net_return"])["Annualised Return"]
    print(f"\nCost drag: {gross - net:.2%} of annualised return "
          f"({result['cost'].sum():.2%} cumulative)")
    print(f"Annual turnover: "
          f"{result['turnover'].sum() / (len(result) / 252):.2f}x")

    _show("COST SENSITIVITY (net)", cost_sensitivity(
        panel, top_n=args.top_n, frequency=args.frequency))
    print("\nA Sharpe that collapses between 0 and 10bp is a cost artefact,")
    print("not a signal. The break-even cost is how much slippage the edge")
    print("can absorb before it disappears.")

    _show(f"REBALANCE FREQUENCY (at {args.cost_bps:g}bp)",
          compare_frequencies(panel, top_n=args.top_n,
                              cost_bps=args.cost_bps))

    output = DATA_DIR / "backtest_results.csv"
    result.to_csv(output)
    print(f"\nSaved per-date results to: {output}")

    print("\nNOTE: a backtest describes the past on one fixed universe of")
    print("currently-listed assets. Weights were not fitted out-of-sample")
    print("here, so these figures are a description, not evidence of an edge,")
    print("and not investment advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

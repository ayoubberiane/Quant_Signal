"""Walk-forward validation entry point.

    python src/validation/validate.py
    python src/validation/validate.py --top-n 1 --frequency weekly --cost-bps 10
    python src/validation/validate.py --train-days 378 --test-days 63
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtesting.engine import DEFAULT_COST_BPS  # noqa: E402
from validation.walkforward import (  # noqa: E402
    TEST_DAYS,
    TRAIN_DAYS,
    overfitting_gap,
    summarise,
    walk_forward,
)


def load_panel(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    paths = sorted(data_dir.glob("*_features.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No *_features.csv in {data_dir}. Run the ingestion, cleaning "
            "and feature steps first."
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["ticker"] = path.stem.replace("_features", "")
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(panel)} observations, "
          f"{panel['ticker'].nunique()} assets, "
          f"{panel['date'].min().date()} to {panel['date'].max().date()}")
    return panel


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward validation of the score weights"
    )
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--frequency", default="monthly",
                        choices=["daily", "weekly", "monthly"])
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--train-days", type=int, default=TRAIN_DAYS)
    parser.add_argument("--test-days", type=int, default=TEST_DAYS)
    parser.add_argument("--grid-step", type=float, default=0.1)
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    panel = load_panel()

    print(f"\nWalk-forward: {args.train_days}d train / {args.test_days}d "
          f"test, top {args.top_n}, {args.frequency}, {args.cost_bps:g}bp")
    print("Fitting weights on each training window only, then applying them")
    print("unchanged to the test window that follows. This may take a minute.")

    result = walk_forward(
        panel,
        top_n=args.top_n,
        frequency=args.frequency,
        cost_bps=args.cost_bps,
        train_days=args.train_days,
        test_days=args.test_days,
        grid_step=args.grid_step,
    )

    windows = result["windows"]
    print(f"\n{len(windows)} windows, {result['grid_size']} candidate "
          f"weight sets per window")

    print("\n=== OUT-OF-SAMPLE PERFORMANCE ===")
    print("Every figure below is on data no fitted weight ever saw.\n")
    summary = summarise(result)
    print(summary.to_string(formatters={
        c: (lambda x: f"{x:.3f}") if ("Ratio" in c or c == "Observations")
        else (lambda x: f"{x:.2%}")
        for c in summary.columns
    }))

    print("\n=== PER-WINDOW: IN-SAMPLE vs OUT-OF-SAMPLE ===")
    gap = overfitting_gap(result)
    print(gap.to_string(float_format=lambda x: f"{x: .3f}"))

    mean_gap = gap["gap"].mean()
    mean_edge = gap["fitted_minus_fixed"].mean()

    print("\n=== READING THIS ===")
    print(f"Mean in-sample minus out-of-sample Sharpe: {mean_gap:+.3f}")
    print("  This is the cost of fitting: performance that existed in the")
    print("  training data and did not survive into the future. A positive")
    print("  number is expected. Its size is the point.")
    print(f"\nMean out-of-sample Sharpe, fitted minus fixed: {mean_edge:+.3f}")
    if mean_edge <= 0:
        print("  Fitting did NOT beat the hand-chosen weights out of sample.")
        print("  On this evidence, searching the weights is adding noise")
        print("  rather than signal, and the fixed weights are preferable")
        print("  because they have no parameters fitted to this history.")
    else:
        print("  Fitting beat the fixed weights out of sample. Worth")
        print("  checking whether it holds on a different train/test split")
        print("  before relying on it: one favourable split is not evidence.")

    fitted_sharpe = summary.loc["Fitted weights (OOS)", "Sharpe Ratio"]
    fixed_sharpe = summary.loc["Fixed weights (OOS)", "Sharpe Ratio"]
    equal_sharpe = summary.loc["Equal weight (no signal)", "Sharpe Ratio"]
    best = max(fitted_sharpe, fixed_sharpe)
    print(f"\nBest strategy Sharpe {best:.3f} vs equal weight "
          f"{equal_sharpe:.3f}")
    if best <= equal_sharpe:
        print("  The signal did not beat equal weighting, which uses no")
        print("  signal at all. On this sample it has not demonstrated an")
        print("  edge. That is a result worth recording, not a bug.")
    else:
        print("  The signal beat equal weighting out of sample. Check the")
        print("  per-window table: a single strong window can carry an")
        print("  average, and consistency matters more than the mean.")

    windows_out = DATA_DIR / "walkforward_windows.csv"
    oos_out = DATA_DIR / "walkforward_oos_returns.csv"
    windows.to_csv(windows_out)
    pd.DataFrame({
        "fitted": result["oos_fitted"],
        "fixed": result["oos_fixed"],
        "equal_weight": result["oos_equal_weight"],
    }).to_csv(oos_out)
    print(f"\nSaved {windows_out.name} and {oos_out.name} to {DATA_DIR}")

    print("\nNOTE: walk-forward validation reduces overfitting; it does not")
    print("eliminate it. The universe is still fixed and currently-listed,")
    print("and the factor set itself was chosen with hindsight. Not")
    print("investment advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

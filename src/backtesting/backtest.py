from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoring import WEIGHTS, add_quant_score  # noqa: E402

FEATURE_FILES = [
    "EEM_features.csv",
    "EFA_features.csv",
    "IWM_features.csv",
    "QQQ_features.csv",
    "SPY_features.csv",
]


def load_features():
    """Load all asset feature datasets."""

    frames = []

    for filename in FEATURE_FILES:
        path = DATA_DIR / filename

        if not path.exists():
            print(f"WARNING: {filename} not found")
            continue

        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])

        ticker = filename.replace("_features.csv", "")
        df["ticker"] = ticker

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No feature files were found."
        )

    return pd.concat(frames, ignore_index=True)


def calculate_quant_score(df):
    """Calculate QuantSignal score for every historical observation.

    Standardisation runs within each date (`by="date"`), so each day's
    cross-section is scored against itself. Standardising over the whole
    panel at once would mix dates together and let a later date's
    distribution inform an earlier date's score, which is look-ahead.

    The score definition lives in src/scoring.py and is shared with
    ranking/rank_assets.py, so the two cannot drift apart.
    """

    df = df.copy()

    required_columns = list(WEIGHTS.keys())

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns required for quant score: {missing}"
        )

    if "date" not in df.columns:
        raise ValueError(
            "calculate_quant_score needs a 'date' column: the score is "
            "standardised within each date's cross-section."
        )

    # Rows still in their indicator warm-up would otherwise enter the
    # cross-section and shift every other asset's z-score that day.
    df = df.dropna(subset=["return_5d", "volatility_20d", "volume_ratio"])

    return add_quant_score(df, weights=WEIGHTS, by="date")


def calculate_forward_returns(df):
    """Calculate next-day returns for each asset."""

    df = df.sort_values(
        ["ticker", "date"]
    ).copy()

    df["next_day_return"] = (
        df.groupby("ticker")["close"].shift(-1)
        / df["close"]
        - 1
    )

    return df


def calculate_portfolio_returns(df):
    """
    Simulate a simple strategy.

    At each date:
    - Rank assets using QuantSignal score.
    - Buy the top-ranked asset.
    - Hold it for the next trading day.
    """

    df = df.dropna(
        subset=[
            "next_day_return",
            "quant_score",
        ]
    ).copy()

    rankings = (
        df.sort_values(
            ["date", "quant_score"],
            ascending=[True, False],
        )
        .groupby("date")
        .head(1)
        .copy()
    )

    rankings["strategy_return"] = (
        rankings["next_day_return"]
    )

    return rankings


def calculate_metrics(strategy_returns):
    """Calculate basic backtest performance metrics."""

    returns = strategy_returns["strategy_return"]

    cumulative_return = (
        (1 + returns).prod() - 1
    )

    annualized_return = (
        (1 + cumulative_return)
        ** (252 / len(returns))
        - 1
        if len(returns) > 0
        else 0
    )

    volatility = (
        returns.std() * (252 ** 0.5)
    )

    sharpe_ratio = (
        annualized_return / volatility
        if volatility != 0
        else 0
    )

    cumulative_curve = (
        1 + returns
    ).cumprod()

    running_max = cumulative_curve.cummax()

    drawdown = (
        cumulative_curve / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    win_rate = (
        returns > 0
    ).mean()

    return {
        "Total Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
        "Win Rate": win_rate,
    }


def main():

    print("Loading feature datasets...")

    df = load_features()

    print(
        f"Loaded {len(df)} total observations"
    )

    print(
        f"Assets: {sorted(df['ticker'].unique())}"
    )

    print("\nCalculating QuantSignal scores...")

    df = calculate_quant_score(df)

    print("\nCalculating forward returns...")

    df = calculate_forward_returns(df)

    print("\nRunning QuantSignal backtest...")

    strategy = calculate_portfolio_returns(df)

    metrics = calculate_metrics(strategy)

    output_path = (
        DATA_DIR / "backtest_results.csv"
    )

    strategy.to_csv(
        output_path,
        index=False
    )

    print("\nBacktest Results")
    print("----------------")

    for name, value in metrics.items():

        if "Ratio" in name:
            print(
                f"{name}: {value:.3f}"
            )
        else:
            print(
                f"{name}: {value:.2%}"
            )

    print(
        f"\nSaved backtest data to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
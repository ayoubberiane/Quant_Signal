from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "SPY_features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "SPY_ranked.csv"


def calculate_score(df):
    """
    Calculate a quantitative score using momentum,
    trend, volatility, and volume.
    """

    # Momentum: stronger recent returns are better
    momentum_score = df["return_5d"].rank(pct=True)

    # Trend: price above its moving average is positive
    trend_score = df["trend_signal"]

    # Volume: unusually high volume receives a positive score
    volume_score = df["volume_ratio"].rank(pct=True)

    # Lower volatility is preferred
    volatility_score = 1 - df["volatility_20d"].rank(pct=True)

    # Combine the signals
    df["quant_score"] = (
        0.35 * momentum_score
        + 0.30 * trend_score
        + 0.20 * volume_score
        + 0.15 * volatility_score
    )

    return df


def rank_assets():
    """Generate the QuantSignal score."""

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    # Remove rows where indicators are not yet available
    df = df.dropna(
        subset=[
            "return_5d",
            "ma_20",
            "ma_50",
            "volatility_20d",
            "volume_ratio",
        ]
    ).copy()

    # Calculate score
    df = calculate_score(df)

    # Rank observations from strongest to weakest
    df["rank"] = (
        df.groupby("date")["quant_score"]
        .rank(ascending=False, method="first")
    )

    # Sort by score
    df = df.sort_values(
        ["date", "quant_score"],
        ascending=[True, False]
    )

    # Save
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated QuantSignal scores for {len(df)} rows")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nLatest signals:")

    latest_date = df["date"].max()

    latest = df[df["date"] == latest_date][
        [
            "date",
            "close",
            "return_5d",
            "volatility_20d",
            "volume_ratio",
            "trend_signal",
            "momentum_signal",
            "quant_score",
            "rank",
        ]
    ]

    print(latest.to_string(index=False))


if __name__ == "__main__":
    rank_assets()

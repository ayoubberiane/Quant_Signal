from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def generate_features(df):
    """Generate quantitative features for one asset."""

    df = df.copy()

    # Daily return
    df["daily_return"] = df["close"].pct_change()

    # 5-day momentum
    df["return_5d"] = df["close"].pct_change(5)

    # Moving averages
    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_50"] = df["close"].rolling(50).mean()

    # 20-day volatility
    df["volatility_20d"] = (
        df["daily_return"].rolling(20).std() * (252 ** 0.5)
    )

    # Volume ratio
    df["volume_ratio"] = (
        df["volume"] / df["volume"].rolling(20).mean()
    )

    # Trend signal
    df["trend_signal"] = (
        df["ma_20"] > df["ma_50"]
    ).astype(int)

    # Momentum signal
    df["momentum_signal"] = (
        df["return_5d"] > 0
    ).astype(int)

    return df


def process_asset(input_file):
    """Generate features for one cleaned asset."""

    ticker = input_file.stem.replace("_clean", "")

    df = pd.read_csv(input_file)

    df = generate_features(df)

    output_file = PROCESSED_DIR / f"{ticker}_features.csv"

    df.to_csv(output_file, index=False)

    print(
        f"{ticker}: generated features for "
        f"{len(df)} rows → {output_file.name}"
    )


def main():
    """Generate features for all cleaned assets."""

    files = sorted(PROCESSED_DIR.glob("*_clean.csv"))

    if not files:
        raise FileNotFoundError(
            f"No cleaned datasets found in {PROCESSED_DIR}"
        )

    for file in files:
        process_asset(file)


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "SPY_clean.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "SPY_features.csv"


def calculate_features():
    """Calculate quantitative features from cleaned market data."""

    # Load cleaned data
    df = pd.read_csv(INPUT_FILE)

    # Make sure the data is chronological
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --------------------------------------------------
    # 1. Daily return
    # --------------------------------------------------
    df["daily_return"] = df["close"].pct_change()

    # --------------------------------------------------
    # 2. 5-day momentum
    # --------------------------------------------------
    df["return_5d"] = df["close"].pct_change(periods=5)

    # --------------------------------------------------
    # 3. Moving averages
    # --------------------------------------------------
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ma_50"] = df["close"].rolling(window=50).mean()

    # --------------------------------------------------
    # 4. 20-day annualized volatility
    # --------------------------------------------------
    df["volatility_20d"] = (
        df["daily_return"]
        .rolling(window=20)
        .std()
        * np.sqrt(252)
    )

    # --------------------------------------------------
    # 5. Volume ratio
    # --------------------------------------------------
    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()

    df["volume_ratio"] = (
        df["volume"] / df["volume_ma_20"]
    )

    # --------------------------------------------------
    # 6. Trend signal
    # --------------------------------------------------
    df["trend_signal"] = (
        df["ma_20"] > df["ma_50"]
    ).astype(int)

    # --------------------------------------------------
    # 7. Momentum signal
    # --------------------------------------------------
    df["momentum_signal"] = (
        df["return_5d"] > 0
    ).astype(int)

    # Save features
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Generated features for {len(df)} rows")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nFeature columns:")
    print([
        "daily_return",
        "return_5d",
        "ma_20",
        "ma_50",
        "volatility_20d",
        "volume_ratio",
        "trend_signal",
        "momentum_signal",
    ])


if __name__ == "__main__":
    calculate_features()

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def calculate_score(df):
    """Calculate the QuantSignal score for each asset."""

    df = df.copy()

    # Remove rows where rolling indicators are not available
    df = df.dropna(
        subset=[
            "return_5d",
            "volatility_20d",
            "volume_ratio",
        ]
    )

    # QuantSignal score
    #
    # Momentum:        30%
    # Trend:           30%
    # Volume:          20%
    # Low volatility: 20%
    df["quant_score"] = (
        0.30 * df["return_5d"]
        + 0.30 * df["trend_signal"]
        + 0.20 * df["volume_ratio"]
        + 0.20 * (1 / (1 + df["volatility_20d"]))
    )

    return df


def main():
    """Rank all assets on the latest available date."""

    files = sorted(PROCESSED_DIR.glob("*_features.csv"))

    if not files:
        raise FileNotFoundError(
            f"No feature files found in {PROCESSED_DIR}"
        )

    all_assets = []

    for file in files:
        df = pd.read_csv(file)

        ticker = file.stem.replace("_features", "")
        df["ticker"] = ticker

        df = calculate_score(df)

        # Keep only the latest observation for cross-asset ranking
        latest = df.iloc[-1].copy()

        all_assets.append(latest)

    ranking = pd.DataFrame(all_assets)

    # Rank assets from highest score to lowest score
    ranking["rank"] = (
        ranking["quant_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    ranking = ranking.sort_values("rank")

    output_file = PROCESSED_DIR / "asset_ranking.csv"
    ranking.to_csv(output_file, index=False)

    print(f"Ranked {len(ranking)} assets")
    print(f"Saved to: {output_file}")

    print("\nLatest QuantSignal ranking:")
    print(
        ranking[
            [
                "ticker",
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
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

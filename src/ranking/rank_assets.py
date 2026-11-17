from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scoring import (  # noqa: E402
    SCORE_COLUMNS,
    WEIGHTS,
    add_quant_score,
    effective_weights,
)

# Indicator columns must be available before an asset can be scored.
REQUIRED_FACTORS = [
    "return_5d",
    "volatility_20d",
    "volume_ratio",
]


def latest_observation(input_file):
    """Return the most recent scoreable row for one asset.

    Rolling indicators are NaN during their warm-up, and `trend_signal`
    reads 0 there rather than missing, so those rows are dropped before the
    latest is taken. Otherwise an asset with a short history would be ranked
    on a row whose trend signal only looks like a downtrend.
    """
    df = pd.read_csv(input_file, parse_dates=["date"])

    ticker = input_file.stem.replace("_features", "")
    df["ticker"] = ticker

    df = df.sort_values("date")
    df = df.dropna(subset=REQUIRED_FACTORS)
    df = df[df["ma_50"].notna()] if "ma_50" in df.columns else df

    if df.empty:
        print(f"WARNING: {ticker} has no rows with complete indicators")
        return None

    return df.iloc[-1].copy()


def calculate_score(df, method="zscore"):
    """Score a cross-section of assets: one row per asset, on one date.

    This takes a cross-section, not a single asset's time series. Each term
    is standardised across the assets present before the weights are
    applied, because the four terms are on different scales and summing
    them raw lets units decide the ranking instead of the weights. See
    src/scoring.py for the measured effect.
    """
    df = df.copy()
    df = df.dropna(subset=REQUIRED_FACTORS)
    return add_quant_score(df, weights=WEIGHTS, method=method)


def main():
    """Rank all assets on the latest available date."""

    files = sorted(PROCESSED_DIR.glob("*_features.csv"))

    if not files:
        raise FileNotFoundError(
            f"No feature files found in {PROCESSED_DIR}"
        )

    # Build the cross-section first. Standardisation compares assets against
    # each other, so every asset has to be in hand before anything is scored.
    rows = [row for row in (latest_observation(f) for f in files)
            if row is not None]

    if not rows:
        raise ValueError("No asset had a complete set of indicators")

    cross_section = pd.DataFrame(rows).set_index("ticker")

    dates = pd.to_datetime(cross_section["date"]).dt.date
    if dates.nunique() > 1:
        print(
            "WARNING: assets' latest dates differ "
            f"({dates.min()} to {dates.max()}). The cross-section mixes "
            "dates, so the comparison is not strictly as-of one date."
        )

    ranking = calculate_score(cross_section)

    ranking["rank"] = (
        ranking["quant_score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    ranking = ranking.sort_values("rank")

    output_file = PROCESSED_DIR / "asset_ranking.csv"
    ranking.to_csv(output_file)

    print(f"Ranked {len(ranking)} assets")
    print(f"Saved to: {output_file}")

    print("\nLatest QuantSignal ranking:")
    display = [
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
    print(ranking[[c for c in display if c in ranking.columns]]
          .to_string())

    print("\nStandardised inputs (z-scores across the assets above):")
    print(ranking[[f"z_{c}" for c in SCORE_COLUMNS]]
          .to_string(float_format=lambda x: f"{x: .3f}"))

    print("\nEach term's share of the ranking spread:")
    print("  'intended' is the term's share of total absolute weight.")
    print("  A large gap means a term is dominating on scale, not intent.\n")
    print(effective_weights(cross_section)
          .to_string(float_format=lambda x: f"{x:.1%}"))

    print(
        "\nNOTE: this ranks the assets against each other on one date. "
        "It is not\na return forecast and not investment advice."
    )


if __name__ == "__main__":
    main()

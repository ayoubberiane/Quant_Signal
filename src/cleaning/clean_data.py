from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


REQUIRED_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "ticker",
]


def clean_file(input_file):
    """Clean one asset's market data."""

    df = pd.read_csv(input_file)

    print(f"\nProcessing {input_file.name}")
    print(f"Loaded {len(df)} rows")

    # Check required columns
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{input_file.name} is missing columns: "
            f"{missing_columns}"
        )

    # Convert date
    df["date"] = pd.to_datetime(df["date"])

    # Remove duplicate observations
    df = df.drop_duplicates(
        subset=["date", "ticker"]
    )

    # Sort chronologically
    df = df.sort_values(
        ["ticker", "date"]
    )

    # Remove rows with missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    df = df.dropna(
        subset=[
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
        ]
    )

    # Basic price validation
    df = df[
        (df["open"] > 0)
        & (df["high"] > 0)
        & (df["low"] > 0)
        & (df["close"] > 0)
        & (df["volume"] >= 0)
    ]

    print(f"Cleaned dataset contains {len(df)} rows")

    return df


def main():
    """Clean all raw market-data files."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_files = sorted(
        RAW_DIR.glob("*.csv")
    )

    # Ignore .gitkeep because it is not a CSV
    csv_files = [
        file
        for file in csv_files
        if file.name != ".gitkeep"
    ]

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw/"
        )

    for input_file in csv_files:

        df = clean_file(input_file)

        ticker = df["ticker"].iloc[0]

        output_file = (
            PROCESSED_DIR
            / f"{ticker}_clean.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print(
            f"Saved cleaned data to: "
            f"{output_file}"
        )


if __name__ == "__main__":
    main()
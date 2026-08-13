import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load API key from .env
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("POLYGON_API_KEY")

START_DATE = "2025-01-01"
END_DATE = "2025-12-31"

# Assets to analyze
TICKERS = [
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "GLD",
]


def download_market_data(ticker):
    """Download daily market data for one asset."""

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/"
        f"{ticker}/range/1/day/{START_DATE}/{END_DATE}"
    )

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)

    if response.status_code != 200:
        raise Exception(
            f"API request failed for {ticker}: "
            f"{response.status_code}\n{response.text}"
        )

    data = response.json()

    if "results" not in data:
        raise Exception(
            f"No results returned for {ticker}: {data}"
        )

    df = pd.DataFrame(data["results"])

    # Convert Polygon timestamp
    df["date"] = pd.to_datetime(df["t"], unit="ms")

    # Rename Polygon columns
    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
        }
    )

    # Keep required columns
    df = df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
        ]
    ]

    # Add ticker
    df["ticker"] = ticker

    # Save the data
    output_file = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / f"{ticker}.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(output_file, index=False)

    print(
        f"{ticker}: downloaded {len(df)} rows → "
        f"{output_file.name}"
    )


def main():
    """Download data for all assets."""

    if not API_KEY:
        raise ValueError(
            "POLYGON_API_KEY was not found in .env"
        )

    for ticker in TICKERS:
        try:
            download_market_data(ticker)
        except Exception as error:
            print(f"ERROR for {ticker}: {error}")


if __name__ == "__main__":
    main()
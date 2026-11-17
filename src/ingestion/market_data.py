import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load API key from .env
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("POLYGON_API_KEY")

# Date range. A 12-month momentum factor needs more than twelve months of
# history, because the first year is consumed by the lookback. Defaults to
# three years ending today, overridable from .env or the command line.
#
#   QS_START_DATE / QS_END_DATE in .env, or
#   python src/ingestion/market_data.py --start 2019-01-01 --end 2026-11-01
DEFAULT_YEARS_OF_HISTORY = 3

END_DATE = os.getenv(
    "QS_END_DATE",
    date.today().isoformat(),
)
START_DATE = os.getenv(
    "QS_START_DATE",
    (
        date.fromisoformat(END_DATE)
        - timedelta(days=365 * DEFAULT_YEARS_OF_HISTORY)
    ).isoformat(),
)

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

# Polygon's free tier allows only a handful of calls per minute, so requests
# are spaced out. Set QS_REQUEST_PAUSE=0 on a paid plan.
REQUEST_PAUSE_SECONDS = float(os.getenv("QS_REQUEST_PAUSE", "13"))


def download_market_data(ticker, start=None, end=None):
    """Download daily market data for one asset."""

    start = start or START_DATE
    end = end or END_DATE

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/"
        f"{ticker}/range/1/day/{start}/{end}"
    )

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)

    if response.status_code == 429:
        raise Exception(
            f"Rate limited by Polygon for {ticker}. The free tier allows "
            f"only a few calls per minute. Raise QS_REQUEST_PAUSE (currently "
            f"{REQUEST_PAUSE_SECONDS:g}s) and run again."
        )

    if response.status_code == 403:
        raise Exception(
            f"Polygon refused the request for {ticker} (403). The key may be "
            f"invalid or revoked, or the plan may not cover this endpoint."
        )

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

    parser = argparse.ArgumentParser(
        description="Download daily OHLCV from Polygon"
    )
    parser.add_argument("--start", default=START_DATE,
                        help=f"start date, YYYY-MM-DD (default {START_DATE})")
    parser.add_argument("--end", default=END_DATE,
                        help=f"end date, YYYY-MM-DD (default {END_DATE})")
    parser.add_argument("--pause", type=float,
                        default=REQUEST_PAUSE_SECONDS,
                        help="seconds between requests, for rate limits")
    args = parser.parse_args()

    if not API_KEY:
        raise ValueError(
            "POLYGON_API_KEY was not found in .env. Copy .env.example to "
            ".env and set it. Never commit .env."
        )

    print(f"Range: {args.start} to {args.end}")
    print(f"Pausing {args.pause:g}s between requests for rate limits\n")

    succeeded, failed = [], []

    for index, ticker in enumerate(TICKERS):
        try:
            download_market_data(ticker, start=args.start, end=args.end)
            succeeded.append(ticker)
        except Exception as error:
            print(f"ERROR for {ticker}: {error}")
            failed.append(ticker)

        if args.pause > 0 and index < len(TICKERS) - 1:
            time.sleep(args.pause)

    print(f"\nDownloaded {len(succeeded)}/{len(TICKERS)}: "
          f"{', '.join(succeeded) if succeeded else 'none'}")

    if failed:
        # Loud, because a partial download silently shrinks the universe and
        # every cross-sectional score downstream is computed against it.
        print(f"FAILED {len(failed)}: {', '.join(failed)}")
        print("The ranking standardises across whichever assets are present,")
        print("so re-run for the missing ones before trusting the output.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

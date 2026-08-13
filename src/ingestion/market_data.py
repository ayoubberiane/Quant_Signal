import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load the API key from .env
load_dotenv()

API_KEY = os.getenv("POLYGON_API_KEY")

TICKER = "SPY"
START_DATE = "2025-01-01"
END_DATE = "2025-12-31"


def download_market_data():
    """Download daily market data from Polygon."""

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/"
        f"{TICKER}/range/1/day/{START_DATE}/{END_DATE}"
    )

    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        raise Exception(
            f"API request failed: {response.status_code}\n"
            f"{response.text}"
        )

    data = response.json()

    if "results" not in data:
        raise Exception(f"No results returned: {data}")

    df = pd.DataFrame(data["results"])

    # Convert Polygon timestamp into a readable date
    df["date"] = pd.to_datetime(df["t"], unit="ms")

    # Rename Polygon's abbreviated column names
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

    # Keep only the columns we need
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

    # Save the data
    df.to_csv("data/raw/SPY.csv", index=False)

    print(f"Downloaded {len(df)} rows for {TICKER}")
    print("Saved to data/raw/SPY.csv")

if __name__ == "__main__":
    download_market_data(

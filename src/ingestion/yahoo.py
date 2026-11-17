"""Daily prices from Yahoo Finance, in the same shape as the Polygon files.

Used by the dashboard, which has to run anywhere without an API key. The
research pipeline still uses Polygon (src/ingestion/market_data.py).

Prices are split- and dividend-adjusted. Yahoo has no VWAP, so `vwap` is
the typical price (high + low + close) / 3. Nothing in the score uses it;
it is filled only so the columns match the cleaning step's contract.
"""

from __future__ import annotations

import pandas as pd

COLUMNS = ["date", "open", "high", "low", "close", "volume", "vwap", "ticker"]


def normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """One ticker's yfinance frame -> the project's long OHLCV format."""
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame.columns = [str(c).lower() for c in frame.columns]
    frame = frame.rename_axis("date").reset_index()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None)
    frame["vwap"] = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    frame["ticker"] = ticker
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    return frame[COLUMNS].sort_values("date").reset_index(drop=True)


def download(tickers: list[str], years: int = 3) -> pd.DataFrame:
    """Download `years` of daily data for each ticker; one long frame.

    Raises if any ticker comes back empty: a silently missing asset shrinks
    the cross-section every score is standardised against.
    """
    import yfinance as yf

    frames, missing = [], []
    for ticker in tickers:
        raw = yf.download(ticker, period=f"{years}y", interval="1d",
                          auto_adjust=True, progress=False)
        if raw is None or raw.empty:
            missing.append(ticker)
            continue
        frames.append(normalise(raw, ticker))

    if missing:
        raise RuntimeError(
            f"Yahoo returned no data for {', '.join(missing)}. "
            "Try again in a minute; Yahoo rate-limits bursts."
        )
    return pd.concat(frames, ignore_index=True)

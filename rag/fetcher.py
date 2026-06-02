"""
fetcher.py
----------
Fetches real stock data using yfinance (free, no API key needed)
and converts each trading day into a plain-English text chunk for RAG.
"""

import yfinance as yf
import pandas as pd
from typing import List, Tuple


def fetch_stock_data(
    ticker: str, period: str = "1y"
) -> Tuple[pd.DataFrame | None, List[str]]:
    """
    Download OHLCV data for a ticker and convert to text chunks.

    Args:
        ticker: Stock symbol e.g. "AAPL"
        period: yfinance period string e.g. "1y", "6mo", "2y"

    Returns:
        (DataFrame with OHLCV data, list of plain-English day descriptions)
    """
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None, []

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna()
        chunks = _build_chunks(df, ticker)
        return df, chunks

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None, []


def _build_chunks(df: pd.DataFrame, ticker: str) -> List[str]:
    """
    Convert each row of the OHLCV dataframe into a natural language description.
    These become the documents that get embedded and searched.
    """
    chunks = []

    for date, row in df.iterrows():
        date_str  = date.strftime("%B %d, %Y")        # e.g. "June 03, 2024"
        day_change = row["Close"] - row["Open"]
        direction  = "up" if day_change >= 0 else "down"
        pct_change = (day_change / row["Open"]) * 100

        text = (
            f"On {date_str}, {ticker} stock opened at ${row['Open']:.2f}, "
            f"reached a high of ${row['High']:.2f}, a low of ${row['Low']:.2f}, "
            f"and closed at ${row['Close']:.2f}. "
            f"The stock moved {direction} by ${abs(day_change):.2f} "
            f"({abs(pct_change):.2f}%) on the day. "
            f"Trading volume was {int(row['Volume']):,} shares."
        )
        chunks.append(text)

    return chunks

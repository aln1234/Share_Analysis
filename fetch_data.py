import asyncio
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import aiohttp
import streamlit as st
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

# Alpaca API credentials
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL_CRYPTO = 'https://data.alpaca.markets/v1beta3/crypto/us'
BASE_URL_ASSETS = 'https://data.alpaca.markets/v2'


async def fetch_ticker_data(asset_class, ticker, session, start_date, end_date, timeframe="1H", limit=5000):
    """
    Fetch historical data for a single ticker within a specified date range.
    Handles cryptocurrencies and other asset classes.
    """
    if asset_class == 'crypto':
        url = f"{BASE_URL_CRYPTO}/bars"
    else:
        url = f"{BASE_URL_ASSETS}/stocks/bars"

    params = {
        "symbols": ticker,
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date,
        "limit": limit
    }
    headers = {
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': API_SECRET
    }

    async with session.get(url, headers=headers, params=params) as response:
        if response.status == 200:
            data = await response.json()
            return data
        else:
            st.error(f"Error fetching data for {
                     ticker}: HTTP {response.status}")
            return {}


async def fetch_all_data(asset_class, tickers, start_date, end_date, timeframe="1H", limit=5000):
    """
    Fetch historical data for multiple tickers within a date range.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_ticker_data(
            asset_class, ticker, session, start_date, end_date, timeframe, limit) for ticker in tickers]
        return await asyncio.gather(*tasks)


def validate_and_process_data(raw_data, tickers, asset_class, required_rows=20):
    """
    Validate and process the fetched data to ensure sufficient data for each ticker.
    """
    processed_data = []
    for ticker, data in zip(tickers, raw_data):
        bars = data.get('bars', {}).get(
            ticker, []) if asset_class == 'crypto' else data.get('bars', [])
        if len(bars) >= required_rows:
            for bar in bars:
                processed_data.append({
                    "symbol": ticker,
                    "timestamp": bar['t'],
                    "open": bar['o'],
                    "high": bar['h'],
                    "low": bar['l'],
                    "close": bar['c'],
                    "volume": bar['v']
                })
        else:
            st.warning(f"Insufficient data for {ticker}: {
                       len(bars)} rows fetched.")

    if processed_data:
        df = pd.DataFrame(processed_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.sort_values(by=['symbol', 'timestamp'])
    else:
        st.warning("No valid data available for processing.")
        return pd.DataFrame()


def compute_fibonacci_levels(df):
    """
    Compute Fibonacci retracement levels for a given symbol's price data.
    """
    high = df['high'].max()
    low = df['low'].min()
    difference = high - low

    levels = {
        "0%": high,
        "23.6%": high - 0.236 * difference,
        "38.2%": high - 0.382 * difference,
        "50%": high - 0.5 * difference,
        "61.8%": high - 0.618 * difference,
        "100%": low
    }

    return levels


def compute_indicators(group):
    """
    Compute technical indicators including Fibonacci levels, MACD, and others.
    """
    if len(group) < 20:
        st.warning(f"Insufficient data for group {group['symbol'].iloc[0]}")
        group['rsi'] = np.nan
        group['macd'] = np.nan
        group['macd_signal'] = np.nan
        group['macd_hist'] = np.nan
        group['bb_upper'], group['bb_lower'] = np.nan, np.nan
        group['atr'] = np.nan
        group['price_change'] = np.nan
        group['volume_spike'] = np.nan
        return group

    # Calculate RSI
    group['rsi'] = ta.rsi(group['close'], length=14)

    # Calculate MACD
    macd = ta.macd(group['close'], fast=12, slow=26, signal=9)
    group['macd'] = macd['MACD_12_26_9']
    group['macd_signal'] = macd['MACDs_12_26_9']
    group['macd_hist'] = macd['MACDh_12_26_9']

    # Calculate Bollinger Bands
    bb = ta.bbands(group['close'], length=20)
    group['bb_upper'] = bb['BBU_20_2.0']
    group['bb_lower'] = bb['BBL_20_2.0']

    # Calculate ATR
    group['atr'] = ta.atr(group['high'], group['low'],
                          group['close'], length=14)

    # Calculate Percentage Price Change
    group['price_change'] = group['close'].pct_change() * 100

    # Calculate Volume Spike
    group['volume_spike'] = group['volume'] / \
        group['volume'].rolling(window=30, min_periods=1).mean()

    # Compute Fibonacci Levels
    fib_levels = compute_fibonacci_levels(group)
    group['fib_0%'] = fib_levels['0%']
    group['fib_23.6%'] = fib_levels['23.6%']
    group['fib_38.2%'] = fib_levels['38.2%']
    group['fib_50%'] = fib_levels['50%']
    group['fib_61.8%'] = fib_levels['61.8%']
    group['fib_100%'] = fib_levels['100%']

    return group


def save_dataset_for_llm(df):
    """
    Save the processed and classified dataset for LLM fine-tuning or inference.
    """
    # Extract relevant fields for the LLM
    llm_data = df[['symbol', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                   'bb_upper', 'bb_lower', 'atr', 'price_change', 'volume_spike',
                   'fib_0%', 'fib_23.6%', 'fib_38.2%', 'fib_50%', 'fib_61.8%', 'fib_100%']].copy()

    # Create formatted text for LLM
    llm_data['text'] = llm_data.apply(
        lambda row: (f"Symbol: {row['symbol']}, RSI: {row['rsi']:.2f}, MACD: {row['macd']:.2f}, "
                     f"MACD Signal: {row['macd_signal']:.2f}, MACD Histogram: {
                         row['macd_hist']:.2f}, "
                     f"Bollinger Bands Upper: {row['bb_upper']:.2f}, Bollinger Bands Lower: {
                         row['bb_lower']:.2f}, "
                     f"ATR: {row['atr']:.2f}, Price Change: {
                         row['price_change']:.2f}%, "
                     f"Volume Spike: {row['volume_spike']:.2f}, Fibonacci Levels: [0%: {
                         row['fib_0%']:.2f}, "
                     f"23.6%: {row['fib_23.6%']:.2f}, 38.2%: {
                         row['fib_38.2%']:.2f}, 50%: {row['fib_50%']:.2f}, "
                     f"61.8%: {row['fib_61.8%']:.2f}, 100%: {row['fib_100%']:.2f}]. What should be the action?"),
        axis=1
    )
    llm_data = llm_data[['text']]  # Keep only text field

    # Convert to DataFrame and save as JSON
    llm_data.to_json("llm_dataset.json", orient="records", indent=4)
    st.success("Dataset for LLM saved as 'llm_dataset.json'.")

    return llm_data


def display_and_prepare_data(df):
    """
    Display processed data in tables and create a DataFrame for LLM.
    """
    st.write("### Processed Technical Indicators with Fibonacci and MACD")
    st.write("The table below shows the computed technical indicators for the assets.")
    st.dataframe(df)

    # Prepare dataset for LLM
    llm_dataset = save_dataset_for_llm(df)
    return llm_dataset


async def fetch_crypto_data(asset_class):
    """
    Fetch cryptocurrency data, compute indicators, and display results in Streamlit.
    """
    headers = {
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': API_SECRET
    }

    response = requests.get(
        "https://paper-api.alpaca.markets/v2/assets",
        headers=headers,
        params={'asset_class': asset_class, 'status': 'active'}
    )

    if response.status_code == 200:
        assets = response.json()
        tickers = [asset['symbol'] for asset in assets if asset['tradable']]
        tickers = tickers[:10]  # Limit the number of tickers for demonstration

        start_date = "2019-01-01T00:00:00Z"
        end_date = (datetime.now(timezone.utc) - timedelta(hours=1)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Fetch historical data
        raw_data = await fetch_all_data(asset_class, tickers, start_date, end_date)

        # Validate and process the fetched data
        df = validate_and_process_data(raw_data, tickers, asset_class)

        if not df.empty:
            # Compute technical indicators and Fibonacci levels
            indicators_df = df.groupby(
                'symbol', group_keys=False).apply(compute_indicators)

            # Display data and prepare dataset for LLM
            llm_dataset = display_and_prepare_data(indicators_df)
            return llm_dataset

    else:
        st.error(f"Failed to fetch assets: HTTP {response.status_code}")
        return None


if __name__ == "__main__":
    st.title("Asset Analysis (Crypto, US Equity, and Index Funds)")

    asset_class = st.radio("Select Asset Class",
                           ("crypto", "us_equity", "index_fund"))
    st.write(f"Fetching and analyzing data for {asset_class}...")

    # Run the analysis and get the dataset for LLM
    llm_dataset = asyncio.run(fetch_crypto_data(asset_class))

    if llm_dataset is not None:
        st.write("### Dataset for LLM")
        st.dataframe(llm_dataset)

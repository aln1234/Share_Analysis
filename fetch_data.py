import asyncio
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import aiohttp
import streamlit as st
from datetime import datetime
from datetime import timezone
import time

from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()


# Alpaca API credentials
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL = 'https://data.alpaca.markets/v1beta3/crypto/us'


async def fetch_ticker_data(ticker, session, start_date, end_date, timeframe="1H", limit=5000):
    """
    Fetch historical data for a single cryptocurrency ticker within a specified date range.
    """
    url = f"{BASE_URL}/bars"
    headers = {
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': API_SECRET
    }
    params = {
        "symbols": ticker,
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date,
        "limit": limit
    }
    async with session.get(url, headers=headers, params=params) as response:
        if response.status == 200:
            data = await response.json()
            return data
        else:
            st.error(f"Error fetching data for {
                     ticker}: HTTP {response.status}")
            return {}


async def fetch_all_data(tickers, start_date, end_date, timeframe="1H", limit=5000):
    """
    Fetch historical data for multiple cryptocurrency tickers within a date range.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_ticker_data(
            ticker, session, start_date, end_date, timeframe, limit) for ticker in tickers]
        return await asyncio.gather(*tasks)


def validate_and_process_data(raw_data, tickers, required_rows=20):
    """
    Validate and process the fetched data to ensure sufficient data for each ticker.
    """
    processed_data = []
    for ticker, data in zip(tickers, raw_data):
        bars = data.get('bars', {}).get(ticker, [])
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


def compute_indicators(group):
    """
    Compute technical indicators using pandas_ta for a group of cryptocurrency data.
    """
    if len(group) < 20:
        st.warning(f"Insufficient data for group {group['symbol'].iloc[0]}")
        group['rsi'] = np.nan
        group['bb_upper'], group['bb_lower'] = np.nan, np.nan
        group['atr'] = np.nan
        group['price_change'] = np.nan
        group['volume_spike'] = np.nan
        return group

    # Calculate RSI
    group['rsi'] = ta.rsi(group['close'], length=14)

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

    return group


def classify_stocks(df):
    """
    Classify stocks based on composite scores derived from RSI, volume spike, and price change.
    """
    df['rsi_score'] = 1 - (df['rsi'] - 20) / (70 - 20)
    df['volume_score'] = (df['volume_spike'] - df['volume_spike'].min()) / (
        df['volume_spike'].max() - df['volume_spike'].min())
    df['price_change_score'] = (df['price_change'] - df['price_change'].min()) / (
        df['price_change'].max() - df['price_change'].min())

    weights = {'rsi_score': 0.4, 'volume_score': 0.4,
               'price_change_score': 0.2}
    df['composite_score'] = (
        df['rsi_score'] * weights['rsi_score'] +
        df['volume_score'] * weights['volume_score'] +
        df['price_change_score'] * weights['price_change_score']
    )
    df['classification'] = np.where(df['composite_score'] > 0.8, 'Buy',
                                    np.where(df['composite_score'] < 0.5, 'Sell', 'Hold'))
    return df


def display_top_shares(df):
    """
    Display the top 5 shares for each classification in Streamlit.
    """
    st.write("### Count of rows for each classification:")
    st.write(df['classification'].value_counts())

    for signal in ['Buy', 'Sell', 'Hold']:
        st.write(f"### {signal} Signals")
        top_shares = df[df['classification'] == signal].sort_values(
            by='composite_score', ascending=False).head(5)
        if top_shares.empty:
            st.warning(f"No shares found for {signal} signal.")
        else:
            st.table(top_shares[['symbol', 'composite_score']])


def display_technical_indicators(df):
    """
    Display top cryptocurrencies for each technical indicator in Streamlit.
    """
    st.write("### Top Cryptocurrencies Based on Technical Indicators")

    # Top RSI
    st.write("#### Top 5 by RSI")
    top_rsi = df.sort_values(by='rsi', ascending=False).head(5)
    if top_rsi.empty:
        st.warning("No data for RSI.")
    else:
        st.table(top_rsi[['symbol', 'rsi']])

    # Top Bollinger Band Upper
    st.write("#### Top 5 by Bollinger Bands Upper")
    top_bb_upper = df.sort_values(by='bb_upper', ascending=False).head(5)
    if top_bb_upper.empty:
        st.warning("No data for Bollinger Bands Upper.")
    else:
        st.table(top_bb_upper[['symbol', 'bb_upper']])

    # Top Bollinger Band Lower
    st.write("#### Top 5 by Bollinger Bands Lower")
    top_bb_lower = df.sort_values(by='bb_lower').head(5)
    if top_bb_lower.empty:
        st.warning("No data for Bollinger Bands Lower.")
    else:
        st.table(top_bb_lower[['symbol', 'bb_lower']])

    # Top ATR
    st.write("#### Top 5 by ATR")
    top_atr = df.sort_values(by='atr', ascending=False).head(5)
    if top_atr.empty:
        st.warning("No data for ATR.")
    else:
        st.table(top_atr[['symbol', 'atr']])

    # Top by Volume Spike
    st.write("#### Top 5 by Volume Spike")
    top_volume_spike = df.sort_values(
        by='volume_spike', ascending=False).head(5)
    if top_volume_spike.empty:
        st.warning("No data for Volume Spike.")
    else:
        st.table(top_volume_spike[['symbol', 'volume_spike']])

    # Top by Price Change
    st.write("#### Top 5 by Price Change")
    top_price_change = df.sort_values(
        by='price_change', ascending=False).head(5)
    if top_price_change.empty:
        st.warning("No data for Price Change.")
    else:
        st.table(top_price_change[['symbol', 'price_change']])


# Set OpenAI API key
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)


# async def send_to_openai(data):
#     """
#     Send detailed analysis including RSI, Bollinger Bands, ATR, and other indicators to OpenAI GPT model.
#     """
#     # Prepare the prompt
#     prompt = f"""
#     Here is the detailed cryptocurrency analysis:
#     {data}

#     The data includes:
#     - RSI: Relative Strength Index
#     - Bollinger Bands (Upper and Lower)
#     - ATR: Average True Range
#     - Price Change (%)
#     - Volume Spike
#     - Composite Score (aggregated indicator)

#     Based on this analysis, please recommend:
#     - Which cryptocurrencies to Buy, Hold, or Sell.
#     - Highlight the reasons for your recommendations.
#     """

#     try:
#         # Call OpenAI's GPT model using the client object
#         chat_completion = client.chat.completions.create(
#             model="gpt-3.5-turbo",  # Use "gpt-4" if "gpt-4o" is unavailable
#             messages=[
#                 {"role": "system", "content": "You are a financial advisor analyzing cryptocurrency data."},
#                 {"role": "user", "content": prompt}
#             ]
#         )

#         # Return the response content
#         return chat_completion['choices'][0]['message']['content']
#     except Exception as e:
#         # Handle and log any errors
#         print(f"Error calling OpenAI API: {e}")
#         return "An error occurred while fetching recommendations. Please try again later."


async def fetch_crypto_data():
    """
    Fetch cryptocurrency data, compute indicators, classify stocks, and display results in Streamlit.
    """
    headers = {
        'APCA-API-KEY-ID': API_KEY,
        'APCA-API-SECRET-KEY': API_SECRET
    }

    response = requests.get(
        "https://paper-api.alpaca.markets/v2/assets",
        headers=headers,
        params={'asset_class': 'crypto', 'status': 'active'}
    )

    if response.status_code == 200:
        assets = response.json()
        tickers = [asset['symbol'] for asset in assets if asset['tradable']]
        tickers = tickers[:10]

        start_date = "2023-01-01T00:00:00Z"
        end_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        raw_data = await fetch_all_data(tickers, start_date, end_date)
        df = validate_and_process_data(raw_data, tickers)

        if not df.empty:
            indicators_df = df.groupby(
                'symbol', group_keys=False).apply(compute_indicators)
            classified_df = classify_stocks(indicators_df)

            # Display classification results
            display_top_shares(classified_df)

            # Display top indicators
            display_technical_indicators(classified_df)

            # Include detailed technical indicators in the analysis summary
            analysis_summary = classified_df[['symbol', 'rsi', 'bb_upper', 'bb_lower', 'atr',
                                              'price_change', 'volume_spike', 'classification', 'composite_score']].to_dict(orient="records")

            # Send the detailed analysis to Claude
            # recommendations = await send_to_openai(analysis_summary)

            if recommendations:
                st.write("### Claude's Recommendations")
                st.write(recommendations)
            else:
                st.warning("No sufficient data to process indicators.")
    else:
        st.error(f"Failed to fetch assets: HTTP {response.status_code}")


if __name__ == "__main__":
    st.title("Cryptocurrency Analysis")
    st.write("Fetching and analyzing cryptocurrency data...")

    # Refresh the page every minute
    asyncio.run(fetch_crypto_data())
    time.sleep(60)

    st.rerun()

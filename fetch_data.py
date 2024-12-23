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
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

load_dotenv()

# Alpaca API credentials
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
BASE_URL_CRYPTO = 'https://data.alpaca.markets/v1beta3/crypto/us'
BASE_URL_ASSETS = 'https://data.alpaca.markets/v2'

# Email credentials
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")


async def fetch_ticker_data(asset_class, ticker, session, start_date, end_date, timeframe="1H", limit=5000):
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
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_ticker_data(
            asset_class, ticker, session, start_date, end_date, timeframe, limit) for ticker in tickers]
        return await asyncio.gather(*tasks)


def compute_fibonacci_levels(df):
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


def validate_and_process_data(raw_data, tickers, asset_class, required_rows=20):
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


def compute_indicators(group):
    """
    Compute technical indicators, including Fibonacci levels, MACD, RSI, Bollinger Bands, and others.
    """
    if len(group) < 20:
        st.warning(f"Insufficient data for group {group['symbol'].iloc[0]}")
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

    # Add Rolling Mean for ATR
    group['atr_rolling_mean'] = group['atr'].rolling(window=14).mean()

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


def refine_scoring_logic(df):
    """
    Refine the scoring logic to incorporate all available technical indicators.
    """
    # Initialize score column
    df['score'] = 0

    # RSI Scoring
    df['score'] += (df['rsi'] < 30).astype(int)  # +1 if RSI < 30 (oversold)
    df['score'] -= (df['rsi'] > 70).astype(int)  # -1 if RSI > 70 (overbought)

    # MACD Histogram Scoring
    # +1 if MACD Histogram > 0 (bullish momentum)
    df['score'] += (df['macd_hist'] > 0).astype(int)
    # -1 if MACD Histogram < 0 (bearish momentum)
    df['score'] -= (df['macd_hist'] < 0).astype(int)

    # Bollinger Bands Scoring
    # +1 if Close < Lower Bollinger Band (oversold)
    df['score'] += (df['close'] < df['bb_lower']).astype(int)
    # -1 if Close > Upper Bollinger Band (overbought)
    df['score'] -= (df['close'] > df['bb_upper']).astype(int)

    # ATR Scoring
    # +1 if ATR > Rolling Mean (high volatility)
    df['score'] += (df['atr'] > df['atr_rolling_mean']).astype(int)

    # Price Change Scoring
    # +1 if Price Change > 0 (positive momentum)
    df['score'] += (df['price_change'] > 0).astype(int)
    # -1 if Price Change < 0 (negative momentum)
    df['score'] -= (df['price_change'] < 0).astype(int)

    # Volume Spike Scoring
    # +1 if Volume Spike > 1.5 (high market interest)
    df['score'] += (df['volume_spike'] > 1.5).astype(int)
    # -1 if Volume Spike < 0.5 (low market interest)
    df['score'] -= (df['volume_spike'] < 0.5).astype(int)

    # Fibonacci Level Scoring
    df['score'] += ((df['close'] >= df['fib_23.6%']) & (df['close']
                    # +1 near Fibonacci support
                                                        <= df['fib_38.2%'])).astype(int)
    df['score'] += ((df['close'] >= df['fib_38.2%']) & (df['close']
                    <= df['fib_50%'])).astype(int)  # +1 near key support
    # -1 near Fibonacci resistance
    df['score'] -= (df['close'] >= df['fib_61.8%']).astype(int)
    # -1 near Fibonacci high
    df['score'] -= (df['close'] >= df['fib_0%']).astype(int)

    return df


# Function to send email
def send_email(best_shares):
    subject = "Daily Top 3 Shares Report"

    # Prepare email content with technical details
    body = """
    <html>
    <body>
        <h2 style='color: #2E86C1;'>Top 3 Shares to Buy Today</h2>
        <p>Here are the top 3 shares to watch based on technical analysis:</p>
    """

    for index, row in best_shares.iterrows():
        body += f"""
        <div style='margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px;'>
            <h3 style='color: #16A085;'>Share: {row['symbol']}</h3>
            <ul>
                <li><strong>RSI:</strong> {row['rsi']:.2f} - Indicates {'oversold' if row['rsi'] < 30 else 'overbought' if row['rsi'] > 70 else 'neutral'} market conditions.</li>
                <li><strong>MACD Histogram:</strong> {row['macd_hist']:.2f} - Suggests {'bullish' if row['macd_hist'] > 0 else 'bearish'} momentum.</li>
                <li><strong>Bollinger Bands:</strong> Close price is {'below' if row['close'] < row['bb_lower'] else 'above' if row['close'] > row['bb_upper'] else 'within'} the band, indicating {'oversold' if row['close'] < row['bb_lower'] else 'overbought' if row['close'] > row['bb_upper'] else 'stable'} conditions.</li>
                <li><strong>Price Change:</strong> {row['price_change']:.2f}% - Reflects {'positive' if row['price_change'] > 0 else 'negative'} momentum.</li>
                <li><strong>Volume Spike:</strong> {row['volume_spike']:.2f} - Suggests {'high' if row['volume_spike'] > 1.5 else 'low'} market interest.</li>
            </ul>
        </div>
        """

    body += """
        <p style='color: #555;'>This report is generated based on the latest technical indicators. Always conduct additional research before making trading decisions.</p>
    </body>
    </html>
    """

    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_TO
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_USER, EMAIL_TO, msg.as_string())
            print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")


def find_best_shares(df, top_n=3):
    if not df.empty:
        return df.sort_values(by='score', ascending=False).head(top_n)
    return pd.DataFrame()


def display_detailed_tables(df):
    st.write("### Detailed Technical Analysis")

    indicator_sorting = {
        'rsi': True,  # Low RSI is good, sort ascending
        'macd_hist': False,  # High MACD Histogram is good, sort descending
        'price_change': False,  # High price change is good, sort descending
        'volume_spike': False,  # High volume spike is good, sort descending
        'atr': False  # High ATR is good, sort descending
    }

    # RSI Description and Table
    st.write("""
    **RSI (Relative Strength Index)**: 
    - RSI is used to measure the speed and change of price movements.
    - RSI values above 70 typically indicate an overbought condition, while values below 30 indicate an oversold condition.
    """)
    sorted_df = df[['symbol', 'rsi']].sort_values(by='rsi', ascending=True)
    st.write(f"#### Top 3 Shares by RSI (Relative Strength Index)")
    st.table(sorted_df.head(3))

    # MACD Description and Table
    st.write("""
    **MACD (Moving Average Convergence Divergence)**:
    - The MACD is used to identify changes in the strength, direction, momentum, and duration of a trend.
    - When the MACD line crosses above the signal line, it's considered bullish, and when it crosses below, it's considered bearish.
    """)
    sorted_df = df[['symbol', 'macd_hist']].sort_values(
        by='macd_hist', ascending=False)
    st.write(f"#### Top 3 Shares by MACD Histogram")
    st.table(sorted_df.head(3))

    # Price Change Description and Table
    st.write("""
    **Price Change**:
    - The price change is calculated as the percentage change in the closing price.
    - Positive values indicate upward momentum, and negative values indicate downward momentum.
    """)
    sorted_df = df[['symbol', 'price_change']].sort_values(
        by='price_change', ascending=False)
    st.write(f"#### Top 3 Shares by Price Change")
    st.table(sorted_df.head(3))

    # Volume Spike Description and Table
    st.write("""
    **Volume Spike**:
    - A volume spike occurs when there is a significant increase in trading volume.
    - A high volume spike indicates strong market interest, often preceding significant price movements.
    """)
    sorted_df = df[['symbol', 'volume_spike']].sort_values(
        by='volume_spike', ascending=False)
    st.write(f"#### Top 3 Shares by Volume Spike")
    st.table(sorted_df.head(3))

    # ATR Description and Table
    st.write("""
    **ATR (Average True Range)**:
    - ATR measures market volatility.
    - A high ATR indicates high volatility, while a low ATR indicates low volatility.
    """)
    sorted_df = df[['symbol', 'atr']].sort_values(by='atr', ascending=False)
    st.write(f"#### Top 3 Shares by ATR (Average True Range)")
    st.table(sorted_df.head(3))

    # Add explanations for Fibonacci levels as well
    st.write("""
    **Fibonacci Levels**:
    - Fibonacci retracement levels are key levels that indicate potential areas of support and resistance.
    - Common levels include 23.6%, 38.2%, 50%, 61.8%, and 100%.
    - These levels are used by traders to identify reversal points in the market.
    """)

    # Optionally, add sorting by Fibonacci levels if required
    # For example, sorting by close price proximity to a Fibonacci level:
    st.write(f"#### Top 3 Shares by Fibonacci Levels")
    sorted_df = df[['symbol', 'fib_23.6%', 'fib_38.2%', 'fib_50%']
                   ].sort_values(by='fib_23.6%', ascending=True)
    st.table(sorted_df.head(3))


async def fetch_crypto_data(asset_class):
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

        start_date = "2015-01-01T00:00:00Z"
        end_date = (datetime.now(timezone.utc) - timedelta(hours=1)
                    ).strftime("%Y-%m-%dT%H:%M:%SZ")

        raw_data = await fetch_all_data(asset_class, tickers, start_date, end_date)
        df = validate_and_process_data(raw_data, tickers, asset_class)

        if not df.empty:
            indicators_df = df.groupby(
                'symbol', group_keys=False).apply(compute_indicators)
            refined_df = refine_scoring_logic(indicators_df)
            best_shares = find_best_shares(refined_df)

            # Display in Streamlit (optional)
            display_detailed_tables(refined_df)
            st.write("### Best Shares to Buy")
            st.table(best_shares[['symbol', 'score']])

            return best_shares  # Explicitly return the result

    else:
        st.error(f"Failed to fetch assets: HTTP {response.status_code}")
        return None  # Return None if the fetch fails


# Global dictionary to store shared results
shared_data = {"best_shares": None}


def scheduler_task():
    try:
        # Get asset class from shared data or use a default
        asset_class = shared_data.get("asset_class", "crypto")
        print(f"Running scheduler_task for asset_class: {asset_class}...")

        # Fetch the best shares data
        best_shares = asyncio.run(fetch_crypto_data(asset_class))

        # Update shared data for UI
        if best_shares is not None and not best_shares.empty:
            shared_data["best_shares"] = best_shares

            # Send email with the fetched data
            send_email(best_shares)
            print("Email sent successfully with the latest best shares.")
        else:
            print("No valid data fetched for email.")
    except Exception as e:
        print(f"Error in scheduler_task: {e}")


if __name__ == "__main__":
    st.title("Asset Analysis (Crypto, US Equity, and Index Funds)")

    # Initialize session state
    if 'best_shares' not in st.session_state:
        st.session_state['best_shares'] = None
    if 'asset_class' not in st.session_state:
        st.session_state['asset_class'] = "crypto"

    # Asset class selection
    asset_class = st.radio("Select Asset Class",
                           ("crypto", "us_equity", "index_fund"))
    st.session_state['asset_class'] = asset_class
    st.write(f"Fetching and analyzing data for {asset_class}...")

    # Display the current best shares if available
    if st.session_state['best_shares'] is not None:
        st.write("### Latest Recommendations")
        st.table(st.session_state['best_shares'])

    # Initialize the scheduler
    scheduler = BackgroundScheduler()
    scheduler.add_job(scheduler_task, IntervalTrigger(
        minutes=30))  # Data updates every 15 minutes
    scheduler.start()

    # Run the first task manually for testing
    scheduler_task()

    st.success(
        "Scheduler started. Data updates every 15 minutes, and emails are sent periodically.")

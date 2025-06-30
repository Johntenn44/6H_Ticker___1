import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import traceback  # For detailed error logging

# --- CONFIGURATION ---

COINS = [
    "XRP/USDT", "XMR/USDT", "GMX/USDT", "LUNA/USDT", "TRX/USDT",
    "EIGEN/USDT", "APE/USDT", "WAVES/USDT", "PLUME/USDT", "SUSHI/USDT",
    "DOGE/USDT", "VIRTUAL/USDT", "CAKE/USDT", "GRASS/USDT", "AAVE/USDT",
    "SUI/USDT", "ARB/USDT", "XLM/USDT", "MNT/USDT", "LTC/USDT", "NEAR/USDT",
]

EXCHANGE_ID = 'kucoin'
INTERVAL = '12h'      # 12-hour candles
LOOKBACK = 210       # Number of candles to fetch (>= 200)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    raise ValueError("Telegram bot token or chat ID not set in environment variables.")

# --- INDICATOR CALCULATIONS ---

def calculate_rsi(series, period=13):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_rsi(df, rsi_length=13, stoch_length=8, smooth_k=5, smooth_d=3):
    rsi = calculate_rsi(df['close'], rsi_length)
    min_rsi = rsi.rolling(window=stoch_length).min()
    max_rsi = rsi.rolling(window=stoch_length).max()
    denominator = max_rsi - min_rsi
    denominator = denominator.replace(0, np.nan)  # avoid division by zero
    stoch_rsi = (rsi - min_rsi) / denominator * 100
    stoch_rsi = stoch_rsi.fillna(method='ffill')
    k = stoch_rsi.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_wr(df, length):
    highest_high = df['high'].rolling(window=length).max()
    lowest_low = df['low'].rolling(window=length).min()
    denominator = highest_high - lowest_low
    denominator = denominator.replace(0, np.nan)  # avoid division by zero
    wr = (highest_high - df['close']) / denominator * -100
    return wr.fillna(method='ffill')

# --- TREND LOGIC ---

def analyze_stoch_rsi_trend(k, d):
    if len(k) < 2 or pd.isna(k.iloc[-2]) or pd.isna(d.iloc[-2]) or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
        return "No clear Stoch RSI trend"
    if k.iloc[-2] < d.iloc[-2] and k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
        return "Uptrend"
    elif k.iloc[-2] > d.iloc[-2] and k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
        return "Downtrend"
    else:
        return "No clear Stoch RSI trend"

def analyze_wr_trend(wr_series):
    if len(wr_series) < 2 or pd.isna(wr_series.iloc[-2]) or pd.isna(wr_series.iloc[-1]):
        return "No clear WR trend"
    prev, curr = wr_series.iloc[-2], wr_series.iloc[-1]
    if prev > -80 and curr <= -80:
        return "WR Oversold - Buy signal"
    elif prev < -20 and curr >= -20:
        return "WR Overbought - Sell signal"
    else:
        return "No clear WR trend"

# --- DATA FETCHING ---

def fetch_ohlcv_ccxt(symbol, timeframe, limit):
    exchange = getattr(ccxt, EXCHANGE_ID)()
    exchange.load_markets()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    return df

# --- TELEGRAM NOTIFICATION ---

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    resp = requests.post(url, data=payload)
    resp.raise_for_status()

# --- MAIN LOGIC ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    coins_with_trends = {}

    WR_PERIODS = [3, 8, 13, 55, 144, 233]

    for symbol in COINS:
        try:
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < max(13, 8, 5, 3, max(WR_PERIODS)):
                print(f"Not enough data for {symbol}")
                continue

            k, d = calculate_stoch_rsi(df, rsi_length=13, stoch_length=8, smooth_k=5, smooth_d=3)
            stoch_trend = analyze_stoch_rsi_trend(k, d)

            wr_trends = {}
            for period in WR_PERIODS:
                wr = calculate_wr(df, period)
                wr_trends[period] = analyze_wr_trend(wr)

            wr_signals = [f"WR{p}: {t}" for p, t in wr_trends.items() if t != "No clear WR trend"]

            if stoch_trend != "No clear Stoch RSI trend" or wr_signals:
                coins_with_trends[symbol] = {
                    "StochRSI": stoch_trend,
                    "WR": ", ".join(wr_signals) if wr_signals else "No WR signals"
                }

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            traceback.print_exc()

    if coins_with_trends:
        msg_lines = [f"<b>Kucoin {INTERVAL.upper()} Stochastic RSI + Williams %R Alert ({dt})</b>",
                     "Coins with detected trends:\n"]
        for coin, trends in coins_with_trends.items():
            msg_lines.append(f"{coin} - StochRSI Trend: {trends['StochRSI']} | WR Signals: {trends['WR']}")
        msg = "\n".join(msg_lines)
        send_telegram_message(msg)
    else:
        send_telegram_message("No coins satisfy the Stochastic RSI or Williams %R trend conditions at this time.")

if __name__ == "__main__":
    main()

import os
import ccxt
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
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
LOOKBACK = 500        # fetch enough data for indicators + backtest

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

def analyze_stoch_rsi_trend(k, d, idx):
    if idx < 1 or pd.isna(k.iloc[idx-1]) or pd.isna(d.iloc[idx-1]) or pd.isna(k.iloc[idx]) or pd.isna(d.iloc[idx]):
        return "No clear Stoch RSI trend"
    if k.iloc[idx-1] < d.iloc[idx-1] and k.iloc[idx] > d.iloc[idx] and k.iloc[idx] < 80:
        return "Uptrend"
    elif k.iloc[idx-1] > d.iloc[idx-1] and k.iloc[idx] < d.iloc[idx] and k.iloc[idx] > 20:
        return "Downtrend"
    else:
        return "No clear Stoch RSI trend"

def analyze_wr_trend(wr_series, idx):
    if idx < 1 or pd.isna(wr_series.iloc[idx-1]) or pd.isna(wr_series.iloc[idx]):
        return "No clear WR trend"
    prev, curr = wr_series.iloc[idx-1], wr_series.iloc[idx]
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

# --- BACKTEST LOGIC ---

def backtest(df):
    k, d = calculate_stoch_rsi(df)
    WR_PERIODS = [3, 8, 13, 55, 144, 233]
    wr_dict = {p: calculate_wr(df, p) for p in WR_PERIODS}

    position = 0  # 0 = no position, 1 = long
    entry_price = 0.0
    trades = []

    backtest_start = df.index[-1] - timedelta(days=5)  # 5-day backtest
    df_bt = df.loc[df.index >= backtest_start].copy()
    k_bt = k.loc[df_bt.index]
    d_bt = d.loc[df_bt.index]
    wr_bt = {p: wr.loc[df_bt.index] for p, wr in wr_dict.items()}

    for i in range(1, len(df_bt)):
        stoch_trend = analyze_stoch_rsi_trend(k_bt, d_bt, i)
        wr_trends = [analyze_wr_trend(wr_bt[p], i) for p in WR_PERIODS]
        buy_signal = any("Oversold" in t for t in wr_trends)
        sell_signal = any("Overbought" in t for t in wr_trends)

        if position == 0 and (stoch_trend == "Uptrend" or buy_signal):
            position = 1
            entry_price = df_bt['close'].iloc[i]
            entry_date = df_bt.index[i]

        elif position == 1 and (stoch_trend == "Downtrend" or sell_signal):
            exit_price = df_bt['close'].iloc[i]
            exit_date = df_bt.index[i]
            ret = (exit_price - entry_price) / entry_price
            trades.append((entry_date.strftime('%Y-%m-%d %H:%M'), exit_date.strftime('%Y-%m-%d %H:%M'), ret))
            position = 0
            entry_price = 0.0

    if position == 1:
        exit_price = df_bt['close'].iloc[-1]
        exit_date = df_bt.index[-1]
        ret = (exit_price - entry_price) / entry_price
        trades.append((entry_date.strftime('%Y-%m-%d %H:%M'), exit_date.strftime('%Y-%m-%d %H:%M'), ret))

    cumulative_return = np.prod([1 + t[2] for t in trades]) - 1 if trades else 0
    net_profit_loss = sum(t[2] for t in trades) if trades else 0

    return cumulative_return, net_profit_loss, trades

# --- MAIN LOGIC ---

def main():
    dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    results = {}

    for symbol in COINS:
        try:
            print(f"Fetching data for {symbol}...")
            df = fetch_ohlcv_ccxt(symbol, INTERVAL, LOOKBACK)
            if len(df) < LOOKBACK:
                print(f"Not enough data for {symbol} (got {len(df)} candles), skipping.")
                continue

            cum_ret, net_pl, trades = backtest(df)
            results[symbol] = (cum_ret, net_pl, trades)
            print(f"Backtest complete for {symbol}: Cumulative Return: {cum_ret*100:.2f}%, Net P/L: {net_pl*100:.2f}%")

        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            traceback.print_exc()

    if results:
        msg_lines = [f"<b>Kucoin {INTERVAL.upper()} Stochastic RSI + WR 5-Day Backtest ({dt})</b>",
                     "Cumulative returns, net profit/loss, and trades:\n"]
        for coin, (cum_ret, net_pl, trades) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
            msg_lines.append(f"{coin}: Cumulative Return: {cum_ret*100:.2f}%, Net Profit/Loss: {net_pl*100:.2f}%")
            if trades:
                for entry_date, exit_date, trade_ret in trades:
                    msg_lines.append(f"  Entry: {entry_date}  Exit: {exit_date}  Return: {trade_ret*100:.2f}%")
            else:
                msg_lines.append("  No trades executed.")
            msg_lines.append("")

        msg = "\n".join(msg_lines)
        send_telegram_message(msg)
    else:
        send_telegram_message("No backtest results available for the selected coins.")

if __name__ == "__main__":
    main()

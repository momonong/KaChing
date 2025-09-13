# In scripts/preprocess.py

import pandas as pd
from FinMind.data import DataLoader
import pandas_ta as ta
import numpy as np

# 1. 定義我們的股票池 (以幾家台灣大型電子股為例)
#    yfinance 的 Ticker 需要加上 .TW
stock_universe = ["2330", "2454", "2317", "2308"] # 台積電, 聯發科, 鴻海, 台達電
print(f"[INFO] Stock Universe: {stock_universe}")

all_stock_data = []
dl = DataLoader()

# 2. 迴圈下載所有股票數據
for stock_id in stock_universe:
    print(f"[INFO] Downloading data for {stock_id}...")
    stock_data = dl.taiwan_stock_daily(stock_id=stock_id, start_date="2020-01-01")
    all_stock_data.append(stock_data)

# 3. 將所有數據合併成一個大的 DataFrame
print("[INFO] Concatenating all stock data...")
data = pd.concat(all_stock_data, ignore_index=True)
data.set_index(['stock_id', 'date'], inplace=True)

# --- 以下是關鍵改造點 ---

# 4. 分組計算技術指標
#    我們必須確保每支股票的技術指標是獨立計算的，避免鴻海的數據去計算台積電的均線
print("[INFO] Calculating features for each stock...")
data.rename(columns={'max': 'High', 'min': 'Low', 'open': 'Open', 'close': 'Close', 'Trading_Volume': 'Volume'}, inplace=True)
# 使用 groupby('stock_id')
data['SMA_10'] = data.groupby('stock_id')['Close'].transform(lambda x: ta.sma(x, length=10))
data['SMA_20'] = data.groupby('stock_id')['Close'].transform(lambda x: ta.sma(x, length=20))
data['RSI_14'] = data.groupby('stock_id')['Close'].transform(lambda x: ta.rsi(x, length=14))
# pandas-ta 的 macd 比較特別，需要這樣處理
macd_df = data.groupby('stock_id', group_keys=False).apply(lambda x: x.ta.macd(append=True))
data = data.join(macd_df[['MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']])

# 5. 分組生成標籤
print("[INFO] Generating labels for each stock...")
future_days = 5
price_change_threshold = 0.02
data['future_change'] = data.groupby('stock_id')['Close'].transform(lambda x: x.shift(-future_days) / x - 1)
# ... (生成 target 的 np.select 程式碼不變) ...
conditions = [data['future_change'] > price_change_threshold, data['future_change'] < -price_change_threshold]
choices = [1, -1]
data['target'] = np.select(conditions, choices, default=0)

# 6. 清理數據並儲存
final_data = data.dropna()
print("\n[INFO] Saving final processed multi-stock data...")
final_data.to_csv('data/final_data_multi.csv')
print("[SUCCESS] Multi-stock data saved.")
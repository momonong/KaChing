# In scripts/preprocess.py

import pandas as pd
from FinMind.data import DataLoader
import pandas_ta as ta
import numpy as np

# 1. 定義股票池
stock_universe = ["2330", "2454", "2317", "2308"]
print(f"[INFO] Stock Universe: {stock_universe}")

dl = DataLoader()

# 2. 【新增】下載台灣加權指數(TAIEX)數據
print("[INFO] Downloading TAIEX data for market context...")
taiex_data = dl.taiwan_stock_daily(stock_id="TAIEX", start_date="2020-01-01")
# 計算大盤每日報酬率，並重新命名，以便合併
taiex_data['TAIEX_return'] = taiex_data['close'].pct_change()
taiex_data.reset_index(inplace=True)
taiex_data['date'] = pd.to_datetime(taiex_data['date'])
taiex_data = taiex_data[['date', 'TAIEX_return']]



# 3. 迴圈下載所有股票數據並合併
all_stock_data = []
for stock_id in stock_universe:
    print(f"[INFO] Downloading data for {stock_id}...")
    stock_data = dl.taiwan_stock_daily(stock_id=stock_id, start_date="2020-01-01")
    all_stock_data.append(stock_data)
data = pd.concat(all_stock_data, ignore_index=True)
data['date'] = pd.to_datetime(data['date'])

# 4. 【新增】將大盤數據合併到我們的主數據中
print("[INFO] Merging TAIEX data with stock data...")
data = pd.merge(data, taiex_data, on='date', how='left')
data.set_index('date', inplace=True)


# ---【新的步驟 5】: 標準化欄位名稱並計算特徵 ---
print("[INFO] Standardizing column names and calculating features...")

# 【修改點】統一將欄位名改為 pandas-ta 偏好的全小寫格式
data.rename(columns={
    'max': 'high',
    'min': 'low',
    'open': 'open', # 保持小寫
    'close': 'close', # 保持小寫
    'Trading_Volume': 'volume'
}, inplace=True)

processed_groups = []
for stock_id, group in data.groupby('stock_id'):
    group = group.copy()
    
    group.ta.sma(length=10, append=True)
    group.ta.sma(length=20, append=True)
    group.ta.rsi(length=14, append=True)
    group.ta.macd(append=True)
    group.ta.bbands(length=20, append=True)
    group.ta.obv(append=True)
    
    # --- 【新增除錯程式碼】 ---
    print(f"\n--- Debugging columns for stock_id: {stock_id} ---")
    print(list(group.columns))
    # --- 結束除錯程式碼 ---

    processed_groups.append(group)

print("[INFO] Combining processed groups...")
data_with_features = pd.concat(processed_groups)


# ---【新的步驟 6】: 分組生成標籤 ---
print("[INFO] Generating labels for each stock...")
future_days = 5
price_change_threshold = 0.02

# 【修改點】這裡也使用小寫的 'close'
data_with_features['future_change'] = data_with_features.groupby('stock_id')['close'].transform(lambda x: x.shift(-future_days) / x - 1)
conditions = [
    data_with_features['future_change'] > price_change_threshold,
    data_with_features['future_change'] < -price_change_threshold
]
choices = [1, -1]
data_with_features['target'] = np.select(conditions, choices, default=0)


# ---【新的步驟 7】: 清理數據並儲存 ---
final_data = data_with_features.dropna()
print("\n[INFO] Saving final processed multi-stock data with new features...")
final_data.to_csv('data/final_data_multi_enhanced.csv')
print("[SUCCESS] Enhanced multi-stock data saved.")
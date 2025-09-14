import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scripts.model import LSTMModel

# --- 步驟 1: 複製與訓練時完全相同的定義 ---

def create_sequences_grouped(data, features_cols, labels_col, seq_length):
    xs, ys, info = [], [], []
    for stock_id, group in data.groupby("stock_id"):
        features = group[features_cols].values
        labels = group[labels_col].values
        dates = group.index
        stock_ids = group["stock_id"].values
        # 【修正點 1】: 使用標準化後的小寫 'close'
        close_prices = group["close"].values

        for i in range(len(features) - seq_length):
            x = features[i : (i + seq_length)]
            y = labels[i + seq_length]
            xs.append(x)
            ys.append(y)
            info.append(
                {
                    "date": dates[i + seq_length],
                    "stock_id": stock_ids[i + seq_length],
                    "price": close_prices[i + seq_length],
                }
            )
    return np.array(xs), np.array(ys), pd.DataFrame(info)

# --- 步驟 2: 載入數據和模型 ---
print("[INFO] Setting up environment and loading data...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 【修正點 3】: 讀取帶有增強特徵的數據檔案
data = pd.read_csv("data/final_data_multi_enhanced.csv", index_col="date", parse_dates=True)
# --- 【修改點】: 根據特徵重要性分析，篩選出 Top 10 特徵 ---
features_cols = [
    'BBU_20_2.0_2.0', 'BBL_20_2.0_2.0', # Top 2: 波動率
    'SMA_20', 'SMA_10',                 # Top 4: 趨勢
    'OBV',                              # Top 5: 價量
    'MACDs_12_26_9',                    # MACD 訊號線
    'BBB_20_2.0_2.0',                   # 布林帶寬度
    'MACD_12_26_9',                     # MACD 主線
    'MACDh_12_26_9',                    # MACD 柱狀圖
    'RSI_14',                           # 相對強弱指標
]
target_col = "target"

scaler = StandardScaler()
data[features_cols] = scaler.fit_transform(data[features_cols])

sequence_length = 10
X, y, info_df = create_sequences_grouped(
    data, features_cols, target_col, sequence_length
)
y = y + 1

# 採用與訓練時完全一致的三向時序分割，以獲取正確的測試集
# --- 【修正點 1】: 採用嚴格的時序分割 ---
print("[INFO] Splitting data chronologically...")
train_split_point = int(len(X) * 0.70)
val_split_point = int(len(X) * 0.85)

# 在回測腳本中，我們只關心測試集的數據
X_test = X[val_split_point:]
y_test = y[val_split_point:]
info_test = info_df.iloc[val_split_point:]

print(f"  - Test set size: {len(X_test)}")
train_last_date = info_df.iloc[val_split_point - 1]['date']
test_first_date = info_test.iloc[0]['date']
print(f"  - Data before {train_last_date.date()} is used for training/validation.")
print(f"  - Backtest starts from {test_first_date.date()}.")

# 載入模型
print("[INFO] Loading pre-trained LSTM model...")
# --- 【修正點 2】: 使用與我們最新訓練的模型完全一致的超參數 ---
# 這是我們 train_lstm_advanced.py 中使用的參數
model = LSTMModel(input_size=len(features_cols), hidden_size=128, num_layers=3, num_classes=3, dropout_rate=0.3)
# 載入我們用專業流程訓練出的最佳模型
# 注意：這裡假設您要回測的是 advanced 版本的模型
model.load_state_dict(torch.load("model/best_lstm_model_pro.pth"))
model.to(device)
model.eval()


# --- 步驟 3: 獲取模型預測 ---
print("[INFO] Getting model predictions on the test set...")
X_test_tensor = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)

info_test = info_test.copy()
info_test["prediction"] = predicted.cpu().numpy() - 1

# --- 步驟 4: 執行回測模擬 (專業重構版) ---
print("[INFO] Starting backtest simulation...")

# 【新】建立一個方便查詢的「訊號字典」，格式：{(日期, 股票ID): 預測訊號}
signals_dict = {
    (row['date'], row['stock_id']): row['prediction'] 
    for _, row in info_test.iterrows()
}

# 【新】建立一個方便查詢的「每日價格表」
price_pivot = data.pivot_table(index='date', columns='stock_id', values='close')

# 初始化投資組合
initial_cash = 1000000
cash = initial_cash
holdings = {stock_id: 0 for stock_id in data['stock_id'].unique()}
portfolio_values = []

# 按日期推進模擬
test_dates = sorted(info_test["date"].unique())
for date in test_dates:
    # --- 1. 計算當日開盤時的資產 ---
    current_portfolio_value = cash
    for stock_id, shares in holdings.items():
        # 從價格表中獲取當日價格，如果當天沒開盤(NaN)，則沿用前一天的價格
        price = price_pivot.loc[date, stock_id]
        if pd.isna(price): # 如果當天價格缺失，找到最近的一個有效價格
            last_valid_price = price_pivot[stock_id].loc[:date].last_valid_index()
            if last_valid_price is not None:
                price = price_pivot.loc[last_valid_price, stock_id]
        
        if pd.notna(price):
            current_portfolio_value += shares * price
    portfolio_values.append(current_portfolio_value)

    # --- 2. 執行交易 ---
    stocks_with_signals_today = [key[1] for key in signals_dict if key[0] == date]
    
    # 賣出操作
    for stock_id in stocks_with_signals_today:
        signal = signals_dict.get((date, stock_id))
        if signal == -1 and holdings.get(stock_id, 0) > 0:
            price = price_pivot.loc[date, stock_id]
            if pd.notna(price):
                shares_to_sell = holdings[stock_id]
                cash += shares_to_sell * price
                holdings[stock_id] = 0

    # 買入操作 (平均分配現金)
    buy_signals_stocks = [s_id for s_id in stocks_with_signals_today if signals_dict.get((date, s_id)) == 1]
    if buy_signals_stocks and cash > 0:
        cash_per_buy = cash / len(buy_signals_stocks)
        for stock_id in buy_signals_stocks:
            if holdings.get(stock_id, 0) == 0: # 尚未持有才買入
                price = price_pivot.loc[date, stock_id]
                if pd.notna(price) and price > 0:
                    shares_to_buy = cash_per_buy // price
                    if shares_to_buy > 0:
                        holdings[stock_id] = shares_to_buy
                        cash -= shares_to_buy * price

# --- 5. 計算最終資產價值 (Bug 修正) ---
# 使用最後一天的價格，計算最終資產
final_cash = cash
final_holdings_value = 0
last_date = test_dates[-1]
for stock_id, shares in holdings.items():
    price = price_pivot.loc[last_date, stock_id]
    if pd.isna(price):
        last_valid_price = price_pivot[stock_id].loc[:last_date].last_valid_index()
        if last_valid_price is not None:
            price = price_pivot.loc[last_valid_price, stock_id]
    
    if pd.notna(price):
        final_holdings_value += shares * price
final_model_value = final_cash + final_holdings_value

# --- 步驟 5: 基準策略回測 (Buy and Hold) ---
print("[INFO] Calculating Buy and Hold benchmark...")
# 【修正點 6】: 使用小寫的 'close'
original_data = pd.read_csv("data/final_data_multi_enhanced.csv", index_col="date", parse_dates=True)
stock_universe = original_data["stock_id"].unique()
cash_per_stock = initial_cash / len(stock_universe)
benchmark_holdings = {}

first_day_prices = original_data.loc[test_dates[0]].set_index("stock_id")["close"]
for stock_id in stock_universe:
    price = first_day_prices.get(stock_id)
    if price and price > 0:
        benchmark_holdings[stock_id] = cash_per_stock // price

last_day_prices = original_data.loc[test_dates[-1]].set_index("stock_id")["close"]
final_benchmark_value = 0
for stock_id, shares in benchmark_holdings.items():
    price = last_day_prices.get(stock_id)
    if price:
        final_benchmark_value += shares * price

# --- 步驟 6: 輸出最終結果 ---
# ... (與您版本相同的輸出程式碼) ...
print("\n--- Backtest Results ---")
print(f"Test Period: {test_dates[0].date()} to {test_dates[-1].date()}")
model_return = (final_model_value / initial_cash - 1) * 100
benchmark_return = (final_benchmark_value / initial_cash - 1) * 100
print(f"\n[AI Model Strategy]")
print(f"  - Initial Portfolio Value: {initial_cash:,.2f}")
print(f"  - Final Portfolio Value:   {final_model_value:,.2f}")
print(f"  - Total Return:            {model_return:.2f}%")
print(f"\n[Benchmark: Buy and Hold]")
print(f"  - Initial Portfolio Value: {initial_cash:,.2f}")
print(f"  - Final Portfolio Value:   {final_benchmark_value:,.2f}")
print(f"  - Total Return:            {benchmark_return:.2f}%")
print("\n--- CONCLUSION ---")
if model_return > benchmark_return:
    print("Congratulations! The AI model strategy successfully outperformed the Buy and Hold benchmark.")
else:
    print("The AI model strategy did not outperform the Buy and Hold benchmark. Further tuning is needed.")
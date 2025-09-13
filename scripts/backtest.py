# In scripts/backtest.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 步驟 1: 複製必要的定義 ---
# 我們需要和 train_lstm.py 完全相同的模型架構和數據處理函數，才能正確地載入模型和數據

def create_sequences_grouped(data, features_cols, labels_col, seq_length):
    xs, ys, info = [], [], []
    for stock_id, group in data.groupby('stock_id'):
        features = group[features_cols].values
        labels = group[labels_col].values
        dates = group.index
        stock_ids = group['stock_id'].values
        close_prices = group['Close'].values

        for i in range(len(features) - seq_length):
            x = features[i:(i + seq_length)]
            y = labels[i + seq_length]
            xs.append(x)
            ys.append(y)
            # 我們額外保存日期、股價和ID，方便回測時對應
            info.append({
                'date': dates[i + seq_length],
                'stock_id': stock_ids[i + seq_length],
                'price': close_prices[i + seq_length]
            })
    return np.array(xs), np.array(ys), pd.DataFrame(info)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# --- 步驟 2: 載入數據和模型 ---
print("[INFO] Setting up environment and loading data...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv('data/final_data_multi.csv', index_col='date', parse_dates=True)
features_cols = ['SMA_10', 'SMA_20', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
target_col = 'target'

# 特徵縮放 (注意：這裡我們用全部數據來 fit_transform，在真實部署時應只用訓練集fit)
scaler = StandardScaler()
data[features_cols] = scaler.fit_transform(data[features_cols])

# 創建序列數據，這次我們需要 info DataFrame
sequence_length = 10
X, y, info_df = create_sequences_grouped(data, features_cols, target_col, sequence_length)
y = y + 1

# 進行與訓練時完全相同的數據切分，以確保測試集一致
# 【新方法 - 時序分割】
# 為了確保回測的公平性，我們嚴格按照時間順序分割數據。
# 我們用前 80% 的數據進行訓練 (在 train_lstm.py 中)，用後 20% 的數據進行回測。
print("[INFO] Splitting data chronologically to get the test set...")
split_ratio = 0.8
split_point = int(len(X) * split_ratio)

# 我們在回測腳本中，只關心分割點之後的「測試集」
X_test = X[split_point:]
y_test = y[split_point:]
# 對於 DataFrame，我們使用 .iloc 來進行基於位置的切片
info_test = info_df.iloc[split_point:]

print(f"  - Total sequences: {len(X)}")
print(f"  - Training sequences (for reference): {split_point}")
print(f"  - Test sequences: {len(X_test)}")

# 檢查時間是否連續
train_last_date = info_df.iloc[split_point - 1]['date']
test_first_date = info_test.iloc[0]['date']
print(f"  - Last training date: {train_last_date.date()}")
print(f"  - First testing date: {test_first_date.date()}")

# 載入模型
print("[INFO] Loading pre-trained LSTM model...")
input_size = len(features_cols)
hidden_size = 50
num_layers = 2
num_classes = 3
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load('model/lstm_multi_stock.pth'))
model.to(device)
model.eval()

# --- 步驟 3: 獲取模型預測 ---
print("[INFO] Getting model predictions on the test set...")
X_test_tensor = torch.FloatTensor(X_test).to(device)
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)

# 將預測結果(-1, 0, 1)合併回 info_test DataFrame
info_test = info_test.copy()
info_test['prediction'] = predicted.cpu().numpy() - 1

# --- 步驟 4: 執行回測模擬 ---
print("[INFO] Starting backtest simulation...")

initial_cash = 1000000
cash = initial_cash
holdings = {} # 用來記錄持有每支股票的數量
portfolio_values = []

# 按日期排序，模擬時間推進
test_dates = sorted(info_test['date'].unique())

for date in test_dates:
    # 當日資產價值 = 現金 + 持有股票總市值
    current_portfolio_value = cash
    for stock_id, shares in holdings.items():
        # 用當日收盤價計算股票市值 (如果當天沒有該股票數據，則用前一天的)
        price = info_test[(info_test['date'] == date) & (info_test['stock_id'] == stock_id)]['price'].values
        if len(price) > 0:
            current_portfolio_value += shares * price[0]
    portfolio_values.append(current_portfolio_value)

    # 獲取當天的所有交易訊號
    signals_today = info_test[info_test['date'] == date]
    
    # 賣出操作
    for _, row in signals_today.iterrows():
        if row['prediction'] == -1 and row['stock_id'] in holdings:
            shares_to_sell = holdings.pop(row['stock_id']) # 賣出全部
            cash += shares_to_sell * row['price']
            # print(f"{date.date()}: SELL {row['stock_id']} at {row['price']:.2f}")

    # 買入操作 (簡單起見，平均分配當前現金給所有買入訊號)
    buy_signals = signals_today[signals_today['prediction'] == 1]
    if not buy_signals.empty and cash > 0:
        cash_per_buy = cash / len(buy_signals)
        for _, row in buy_signals.iterrows():
            if row['stock_id'] not in holdings: # 如果尚未持有，才買入
                if row['price'] > 0:
                    shares_to_buy = cash_per_buy // row['price']
                if shares_to_buy > 0:
                    holdings[row['stock_id']] = shares_to_buy
                    cash -= shares_to_buy * row['price']
                    # print(f"{date.date()}: BUY {row['stock_id']} at {row['price']:.2f}")

final_model_value = portfolio_values[-1]

# --- 步驟 5: 基準策略回測 (Buy and Hold) ---
print("[INFO] Calculating Buy and Hold benchmark...")
benchmark_cash = initial_cash
stock_universe = data['stock_id'].unique()
cash_per_stock = benchmark_cash / len(stock_universe)
benchmark_holdings = {}

first_day_prices = data.loc[test_dates[0]].set_index('stock_id')['Close']
for stock_id in stock_universe:
    price = first_day_prices.get(stock_id)
    if price:
        shares_to_buy = cash_per_stock // price
        benchmark_holdings[stock_id] = shares_to_buy

last_day_prices = data.loc[test_dates[-1]].set_index('stock_id')['Close']
final_benchmark_value = 0
for stock_id, shares in benchmark_holdings.items():
     price = last_day_prices.get(stock_id)
     if price:
        final_benchmark_value += shares * price

# --- 步驟 6: 輸出最終結果 ---
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
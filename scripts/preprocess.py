from FinMind.data import DataLoader
import pandas_ta as ta
import numpy as np

# 初始化 DataLoader
dl = DataLoader()

# 下載台積電股價資料 (股票代碼, 開始日期, 結束日期)
# FinMind 會自動處理 Open, High, Low, Close, Volume 等欄位
stock_data = dl.taiwan_stock_daily(
    stock_id="2330",
    start_date="2020-01-01",
    end_date="2025-09-12"
)

# FinMind 的欄位名稱與 yfinance 稍有不同，我們將其統一
# 例如：'Trading_Volume' -> 'Volume', 'open' -> 'Open'
stock_data.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)

# 將 'date' 欄位設為索引 (index)，這跟 yfinance 的格式一樣
stock_data.set_index('date', inplace=True)


print(stock_data.head())


print(stock_data.info())
print("\n[INFO] Starting feature engineering...")

# 1. 計算各種技術指標
#    pandas-ta 會很方便地直接幫我們在 DataFrame 上新增欄位
stock_data.ta.sma(length=10, append=True)  # 10日均線 (SMA_10)
stock_data.ta.sma(length=20, append=True)  # 20日均線 (SMA_20)
stock_data.ta.rsi(length=14, append=True)  # 14日相對強弱指數 (RSI_14)
stock_data.ta.macd(append=True)            # MACD 指標 (MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)

# 2. 清理因計算指標而在前期產生的空值 (NaN)
stock_data.dropna(inplace=True)

# 3. 顯示加入特徵後的最後幾筆資料，檢查結果
print("[SUCCESS] Features added successfully.")
print("Data with new features (tail):")
print(stock_data.tail())

print("\n[INFO] Starting label generation...")

# 1. 設定參數
future_days = 5  # 我們看未來 5 天的價格變化
price_change_threshold = 0.02  # 漲跌幅超過 2% 才算一個訊號

# 2. 計算 N 天後的價格變化率
#    .shift(-future_days) 的意思是把 N 天後的價格往前挪到今天這一列，方便計算
stock_data['future_change'] = stock_data['close'].shift(-future_days) / stock_data['close'] - 1

# 3. 根據變化率生成標籤 (1: Buy, -1: Sell, 0: Hold)
conditions = [
    stock_data['future_change'] > price_change_threshold,
    stock_data['future_change'] < -price_change_threshold
]
choices = [1, -1]  # 1 對應買進, -1 對應賣出
stock_data['target'] = np.select(conditions, choices, default=0) # 其餘是 0 (持平)

# 4. 移除因為無法計算未來價格而產生空值的最後幾筆資料
final_data = stock_data.dropna()

# 5. 顯示最終處理完的資料，並檢查標籤分佈
print("[SUCCESS] Labels generated successfully.")
print("Final data with features and labels (tail):")
print(final_data.tail())

print("\nLabel distribution:")
print(final_data['target'].value_counts())


print("\n[INFO] Starting backtesting of the labeling strategy...")

# --- 1. 策略回測 ---
initial_cash = 1000000  # 初始資金一百萬
cash = initial_cash
shares = 0
portfolio_value = []

for index, row in final_data.iterrows():
    current_price = row['close']
    
    # 買進訊號 (target=1)，且我們有足夠現金
    if row['target'] == 1 and cash > current_price:
        shares_to_buy = cash // current_price
        shares += shares_to_buy
        cash -= shares_to_buy * current_price
        
    # 賣出訊號 (target=-1)，且我們有股票可賣
    elif row['target'] == -1 and shares > 0:
        cash += shares * current_price
        shares = 0
        
    # 計算當日的總資產價值
    current_portfolio_value = cash + shares * current_price
    portfolio_value.append(current_portfolio_value)

# 最後一天的總資產
final_strategy_value = portfolio_value[-1]
print(f"\n[RESULT] Perfect Strategy:")
print(f"  - Initial Portfolio Value: {initial_cash:,.2f}")
print(f"  - Final Portfolio Value:   {final_strategy_value:,.2f}")


# --- 2. 單純持有策略 (Buy and Hold) 作為對照組 ---
initial_shares = initial_cash // final_data['close'].iloc[0]
remaining_cash = initial_cash - initial_shares * final_data['close'].iloc[0]
final_buy_and_hold_value = remaining_cash + initial_shares * final_data['close'].iloc[-1]

print(f"\n[BENCHMARK] Buy and Hold Strategy:")
print(f"  - Initial Portfolio Value: {initial_cash:,.2f}")
print(f"  - Final Portfolio Value:   {final_buy_and_hold_value:,.2f}")


# --- 3. 績效比較 ---
strategy_return = (final_strategy_value / initial_cash - 1) * 100
buy_and_hold_return = (final_buy_and_hold_value / initial_cash - 1) * 100
print(f"\n[PERFORMANCE] Comparison:")
print(f"  - Perfect Strategy Return: {strategy_return:.2f}%")
print(f"  - Buy and Hold Return:     {buy_and_hold_return:.2f}%")

if strategy_return > buy_and_hold_return:
    print("\n[CONCLUSION] The labeling strategy is potentially profitable and outperforms Buy & Hold.")
else:
    print("\n[CONCLUSION] The labeling strategy does NOT outperform Buy & Hold. Consider adjusting parameters.")

import os

if not os.path.exists('data'):
    os.makedirs('data')

print("\n[INFO] Saving final processed data to data/final_data.csv...")
final_data.to_csv('data/final_data.csv')
print("[SUCCESS] Data saved.")
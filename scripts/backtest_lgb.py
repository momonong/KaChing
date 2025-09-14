# In scripts/backtest_xgb.py

import pandas as pd
import joblib

# --- 主要執行流程 ---
if __name__ == '__main__':
    print("[INFO] Loading data for backtest...")
    data = pd.read_csv('data/final_data_multi_enhanced.csv', index_col='date', parse_dates=True)
    features_cols = [
        'BBU_20_2.0_2.0', 'BBL_20_2.0_2.0', 'SMA_20', 'SMA_10', 'OBV',
        'MACDs_12_26_9', 'BBB_20_2.0_2.0', 'MACD_12_26_9', 'MACDh_12_26_9', 'RSI_14'
    ]
    target_col = 'target'
    X = data[features_cols]

    # 獲取與訓練時完全一致的測試集
    split_point = int(len(X) * 0.7)
    X_test = X.iloc[split_point:]
    
    # 載入模型
    print("[INFO] Loading pre-trained LGBoost model...")
    model = joblib.load('model/lgb_model_final.joblib')

    # 獲取模型預測
    print("[INFO] Getting model predictions...")
    y_pred = model.predict(X_test)

    # 準備回測數據
    info_test = data.iloc[split_point:][['stock_id', 'close']].copy()
    info_test['prediction'] = y_pred - 1
    info_test.rename(columns={'close': 'price'}, inplace=True)
    info_test.reset_index(inplace=True)
    
    # 執行回測 (與 evaluate_lgb.py 腳本中的回測邏輯完全相同)
    print("[INFO] Starting backtest simulation...")
    # ... (此處省略，請將 evaluate_lgb.py 中完整的、精確的回測與基準測試程式碼複製到這裡) ...
    # 從 initial_cash = 1000000 開始，一直到最後的 print CONCLUSION

    initial_cash = 1000000
price_pivot = data.pivot_table(index='date', columns='stock_id', values='close')
signals_dict = { (row['date'], row['stock_id']): row['prediction'] for _, row in info_test.iterrows() }
cash = initial_cash
holdings = {stock_id: 0 for stock_id in data['stock_id'].unique()}
portfolio_values = []
test_dates = sorted(info_test["date"].unique())

for date in test_dates:
    current_portfolio_value = cash
    for stock_id, shares in holdings.items():
        price = price_pivot.loc[date, stock_id]
        if pd.isna(price):
            last_valid_price = price_pivot[stock_id].loc[:date].last_valid_index()
            if last_valid_price is not None: price = price_pivot.loc[last_valid_price, stock_id]
        if pd.notna(price): current_portfolio_value += shares * price
    portfolio_values.append(current_portfolio_value)
    stocks_with_signals_today = [key[1] for key in signals_dict if key[0] == date]
    for stock_id in stocks_with_signals_today:
        signal = signals_dict.get((date, stock_id))
        if signal == -1 and holdings.get(stock_id, 0) > 0:
            price = price_pivot.loc[date, stock_id]
            if pd.notna(price):
                shares_to_sell = holdings[stock_id]
                cash += shares_to_sell * price
                holdings[stock_id] = 0
    buy_signals_stocks = [s_id for s_id in stocks_with_signals_today if signals_dict.get((date, s_id)) == 1]
    if buy_signals_stocks and cash > 0:
        cash_per_buy = cash / len(buy_signals_stocks)
        for stock_id in buy_signals_stocks:
            if holdings.get(stock_id, 0) == 0:
                price = price_pivot.loc[date, stock_id]
                if pd.notna(price) and price > 0:
                    shares_to_buy = cash_per_buy // price
                    if shares_to_buy > 0:
                        holdings[stock_id] = shares_to_buy
                        cash -= shares_to_buy * price

final_cash = cash
final_holdings_value = 0
last_date = test_dates[-1]
for stock_id, shares in holdings.items():
    price = price_pivot.loc[last_date, stock_id]
    if pd.isna(price):
        last_valid_price = price_pivot[stock_id].loc[:last_date].last_valid_index()
        if last_valid_price is not None: price = price_pivot.loc[last_valid_price, stock_id]
    if pd.notna(price): final_holdings_value += shares * price
final_model_value = final_cash + final_holdings_value

# 基準策略回測 (Buy and Hold)
# --- 基準策略回測 (Buy and Hold) (精確模擬版) ---
print("[INFO] Calculating Buy and Hold benchmark with precise simulation...")

# 選取測試期間的原始數據
test_period_data = data.iloc[split_point:]
stock_universe = test_period_data['stock_id'].unique()
cash_per_stock = initial_cash / len(stock_universe)
benchmark_holdings = {}

# 獲取測試期第一天的日期和價格
first_day_of_test = test_dates[0]
first_day_prices = test_period_data.loc[first_day_of_test]
# 處理當天只有一筆數據的特殊情況
if isinstance(first_day_prices, pd.Series):
    first_day_prices = first_day_prices.to_frame().T
first_day_prices = first_day_prices.set_index('stock_id')

# 模擬建倉
remaining_cash = 0
for stock_id in stock_universe:
    if stock_id in first_day_prices.index:
        price = first_day_prices.loc[stock_id, 'close']
        if price and price > 0:
            shares_to_buy = cash_per_stock // price
            benchmark_holdings[stock_id] = shares_to_buy
            cost = shares_to_buy * price
            remaining_cash += (cash_per_stock - cost)
    else:
        # 如果第一天該股票未交易，則資金保留
        remaining_cash += cash_per_stock

# 獲取測試期最後一天的價格並計算最終資產
last_day_of_test = test_dates[-1]
last_day_prices = test_period_data.loc[last_day_of_test]
if isinstance(last_day_prices, pd.Series):
    last_day_prices = last_day_prices.to_frame().T
last_day_prices = last_day_prices.set_index('stock_id')

final_holdings_value = 0
for stock_id, shares in benchmark_holdings.items():
    if stock_id in last_day_prices.index:
        price = last_day_prices.loc[stock_id, 'close']
        final_holdings_value += shares * price
    # 如果最後一天該股票未交易，則其持有價值視為零

final_benchmark_value = remaining_cash + final_holdings_value

# --- 步驟 6: 輸出最終結果 ---
print("\n--- LGBoost Final Evaluation ---")
model_return = (final_model_value / initial_cash - 1) * 100
benchmark_return = (final_benchmark_value / initial_cash - 1) * 100
print(f"\n[AI Model Strategy (LGBoost)]")
print(f"  - Final Portfolio Value:   {final_model_value:,.2f}")
print(f"  - Total Return:            {model_return:.2f}%")
print(f"\n[Benchmark: Buy and Hold]")
print(f"  - Final Portfolio Value:   {final_benchmark_value:,.2f}")
print(f"  - Total Return:            {benchmark_return:.2f}%")
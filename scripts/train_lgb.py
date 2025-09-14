# In scripts/train_xgb.py
import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report
from utils.random_seed import set_seed
import joblib # 引入 joblib 來保存模型


# --- 主要執行流程 ---
if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)
    print(f"[INFO] Random seed set to {SEED}")

    # 載入數據
    print("[INFO] Loading enhanced multi-stock data...")
    data = pd.read_csv('data/final_data_multi_enhanced.csv', index_col='date', parse_dates=True)
    features_cols = [
        'BBU_20_2.0_2.0', 'BBL_20_2.0_2.0', 'SMA_20', 'SMA_10', 'OBV',
        'MACDs_12_26_9', 'BBB_20_2.0_2.0', 'MACD_12_26_9', 'MACDh_12_26_9', 'RSI_14'
    ]
    target_col = 'target'
    X = data[features_cols]
    y = data[target_col]

    # 時序分割 (70% 訓練, 30% 測試)
    print("[INFO] Splitting data chronologically...")
    split_point = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

    # 訓練 XGBoost 模型
    print("[INFO] Training XGBoost model...")
    y_train_mapped = y_train + 1
    y_test_mapped = y_test + 1

    model = lgb.LGBMClassifier(random_state=SEED)
    model.fit(X_train, y_train_mapped)
    print("[SUCCESS] Model training completed.")

    # 評估並印出報告
    print("\n[INFO] Evaluating model classification performance...")
    y_pred = model.predict(X_test)
    print("\n  - Classification Report:")
    print(classification_report(y_test_mapped, y_pred, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))
    
    # 保存模型
    print("\n[INFO] Saving the trained LGBoost model...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, 'model/lgb_model_final.joblib')
    print("[SUCCESS] Model saved to model/lgb_model_final.joblib")
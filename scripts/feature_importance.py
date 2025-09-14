# In scripts/feature_importance.py

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from utils.random_seed import set_seed

SEED = 42
set_seed(SEED)
print(f"[INFO] Random seed set to {SEED}")

# --- 數據準備 ---
print("[INFO] Loading enhanced multi-stock data...")
data = pd.read_csv('data/final_data_multi_enhanced.csv', index_col='date', parse_dates=True)

features_cols = [
    'SMA_10', 'SMA_20', 'RSI_14', 
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'BBL_20_2.0_2.0', 'BBM_20_2.0_2.0', 'BBU_20_2.0_2.0', 'BBB_20_2.0_2.0', 'BBP_20_2.0_2.0',
    'OBV', 
    'TAIEX_return'
]
target_col = 'target'

X = data[features_cols]
y = data[target_col]

# 採用嚴謹的時序分割
print("[INFO] Splitting data chronologically...")
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# --- 模型訓練 ---
print("[INFO] Training XGBoost model to analyze feature importance...")
# XGBoost 的標籤不需要 +1，它可以直接處理 -1, 0, 1
# 但為了與 scikit-learn 兼容，先將標籤映射到 0, 1, 2
y_train_mapped = y_train + 1
y_test_mapped = y_test + 1

model = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=3, 
    eval_metric='mlogloss', 
    use_label_encoder=False, 
    random_state=SEED
)
model.fit(X_train, y_train_mapped)

print("[SUCCESS] Model training completed.")

# --- 特徵重要性分析與視覺化 ---
print("[INFO] Plotting feature importance...")

# 提取特徵重要性
importance = model.feature_importances_
feature_importance = pd.DataFrame({'feature': features_cols, 'importance': importance})
feature_importance = feature_importance.sort_values('importance', ascending=True)

# 繪製水平長條圖
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.title('Feature Importance Analysis (XGBoost)')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout() # 自動調整佈局

# 解決 matplotlib 中文亂碼問題 (如果需要)
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
# plt.rcParams['axes.unicode_minus'] = False

# 儲存並顯示圖表
plt.savefig('figures/feature_importance.png')
print("[SUCCESS] Feature importance chart saved to feature_importance.png")

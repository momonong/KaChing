import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- 機器學習基礎 1: 載入並定義特徵 (X) 與目標 (y) ---

# 讀取我們在 preprocess.py 中準備好的數據
print("[INFO] Loading data...")
data = pd.read_csv('data/final_data.csv', index_col='date', parse_dates=True)

# 定義我們要用來預測的特徵欄位
# 簡單起見，我們先用最經典的幾個指標
features = ['SMA_10', 'SMA_20', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
target = 'target'

# X 是我們的「問題」(特徵)，y 是我們的「答案」(目標)
X = data[features]
y = data[target]

print(f"[INFO] Features (X) shape: {X.shape}")
print(f"[INFO] Target (y) shape: {y.shape}")


# --- 機器學習基礎 2: 切分訓練集與測試集 ---

# 為什麼要切分？ 
# 我們需要一部分從未見過的數據（測試集）來評估模型的好壞，確保它不是只會「背答案」。
# test_size=0.2 代表我們保留 20% 的數據作為測試集。
# stratify=y 能確保在切分後，訓練集和測試集中的買/賣/持平訊號比例與原始數據一致，這對分類問題很重要。
# random_state=42 確保每次切分的結果都一樣，方便我們重現實驗。
print("\n[INFO] Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  - Training set size: {len(X_train)}")
print(f"  - Testing set size: {len(X_test)}")


# --- 機器學習基礎 3: 選擇模型並進行訓練 ---

# 我們選擇「隨機森林分類器 (RandomForestClassifier)」作為第一個模型。
# 為什麼？ 它是一個非常強大且通用的模型，由多個「決策樹」組成，不容易過擬合，效果通常很好。
# n_estimators=100 代表這個「森林」由 100 棵樹組成。
print("\n[INFO] Training the RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# .fit() 就是「訓練」的指令，模型會從 X_train 和 y_train 中學習規律
model.fit(X_train, y_train)

print("[SUCCESS] Model training completed.")


# --- 機器學習基礎 4: 在測試集上評估模型 ---

# 使用訓練好的模型，對「它沒見過的」測試集 X_test 進行預測
print("\n[INFO] Evaluating model on the test set...")
y_pred = model.predict(X_test)

# 計算並印出評估指標
accuracy = accuracy_score(y_test, y_pred)
print(f"\n  - Accuracy: {accuracy:.4f}")

# 分類報告提供了更詳細的指標：
# - Precision (精確率): 模型預測為「買進」的訊號中，有多少是真的「買進」。
# - Recall (召回率): 所有真實的「買進」訊號中，模型成功找出了多少。
# - F1-score: Precision 和 Recall 的綜合指標。
print("\n  - Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))
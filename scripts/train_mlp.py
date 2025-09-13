import pandas as pd
import numpy as np
from scipy.integrate._ivp.radau import predict_factor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight # <-- 新增這行

from scripts.train_ml import X_train, y_pred

# --- 深度學習基礎 1: 數據準備 (Tensors & Scaling) ---
print("[INFO] Loading and preparing data for PyTorch...")

# 讀取數據
data = pd.read_csv("data/final_data.csv", index_col="date", parse_dates=True)
features = [
    "SMA_10",
    "SMA_20",
    "RSI_14",
    "MACD_12_26_9",
    "MACDh_12_26_9",
    "MACDs_12_26_9",
]
target = "target"

X = data[features]
# 我們的 target 是 -1, 0, 1，PyTorch 的 CrossEntropyLoss 需要從 0 開始的標籤 (0, 1, 2)
# 所以我們把所有標籤 +1
y = data[target] + 1

# **關鍵步驟：特徵縮放 (Feature Scaling)**
# 神經網絡對數據的尺度很敏感。我們需要將所有特徵（如 RSI 在 0-100，SMA 在股價範圍）
# 縮放到一個相似的尺度（例如，平均值為0，標準差為1），這能幫助模型更快、更穩定地學習。
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 將數據轉換為 PyTorch Tensors
# Tensor 是 PyTorch 中的基本數據結構，類似於 NumPy 陣列
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test.values)

# --- 深度學習基礎 2: 定義神經網絡架構 ---
print("[INFO] Defining the MLP model architecture...")

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64) # 第一層：從輸入特徵到 64 個神經元
        self.layer_2 = nn.Linear(64, 32)         # 第二層：從 64 個神經元到 32 個
        self.output_layer = nn.Linear(32, num_classes) # 輸出層：從 32 個神經元到最終的分類數量

        # 定義「激活函數」，決定神經元的訊號是否要往下傳遞
        self.relu = nn.ReLU()
        # Dropout 是一種正則化技巧，訓練時隨機「關閉」一些神經元，防止模型過擬合
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x

# 根據我們的數據實例化模型
# input_size 是特徵的數量 (6), num_classes 是分類的數量 (3: Sell, Hold, Buy)
model = MLP(input_size=len(features), num_classes=3)

# --- 深度學習基礎 3: 定義損失函數與優化器 ---
print("[INFO] Defining loss function and optimizer...")

# 【修改點】計算類別權重來解決不平衡問題
# 我們只在訓練集上計算權重，以避免數據洩漏
class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
# 將 NumPy 權重陣列轉換為 PyTorch Tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

print(f"  - Calculated Class Weights: {class_weights_tensor.numpy()}")

# 損失函數 (Loss Function)：用來衡量模型預測的好壞，「錯得有多離譜」。
# CrossEntropyLoss 是分類問題中最常用的損失函數。
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# 優化器 (Optimizer)：根據損失函數的結果，來更新模型內部參數（權重）的演算法，「如何讓模型進步」。
# Adam 是一個非常強大且通用的優化器。
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# --- 深度學習基礎 4: 訓練模型 (The Training Loop) ---
print("[INFO] Starting model training...")
epochs = 500 # 我們讓模型完整地學習所有訓練數據 100 次

for epoch in range(epochs):
    model.train() # 將模型設置為訓練模式

    # 1. 前向傳播 (Forward Pass)
    outputs = model(X_train_tensor)

    # 2. 計算損失
    loss = criterion(outputs, y_train_tensor)

    # 3. 反向傳播與優化 (Backward Pass & Optimization)
    optimizer.zero_grad() # 清除上一輪的梯度
    loss.backward()       # 計算梯度
    optimizer.step()      # 更新權重

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    

print("[SUCCESS] Model training completed.")

# --- 深度學習基礎 5: 評估模型 ---
print("\n[INFO] Evaluating model on the test set...")
model.eval() # 將模型設置為評估模式

with torch.no_grad(): # 在評估時，我們不需要計算梯度
    # 得到模型在測試集上的原始輸出 (logits)
    outputs = model(X_test_tensor)
    # 找到每個樣本中，概率最高的那個類別作為預測結果
    _, predicted = torch.max(outputs.data, 1)

    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"\n  - Accuracy: {accuracy:.4f}")
    
    print("\n  - Classification Report:")
    # 將 PyTorch Tensors 轉回 NumPy 陣列，才能使用 scikit-learn 的工具
    y_pred_np = predicted.numpy()
    y_test_np = y_test_tensor.numpy()
    # 我們的標籤是 0, 1, 2，對應原本的 -1, 0, 1
    print(classification_report(y_test_np, y_pred_np, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))

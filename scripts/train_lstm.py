import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scripts.model import LSTMModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# --- LSTM 基礎 1: 創建時間序列數據集 ---
# 這是與 MLP 最大的不同之處。我們需要將數據從 (樣本數, 特徵數) 轉換為 (樣本數, 時間窗長度, 特徵數)
# --- 關鍵改造點：分組創建序列 ---
def create_sequences_grouped(data, features_cols, labels_col, seq_length):
    xs, ys = [], []
    # 使用 groupby 來確保序列不會跨越不同的股票
    for stock_id, group in data.groupby('stock_id'):
        features = group[features_cols].values
        labels = group[labels_col].values
        for i in range(len(features) - seq_length):
            x = features[i:(i + seq_length)]
            y = labels[i + seq_length]
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)

# --- 數據準備 ---
print("[INFO] Loading and preparing data for LSTM...")

data = pd.read_csv('data/final_data_multi_enhanced.csv', index_col='date', parse_dates=True)
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
target_col = 'target'

# 特徵縮放 (與 MLP 相同)
scaler = StandardScaler()
data[features_cols] = scaler.fit_transform(data[features_cols])

# 創建序列數據
sequence_length = 10  # 我們讓模型一次回看過去 10 天的數據
X, y = create_sequences_grouped(data, features_cols, 'target', sequence_length)

# 我們的 target 是 -1, 0, 1，PyTorch 需要 0, 1, 2
y = y + 1

# 切分數據集
# --- 【修改點 1】: 三向數據分割 (訓練集/驗證集/測試集) ---
print("[INFO] Splitting data into train, validation, and test sets...")

# 我們採用 70% 訓練, 15% 驗證, 15% 測試的時序分割
train_split_point = int(len(X) * 0.70)
val_split_point = int(len(X) * 0.85)

X_train, X_val, X_test = X[:train_split_point], X[train_split_point:val_split_point], X[val_split_point:]
y_train, y_val, y_test = y[:train_split_point], y[train_split_point:val_split_point], y[val_split_point:]

print(f"  - Training set size: {len(X_train)}")
print(f"  - Validation set size: {len(X_val)}")
print(f"  - Test set size: {len(X_test)}")

# 轉換為 PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# --- LSTM 基礎 2: 定義 LSTM 模型架構 ---
print("[INFO] Defining the LSTM model architecture...")


# 實例化模型
input_size = len(features_cols)  # 每個時間點的特徵數 (6)
hidden_size = 128                 # LSTM 記憶單元的大小 (可調超參數)
num_layers = 1                   # 堆疊 3 層 LSTM (可調超參數)
num_classes = 3                  # 輸出類別數
batch_size = 64
dropout_rate = 0.3

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 訓練集需要打亂

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = LSTMModel(input_size, hidden_size, num_layers, num_classes, dropout_rate)
model.to(device)

# --- 損失函數 (帶權重) 與優化器 ---
print("[INFO] Defining loss function with class weights and optimizer...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# --- 訓練模型 ---
# --- 【修改點 3】: 採用提早停止的訓練迴圈 ---
print("[INFO] Starting model training with Early Stopping...")
epochs = 500 # Mini-batch 訓練通常收斂更快
patience = 25
patience_counter = 0
best_val_loss = float('inf')
best_model_path = 'model/best_lstm_model_pro.pth'

for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    # --- 【修改點】: Mini-batch 訓練迴圈 ---
    for data, target in train_loader:
        outputs = model(data)
        loss = criterion(outputs, target)
        
        optimizer.zero_grad()
        loss.backward()
        # --- 【新引入】: 梯度裁剪 ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_train_loss = total_train_loss / len(train_loader)

    # 驗證步驟
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            val_outputs = model(data)
            val_loss = criterion(val_outputs, target)
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    
    # 更新學習率
    scheduler.step(avg_val_loss)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # 提早停止邏輯
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f'Early stopping triggered after {epoch + 1} epochs.')
        break

print("[SUCCESS] Model training completed.")

# --- 【修改點】: 採用與 DataLoader 兼容的評估流程 ---
print("\n[INFO] Loading best model and evaluating on the test set...")

# 載入在驗證集上表現最好的模型
# 確保模型架構定義與保存時一致
model = LSTMModel(input_size=len(features_cols), hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, dropout_rate=dropout_rate).to(device)
model.load_state_dict(torch.load(best_model_path))
model.eval()

# 儲存所有批次的預測和真實標籤
all_predictions = []
all_targets = []

with torch.no_grad():
    # 遍歷測試數據的每一個批次
    for data, target in test_loader:
        # 將預測結果和真實標籤加入列表
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.append(predicted)
        all_targets.append(target)

# 將列表中所有批次的 tensor 合併成一個大的 tensor
all_predictions = torch.cat(all_predictions)
all_targets = torch.cat(all_targets)

# 一次性計算最終的評估指標
accuracy = (all_predictions == all_targets).sum().item() / len(all_targets)
print(f"\n  - Accuracy: {accuracy:.4f}")

print("\n  - Classification Report:")
y_pred_np = all_predictions.cpu().numpy()
y_test_np = all_targets.cpu().numpy()
print(classification_report(y_test_np, y_pred_np, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))

# 腳本結尾處不再需要保存模型，因為最好的模型已經在訓練迴圈中保存了
print(f"\n[INFO] Best model was saved to {best_model_path} during training.")
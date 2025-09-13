import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

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

data = pd.read_csv('data/final_data_multi.csv', index_col='date', parse_dates=True)
features_cols = ['SMA_10', 'SMA_20', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 轉換為 PyTorch Tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)

# --- LSTM 基礎 2: 定義 LSTM 模型架構 ---
print("[INFO] Defining the LSTM model architecture...")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 層：它會處理時間序列數據
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # 全連接層：將 LSTM 最後的輸出，轉換為我們的 3 個分類
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初始化 LSTM 的隱藏狀態和細胞狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 將數據傳遞給 LSTM 層
        out, _ = self.lstm(x, (h0, c0))
        
        # 我們只取序列中「最後一個時間點」的輸出來做預測
        out = self.fc(out[:, -1, :])
        return out

# 實例化模型
input_size = len(features_cols)  # 每個時間點的特徵數 (6)
hidden_size = 50                 # LSTM 記憶單元的大小 (可調超參數)
num_layers = 2                   # 堆疊 2 層 LSTM (可調超參數)
num_classes = 3                  # 輸出類別數
model = LSTMModel(input_size, hidden_size, num_layers, num_classes)
model.to(device)

# --- 損失函數 (帶權重) 與優化器 ---
print("[INFO] Defining loss function with class weights and optimizer...")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- 訓練模型 ---
print("[INFO] Starting model training...")
epochs = 1800 # LSTM 通常收斂得更快，我們先用 200 次

for epoch in range(epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("[SUCCESS] Model training completed.")

# --- 評估模型 ---
print("\n[INFO] Evaluating model on the test set...")
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    
    accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
    print(f"\n  - Accuracy: {accuracy:.4f}")
    
    print("\n  - Classification Report:")
    y_pred_np = predicted.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()
    print(classification_report(y_test_np, y_pred_np, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))
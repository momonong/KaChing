import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import random
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

# --- 新增：設定隨機種子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 設定一個你喜歡的數字作為種子
SEED = 42
set_seed(SEED)
print(f"[INFO] Random seed set to {SEED}")

# --- 數據準備 (與 LSTM 腳本完全相同) ---
def create_sequences_grouped(data, features_cols, labels_col, seq_length):
    xs, ys = [], []
    for stock_id, group in data.groupby('stock_id'):
        features = group[features_cols].values
        labels = group[labels_col].values
        for i in range(len(features) - seq_length):
            x = features[i:(i + seq_length)]
            y = labels[i + seq_length]
            xs.append(x)
            ys.append(y)
    return np.array(xs), np.array(ys)

# --- Transformer 基礎 1: 位置編碼 (Positional Encoding) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- Transformer 基礎 2: 定義 Transformer 模型架構 ---
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_classes, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        
        # 將輸入特徵維度映射到模型內部維度 (d_model)
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer Encoder 層
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        self.d_model = d_model
        # 最終的分類層
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, src):
        src = self.input_fc(src) # (batch, seq_len, input_size) -> (batch, seq_len, d_model)
        # PyTorch Transformer's Positional Encoding expects (seq_len, batch, d_model)
        src = src.permute(1, 0, 2) # (batch, seq_len, d_model) -> (seq_len, batch, d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2) # Back to (batch, seq_len, d_model)
        
        output = self.transformer_encoder(src, self.src_mask)
        
        # 我們取序列輸出的平均值來做最終分類
        output = output.mean(dim=1)
        
        output = self.classifier(output)
        return output

# --- 主要執行流程 ---
if __name__ == '__main__':
    # 數據準備 (與 LSTM 腳本相同)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    data = pd.read_csv('data/final_data_multi.csv', index_col='date', parse_dates=True)
    features_cols = ['SMA_10', 'SMA_20', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
    
    scaler = StandardScaler()
    data[features_cols] = scaler.fit_transform(data[features_cols])
    
    sequence_length = 10
    X, y = create_sequences_grouped(data, features_cols, 'target', sequence_length)
    y = y + 1

    # 時序分割
    split_point = int(len(X) * 0.8)
    X_train, X_test = X[:split_point], X[split_point:]
    y_train, y_test = y[:split_point], y[split_point:]

    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    # 實例化模型
    input_size = len(features_cols) # 輸入特徵數 (6)
    d_model = 64                   # 模型內部維度 (需大於 nhead)
    nhead = 8                      # 注意力頭的數量 (d_model 需能被 nhead 整除)
    num_encoder_layers = 3         # Encoder 層數
    num_classes = 3
    model = TransformerModel(input_size, d_model, nhead, num_encoder_layers, num_classes).to(device)

    # 損失函數與優化器
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # 【新增】學習率規劃器
    # 這裡我們設定：每訓練 100 個 epochs，學習率就乘以 0.5 (衰減)
    # 這能讓模型在初期快速學習，在後期精細微調
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    # 訓練模型
    print("[INFO] Starting Transformer model training...")
    epochs = 400 # Transformer 可能收斂更快或更慢，我們先從 400 開始

    for epoch in range(epochs):
        model.train()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 【新增】在每個 epoch 結束後，更新學習率
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
    
    # 評估模型
    print("\n[INFO] Evaluating Transformer model on the test set...")
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
        
        accuracy = (predicted == y_test_tensor).sum().item() / len(y_test_tensor)
        print(f"\n  - Accuracy: {accuracy:.4f}")
        
        print("\n  - Classification Report:")
        y_pred_np = predicted.cpu().numpy()
        y_test_np = y_test_tensor.cpu().numpy()
        print(classification_report(y_pred_np, y_test_np, target_names=['Sell (-1)', 'Hold (0)', 'Buy (1)']))

    print("\n[INFO] Saving the trained Transformer model to model/transformer_multi_stock.pth...")
    torch.save(model.state_dict(), 'model/transformer_multi_stock.pth')
    print("[SUCCESS] Model saved.")
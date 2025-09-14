import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 層：它會處理時間序列數據
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        
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
# node/query_optimize/logic.py
from datetime import datetime

# 股票名稱 ➝ 股票代號 對照表
STOCK_SYMBOL_TO_ID = {
    "TSMC": "2330",
    "鴻海": "2317",
    "聯電": "2303",
    "台積電": "2330",
    # 可持續擴充
}


def get_current_datetime():
    return {"current_date": datetime.now().strftime("%Y-%m-%d")}

def normalize_stock_id(stock_symbol: str) -> str:
    return STOCK_SYMBOL_TO_ID.get(stock_symbol.upper(), stock_symbol)
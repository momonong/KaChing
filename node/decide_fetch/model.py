# node/decide_fetch/model.py
from typing import Dict, Optional
import pandas as pd
from pydantic import BaseModel


class StockState(BaseModel):
    query: Optional[str] = None
    stock_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    llm_response: Optional[str] = None

    fetch_results: Dict[str, pd.DataFrame] = {}  # 所有工具結果統一存這

    model_config = {"arbitrary_types_allowed": True}  # ✅ for Pydantic v2+

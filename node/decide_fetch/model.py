# node/decide_fetch/model.py
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Any
import pandas as pd

class StockState(BaseModel):
    query: Optional[str] = None
    stock_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    fetch_tasks: List[str] = []
    df_price: Optional[pd.DataFrame] = None
    df_revenue: Optional[pd.DataFrame] = None
    llm_response: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

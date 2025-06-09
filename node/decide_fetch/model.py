# node/decide_fetch/model.py
from typing import Dict, Optional
import pandas as pd
from pydantic import BaseModel, Field


class StockState(BaseModel):
    query: Optional[str] = None
    stock_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    llm_response: Optional[str] = None

    fetch_results: Dict[str, pd.DataFrame] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}  # ✅ for Pydantic v2+

from pydantic import BaseModel
from typing import Dict
import pandas as pd

class ExplainState(BaseModel):
    query: str
    stock_id: str
    fetch_results: Dict[str, pd.DataFrame]  # 抓到的所有結果
    explanation: str = ""  # 解釋結果

    model_config = {
        "arbitrary_types_allowed": True
    }

from typing import Optional, List
from pydantic import BaseModel
from datetime import date

class QueryOutput(BaseModel):
    stock_id: Optional[str]
    start_date: Optional[date]
    end_date: Optional[date]
    focus: List[str]  # 例如 ['股價', '營收']

class QueryState(BaseModel):
    query: str
    result: Optional[QueryOutput] = None

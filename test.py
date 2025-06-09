# main.py
from node.query_optimize.model import QueryState
from node.query_optimize.agent import query_optimize_agent

query_state = QueryState(query="幫我查一下 TSMC 最近一週的股價跟營收")
query_result = query_optimize_agent(query_state)
print("✅ 結果:", query_result)

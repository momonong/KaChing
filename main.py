# main.py
from langgraph.graph import StateGraph
from node.decide_fetch.model import StockState
from node.decide_fetch.agent import decide_fetch_agent

workflow = StateGraph(StockState)

workflow.add_node("decide", decide_fetch_agent)

workflow.set_entry_point("decide")

app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke({
        "query": "我想知道台積電最近的股價與營收如何",
        "stock_id": "2330",
        "start_date": "2025-01-01",
        "end_date": "2025-05-31",
    })

    print("🧠 fetch_tasks:", result.get("fetch_tasks"))
    print("📈 df_price:\n", result.get("df_price").head() if result.get("df_price") is not None else "無資料")
    print("💰 df_revenue:\n", result.get("df_revenue").head() if result.get("df_revenue") is not None else "無資料")
    print("🗣️ llm_response:\n", result.get("llm_response", ""))

from node.decide_fetch.graph import build_decide_fetch_graph
from node.explain.graph import build_explain_graph
from node.explain.convert import convert_to_explain_state
from node.decide_fetch.model import StockState
from node.explain.model import ExplainState


if __name__ == "__main__":
    query = {
        "query": "我想知道台積電最近的股價與營收如何",
        "stock_id": "2330",
        "start_date": "2025-01-01",
        "end_date": "2025-05-31",
    }

    fetch_app = build_decide_fetch_graph()
    explain_app = build_explain_graph()

    fetch_result = fetch_app.invoke(query)
    stock_state = StockState(**fetch_result)
    explain_input = convert_to_explain_state(stock_state)
    explain_result = explain_app.invoke(explain_input)
    explain_state = ExplainState(**explain_result)

    print("📦 所有 fetch_results：")
    for key, df in explain_state.fetch_results.items():
        print(f"🔹 {key}:\n", df.head())

    print("\n🧠 LLM Explanation:")
    print(explain_state.explanation)

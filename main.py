from node.query_optimize.graph import build_query_optimize_graph
from node.decide_fetch.graph import build_decide_fetch_graph
from node.explain.graph import build_explain_graph
from node.decide_fetch.convert import convert_to_stock_input
from node.explain.convert import convert_to_explain_state
from node.decide_fetch.model import StockState
from node.explain.model import ExplainState
from node.query_optimize.model import QueryState, QueryOutput

if __name__ == "__main__":
    # ✅ 使用者輸入自然語言
    user_query = "我想知道台積電最近幾天的股價與營收如何"

    # 🔍 Step 1: query_optimize
    query_optimize_app = build_query_optimize_graph()
    query_state = QueryState(query=user_query)
    query_result = query_optimize_app.invoke(query_state)
    query_state = QueryState(**dict(query_result))  # 明確轉型
    parsed: QueryOutput = query_state.result

    print("🧠 LLM parsed query:")
    print(parsed)

    # 📈 Step 2: decide_fetch
    fetch_app = build_decide_fetch_graph()
    query_dict = convert_to_stock_input(query_state)

    print("📤 轉換後的 query_dict:")
    print(query_dict)

    fetch_result = fetch_app.invoke(query_dict)

    print("🗂 Raw fetch_result dict:")
    print(fetch_result)

    stock_state = StockState(**fetch_result)

    print("📥 StockState 決策與內容:")
    print(stock_state)

    # 🧠 Step 3: explain
    explain_app = build_explain_graph()
    explain_input = convert_to_explain_state(stock_state)
    explain_result = explain_app.invoke(explain_input)
    explain_state = ExplainState(**explain_result)

    # ✅ 輸出結果
    print("📦 所有 fetch_results：")
    for key, df in explain_state.fetch_results.items():
        print(f"🔹 {key}:")
        print("Shape:", df.shape)
        print(df)

    print("\n🧠 LLM Explanation:")
    print(explain_state.explanation)


import json
from node.decide_fetch.tools import get_registry

FUNCTION_REGISTRY = get_registry()


def dataframe_to_json_summary(df):
    if df.empty:
        return json.dumps({"summary": "No data found."})
    return df.head(5).to_json(orient="records", force_ascii=False)


def handle_tool_calls(tool_calls):
    tool_messages = []
    state_updates = {"fetch_results": {}}  # 改成集中到這裡

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        fetch_fn = FUNCTION_REGISTRY.get(tool_name)
        if not fetch_fn:
            print(f"⚠️ 工具未註冊: {tool_name}")
            continue

        df = fetch_fn(**args)

        summary_json = dataframe_to_json_summary(df)
        tool_messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": summary_json,
            }
        )

        # 儲存進 fetch_results 字典，以工具名去除 fetch_ 為 key
        result_key = tool_name.replace("fetch_", "")  # 如 stock_price
        state_updates["fetch_results"][result_key] = df

    return tool_messages, state_updates

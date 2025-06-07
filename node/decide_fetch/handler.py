# node/decide_fetch/handler.py
import json
from node.decide_fetch.tools import get_registry

FUNCTION_REGISTRY = get_registry()


def dataframe_to_json_summary(df):
    if df.empty:
        return json.dumps({"summary": "No data found."})
    return df.head(5).to_json(orient="records", force_ascii=False)


def handle_tool_calls(tool_calls):
    tool_messages = []
    state_updates = {}

    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        fetch_fn = FUNCTION_REGISTRY.get(tool_name)

        if fetch_fn:
            args = json.loads(tool_call.function.arguments)
            df = fetch_fn(**args)
            print(f"📊 Fetched data from {tool_name}: {df.shape}")

            json_data = dataframe_to_json_summary(df)

            tool_messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": json_data,
            })

            # 將資料直接放入 state
            key = f"df_{tool_name.replace('fetch_', '')}"
            state_updates[key] = df
        else:
            print(f"⚠️ 未註冊工具: {tool_name}")

    return tool_messages, state_updates

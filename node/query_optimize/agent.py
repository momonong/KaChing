# node/query_optimize/agent.py
from node.query_optimize.model import QueryState, QueryOutput
from node.query_optimize.tools import get_tools, get_registry
from node.query_optimize.logic import normalize_stock_id
from clients.gpt4_mini import client, deployment
import json
import re


def query_optimize_agent(state: QueryState) -> QueryState:
    tools = get_tools()
    registry = get_registry()
    user_query = state.query

    messages = [
        {
            "role": "system",
            "content": """
                        你是一位財經助理。你需要將使用者輸入的自然語言查詢，轉換為一個結構化參數（包括股票代號、關鍵字與起迄時間）。
                        - 請主動將公司名稱轉換為正確的股票代號，例如：
                        - TSMC ➝ 2330
                        - 鴻海 ➝ 2317
                        - 聯電 ➝ 2303
                        - 台積電 ➝ 2330
                        - 若有缺少日期請使用工具 get_current_datetime 補足。
                        你應該永遠以 function_call 的方式輸出結果。
                        請以 function_call 或純 JSON 格式回覆，避免 Markdown 格式（如 ```json）
                    """,
        },
        {"role": "user", "content": user_query},
    ]

    # 第一次 call
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        functions=[tool["function"] for tool in tools],
        function_call="auto",
    )
    message = response.choices[0].message

    # 若呼叫工具
    if message.function_call:
        fn_name = message.function_call.name
        args = json.loads(message.function_call.arguments)
        if fn_name in registry:
            result = registry[fn_name](**args) if args else registry[fn_name]()
            messages.append(
                {
                    "role": "assistant",
                    "function_call": message.function_call.model_dump(),
                }
            )
            messages.append(
                {"role": "function", "name": fn_name, "content": json.dumps(result)}
            )

            # 再次呼叫 LLM 獲得完整輸出
            response = client.chat.completions.create(
                model=deployment,
                messages=messages,
                functions=[tool["function"] for tool in tools],
                function_call="auto",
            )
            message = response.choices[0].message

    # 處理結果
    if message.function_call:
        args = json.loads(message.function_call.arguments)
    elif message.content:
        raw = message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        args = json.loads(raw)
    else:
        raise ValueError("沒有回傳 function_call 或 JSON 內容")

    # ✅ fallback: 對股票名稱做轉換
    if "stock_symbol" in args and "stock_id" not in args:
        args["stock_id"] = normalize_stock_id(args["stock_symbol"])
    if "keywords" in args and "focus" not in args:
        args["focus"] = args["keywords"]

    output = QueryOutput(
        stock_id=args["stock_id"],
        start_date=args["start_date"],
        end_date=args["end_date"],
        focus=args["focus"],
    )
    return QueryState(query=user_query, result=output)

# node/decide_fetch/agent.py
from clients.gpt4_mini import client, deployment
from node.decide_fetch.tools import get_tools
from node.decide_fetch.handler import handle_tool_calls
from node.decide_fetch.model import StockState


def decide_fetch_agent(state: StockState) -> StockState:
    query = state.query or ""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一位智慧財經助理。以下是使用者查詢條件，請基於這些條件判斷是否呼叫查詢工具：\n"
                f"- 股票代碼（stock_id）: {state.stock_id}\n"
                f"- 起始日: {state.start_date}\n"
                f"- 結束日: {state.end_date}\n"
                "請勿自行猜測股票代碼。"
            ),
        },
        {"role": "user", "content": query},
    ]

    # 第一次 API：要求模型判斷是否要呼叫工具
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        tools=get_tools(),
        tool_choice="auto",
    )

    response_message = response.choices[0].message
    messages.append(response_message)

    if response_message.tool_calls:
        tool_messages, state_updates = handle_tool_calls(response_message.tool_calls)
        messages.extend(tool_messages)

        try:
            return state.model_copy(update=state_updates)

        except Exception as e:
            return state.model_copy(update={**state_updates, "fetch_tasks": []})

    else:
        print("📭 No tool calls detected.")

    # 第二次 API：生成自然語言總結回應
    final_response = client.chat.completions.create(
        model=deployment,
        messages=messages,
    )

    print("🗣️ Final LLM Response:", final_response.choices[0].message.content)

    return state.model_copy(
        update={"llm_response": final_response.choices[0].message.content}
    )

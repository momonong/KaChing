from node.explain.model import ExplainState
from clients.gpt4_mini import client, deployment

def explain_result_llm(state: ExplainState) -> ExplainState:
    content = "\n\n".join(
        f"{key}:\n{df.head(5).to_markdown(index=False)}"
        for key, df in state.fetch_results.items()
    )

    messages = [
        {
            "role": "system",
            "content": "你是一位財經分析師，根據資料為使用者提供總結與分析。不要用 markdown 格式，直接輸出純文字。",
        },
        {
            "role": "user",
            "content": f"查詢內容：{state.query}\n\n資料如下：\n{content}",
        },
    ]

    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
    )

    summary = response.choices[0].message.content or ""

    return state.model_copy(update={"explanation": summary})

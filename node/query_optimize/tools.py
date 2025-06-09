import inspect
from node.query_optimize import logic

def get_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "取得目前的日期與時間（格式為 YYYY-MM-DD）",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "extract_query_params",
                "description": "從使用者查詢中抽取最終的查詢參數",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stock_id": {"type": "string"},
                        "focus": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                    },
                    "required": ["stock_id", "focus", "start_date", "end_date"],
                },
            },
        },
    ]



def get_registry():
    return {
        name: fn
        for name, fn in inspect.getmembers(logic, inspect.isfunction)
        if name == "get_current_datetime"
    }

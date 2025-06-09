# node/decide_fetch/tools.py
import inspect
from node.decide_fetch import logic

def get_tools():
    tools = []

    for name, fn in inspect.getmembers(logic, inspect.isfunction):
        if name.startswith("fetch_"):
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"{name} 查詢函式",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "stock_id": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"},
                        },
                        "required": ["stock_id", "start_date", "end_date"]
                    }
                }
            })

    return tools


def get_registry():
    return {
        name: fn for name, fn in inspect.getmembers(logic, inspect.isfunction)
        if name.startswith("fetch_")
    }

# node/query_optimize/convert.py

from node.query_optimize.model import QueryState

def convert_to_stock_input(state: QueryState) -> dict:
    return {
        "query": state.query,
        "stock_id": state.result.stock_id,
        "start_date": str(state.result.start_date),
        "end_date": str(state.result.end_date),
    }

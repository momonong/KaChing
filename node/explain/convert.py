from node.decide_fetch.model import StockState
from node.explain.model import ExplainState

def convert_to_explain_state(stock_state: StockState) -> ExplainState:
    return ExplainState(
        query=stock_state.query,
        stock_id=stock_state.stock_id,
        fetch_results=stock_state.fetch_results
    )

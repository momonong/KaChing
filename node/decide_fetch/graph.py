# node/decide_fetch/graph.py
from langgraph.graph import StateGraph
from node.decide_fetch.model import StockState
from node.decide_fetch.agent import decide_fetch_agent

def build_decide_fetch_graph():
    g = StateGraph(StockState)
    g.add_node("decide", decide_fetch_agent)
    g.set_entry_point("decide")
    return g.compile()

from langgraph.graph import StateGraph
from node.query_optimize.model import QueryState
from node.query_optimize.agent import query_optimize_agent

def build_query_optimize_graph():
    g = StateGraph(QueryState)
    g.add_node("optimize", query_optimize_agent)
    g.set_entry_point("optimize")
    g.set_finish_point("optimize")
    return g.compile()

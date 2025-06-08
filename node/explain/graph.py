# node/explain/graph.py
from langgraph.graph import StateGraph
from node.explain.model import ExplainState
from node.explain.agent import explain_result_llm

def build_explain_graph():
    g = StateGraph(ExplainState)
    g.add_node("explain", explain_result_llm)
    g.set_entry_point("explain")
    return g.compile()

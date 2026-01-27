from langgraph.graph.state import StateNode

Node = StateNode | tuple[str, StateNode]

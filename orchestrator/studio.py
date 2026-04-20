from orchestrator.graph import build_graph

# LangGraph Studio automatically provides its own advanced checkpointer!
# So we just pass None here to let the UI take control.
graph = build_graph(None)

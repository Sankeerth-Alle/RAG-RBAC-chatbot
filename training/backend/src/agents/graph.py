from langgraph.graph import START, StateGraph

from agents.states import State
from agents.nodes import retrieve, generate


def chatbot_graph():

    graph_builder = StateGraph(State)

    graph_builder.add_sequence([
        retrieve,
        generate
    ])

    graph_builder.add_edge(
        START,
        "retrieve"
    )

    graph = graph_builder.compile()

    return graph
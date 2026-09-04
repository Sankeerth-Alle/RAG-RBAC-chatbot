from agents.graph import chatbot_graph


def chatbot_service(access_level: str, question: str):
    initial_state = {
        "access_level": access_level,
        "question": question
    }
    chatbot_agent = chatbot_graph()
    try:
        output = chatbot_agent.invoke(initial_state)
        if output and isinstance(output.get("answer"), dict):
            return output["answer"]
    except Exception as e:
        print(f"ERROR: Chatbot workflow failed: {type(e).__name__}")

    return {
        "answer": "Unable to process the question.",
        "sources": []
    }

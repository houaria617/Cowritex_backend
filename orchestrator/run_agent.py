"""
run_agent.py
Manual testing script for the Cowritex LangGraph Orchestrator.
"""
import uuid
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from orchestrator.graph import build_graph

# Load your Supabase and LLM API keys from .env
load_dotenv()


def main():
    # 1. Initialize checkpointer and compile the graph
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer)

    # 2. Setup a unique thread for memory tracking
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # 3. Define the initial state (matching your GraphState schema)
    # Feel free to change the user_message to test 'visualize' or 'chat' intents!
    initial_state = {
        "project_id": "manual-test-proj",
        "user_id": "test-user-1",
        "user_message": "Draft an introduction about Random Forest stacking for a data mining paper.",
        "preferences": {
            "llm_provider": "groq",  # Or change to "google-genai" if you fixed the quota!
            "writing_style": "academic",
            "tone": "formal"
        }
    }

    print("🚀 Starting Cowritex Agent...")
    print(f"💬 User Prompt: '{initial_state['user_message']}'\n")

    # 4. Stream the graph execution so we can see nodes running in real-time
    print("--- Execution Log ---")
    for event in graph.stream(initial_state, config):
        for node_name, node_state in event.items():
            print(f"✅ Node [{node_name}] executed.")

    # 5. Check if the graph paused at the HITL node
    snapshot = graph.get_state(config)
    if snapshot.next and "hitl" in snapshot.next:
        print("\n⏸️ Graph is PAUSED waiting for Human-in-the-Loop (HITL) review.")
        current_output = snapshot.values.get(
            "agent_output", "No output generated.")

        print(f"\n📄 Agent Draft:\n{'-'*50}\n{current_output}\n{'-'*50}\n")

        # 6. Prompt you (the developer) for manual input in the terminal
        action = input(
            "What would you like to do? (approve / reject / edit): ").strip().lower()
        update_data = {"hitl_action": action}

        if action == "edit":
            update_data["human_edited_text"] = input(
                "✍️ Enter your edited text: ")
        elif action in ("reject", "regenerate"):
            update_data["hitl_feedback"] = input(
                "💬 Enter feedback for the agent (e.g., 'Make it longer'): ")

        # 7. Inject the human feedback into the state and resume the graph
        print(f"\n▶️ Resuming graph with action: '{action}'...")
        graph.update_state(config, update_data)

        for event in graph.stream(None, config):
            for node_name, node_state in event.items():
                print(f"✅ Node [{node_name}] executed.")

    # 8. Fetch and print the final state after everything completes
    final_state = graph.get_state(config).values
    print("\n🎉 Flow Complete!")

    # If the flow routed to the error_handler, print that out
    if final_state.get("error"):
        print(f"⚠️ Error encountered: {final_state['error']}")


if __name__ == "__main__":
    main()

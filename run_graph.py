from langchain_core.messages import AIMessage, ToolMessage
from dotenv import load_dotenv
import logging

from chatbot import graph, create_response

logging.basicConfig(level=logging.INFO)


def handle_tool_interruption(snapshot, config_1, events):
    last_message = snapshot.values["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls or not isinstance(tool_calls, list) or not tool_calls:
        logging.error("No valid tool_calls found in last_message.")
        print("Error: No valid tool calls found.")
        return
    print(f"Assistant wants to use tools: {tool_calls}")
    approval = input("Approve? (y/n): ").strip().lower()
    logging.info(f'Tool approval: {approval}')
    if approval == 'y':
        events = list(graph.stream(
            None,
            config_1,
            stream_mode="values"
        ))
        print("Assistant:", events[-1]["messages"][-1].pretty_print())
    else:
        print("Tool call rejected - conversation continues... \n")
        custom_input = input("Provide your input: ").strip()
        if not custom_input:
            print("No input provided. Skipping.")
            return
        logging.info(f'Custom input after rejection: {custom_input}')
        answer = custom_input
        new_messages = [
            ToolMessage(content=answer, tool_call_id=tool_calls[0]["id"]),
            AIMessage(content=answer),
        ]
        graph.update_state(
            config_1,
            {"messages": new_messages},
            as_node="chatbot",
        )
        print("\n\n")
        print('Assistant: ', events[-1]["messages"][-1].pretty_print())

def handle_human_interruption(snapshot, config_1, events):
    print("Human intervention required.")
    approval = input("Assistance? (y/n): ").strip().lower()
    if approval == 'y':
        assist_input = input("Provide your input: ").strip()
        if not assist_input:
            print("No input provided. Skipping.")
            return
        ai_message = snapshot.values["messages"][-1]
        human_response = assist_input
        tool_message = create_response(human_response, ai_message)
        graph.update_state(config_1, {"messages": [tool_message]})
        print("\n\n")
        print('Assistant: ', events[-1]["messages"][-1].pretty_print())
    else:
        print("Human intervention rejected - conversation continues... \n")
        ai_message = snapshot.values["messages"][-1]
        human_response = (
            "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
            " It's much more reliable and extensible than simple autonomous agents."
        )
        tool_message = create_response(human_response, ai_message)
        graph.update_state(config_1, {"messages": [tool_message]})

def run_graph():
    # Initialize the graph with the configuration

    config_1 = {"configurable": {"thread_id": "chatbot_1"}}

    load_dotenv()
    logging.info('LangGraph Chatbot started.')

    print('Welcome to the LangGraph Chatbot! Type "quit" to exit.\n')

    while True:
        user_input = input('User: ').strip()
        logging.info(f'User input: {user_input}')
        if user_input.lower() in ["quit", "exit", "q"]:
            logging.info('User exited the chatbot.')
            print("Goodbye!")
            break

        try:
            # STEP 1: Send user message and get to the interruption point
            events = list(graph.stream(
                {"messages": ("user", user_input)},
                config_1,
                stream_mode="values"
            ))

            # STEP 2: Check if we're interrupted (waiting for tool approval)
            snapshot = graph.get_state(config_1)

            if snapshot.next == 'tools':  # We're at an interruption
                logging.info('Tool call detected.')
                print("Tool call detected!")
                handle_tool_interruption(snapshot, config_1, events)
            elif snapshot.next == 'human':
                handle_human_interruption(snapshot, config_1, events)
            elif snapshot.next == '__end__':
                print("Conversation ended.")
                break
            else:
                # No interruption, just show the response
                print("Assistant:", events[-1]["messages"][-1].pretty_print())
        except Exception as e:
            logging.error(f'Error during chatbot execution: {e}')
            print(f'Error: {e}')

import os
from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict
import logging

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AIMessage, ToolMessage
from langchain_tavily import TavilySearch
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)

memory = MemorySaver()

load_dotenv()

def get_tavily_tool():
    # Ensure the API key is set before creating the tool
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        logging.error("TAVILY_API_KEY environment variable not set.")
        raise ValueError("TAVILY_API_KEY environment variable not set.")
    logging.info("TavilySearch tool initialized.")
    return TavilySearch(max_results=2)


class State(TypedDict):
  messages: Annotated[list, add_messages]
  ask_human: bool


graph_builder = StateGraph(State)

tool = get_tavily_tool()
tools=[tool]

llm = ChatOpenAI(temperature=0)
llm_with_tools = llm.bind_tools(tools=tools)


class RequestAssistance(BaseModel):
    """Escalate the conversation to an expert. Use this if you are unable to assist directly or if the user requires support beyond your permissions.

    To use this function, relay the user's 'request' so the expert can provide the right guidance.
    """

    request: str


def create_response(response: str, ai_message: AIMessage):
    if not hasattr(ai_message, 'tool_calls') or not ai_message.tool_calls:
        logging.warning("AIMessage missing tool_calls; cannot create ToolMessage.")
        return ToolMessage(content=response, tool_call_id="unknown")
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0].id
    )


def chatbot(state: State):
    if not state["messages"]:
        logging.error("No messages in state; cannot invoke LLM.")
        return {"messages": [], "ask_human": False}
    logging.info(f"Invoking LLM with messages: {state['messages']}")
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    # More robust tool call detection
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for call in response.tool_calls:
            if call["name"] == RequestAssistance.__name__:
                logging.info(f"Tool call for RequestAssistance detected: {call['name']}")
                ask_human = True
                break
    return {
        "messages": [response],
        "ask_human": ask_human
    }


def human_node(state: State):
    new_messages = []
    if not state["messages"]:
        logging.warning("No messages in state for human_node.")
        return {"messages": [], "ask_human": False}
    last_message = state["messages"][-1]
    if not isinstance(last_message, ToolMessage):
        logging.info("No response from human, creating default ToolMessage.")
        new_messages.append(
            create_response("No response from human.", last_message)
        )
    # Optionally, handle actual human input here if available
    return {
        "messages": new_messages,
        "ask_human": False
    }


def select_next_node(state: State):
    logging.info(f"Selecting next node based on state: {state}")
    if state.get("ask_human", False):
        return "human"
    else:
        return tools_condition(state)


# Modularize graph construction
nodes = [
    ("human", human_node),
    ("chatbot", chatbot),
    ("tools", ToolNode(tools=tools)),
]
for name, node in nodes:
    graph_builder.add_node(name, node)
edges = [
    ("tools", "chatbot"),
    ("human", "chatbot"),
]
for src, dst in edges:
    graph_builder.add_edge(src, dst)
graph_builder.add_conditional_edges("chatbot", select_next_node, {
    "human": "human",
    "tools": "tools",
    "__end__": "__end__"
})

graph_builder.set_entry_point("chatbot")

graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"]
)
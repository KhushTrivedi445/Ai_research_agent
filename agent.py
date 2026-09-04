"""
agent.py
Core LangGraph agent logic: state, tools, model, and graph definition.
Import `chatbot` from this file wherever you need to run the agent
(e.g., app.py for the Streamlit UI).
"""

from typing import TypedDict, Annotated

import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

import sqlite3

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b")

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
tavily_search = TavilySearch(
    max_results=5,
    topic="general",  # options: "general", "news", "finance"
)


@tool
def get_weather(location: str) -> dict:
    """
    Get the current weather for a given city or location name.
    Use this tool when the user asks about weather, temperature,
    humidity, or wind conditions for a specific place.
    """
    # Step 1: Geocode the location name to lat/lon
    geo_url = "https://geocoding-api.open-meteo.com/v1/search"
    geo_params = {"name": location, "count": 1}
    geo_res = requests.get(geo_url, params=geo_params).json()

    if not geo_res.get("results"):
        return {"error": f"Could not find location: {location}"}

    place = geo_res["results"][0]
    lat, lon = place["latitude"], place["longitude"]

    # Step 2: Fetch current weather for those coordinates
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
    }
    weather_res = requests.get(weather_url, params=weather_params).json()
    current = weather_res["current"]

    return {
        "location": f"{place['name']}, {place.get('country', '')}",
        "temperature_C": current["temperature_2m"],
        "humidity_percent": current["relative_humidity_2m"],
        "wind_speed_kmh": current["wind_speed_10m"],
    }


tools = [tavily_search, get_weather]
llm_with_tools = llm.bind_tools(tools)

# ---------------------------------------------------------------------------
# System prompt — tells the model HOW to behave, including citing sources
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = SystemMessage(content=(
    "You are a research assistant. When you use the search tool to answer a "
    "question, you MUST cite your sources. After your answer, include a "
    "'Sources:' section listing the URLs you used, numbered like [1], [2], "
    "and reference them inline in your answer (e.g., 'according to [1]'). "
    "If you answer from your own knowledge without using a tool, do not "
    "fabricate sources."
))

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def chat_node(state: ChatState) -> ChatState:
    """LLM node that may answer directly or request a tool call."""
    messages = state["messages"]

    # Prepend the system prompt on every call so the model always
    # remembers the citation instructions, even deep in a long thread.
    full_messages = [SYSTEM_PROMPT] + messages

    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


# check_same_thread=False is required because Streamlit can access the
# connection from a different thread than the one that created it.
conn = sqlite3.connect("chat_history.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

chatbot = graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helpers for the Streamlit UI: list past threads, reload a thread's history
# ---------------------------------------------------------------------------
def list_thread_ids(username: str | None = None) -> list[str]:
    """
    Return distinct thread_ids that have saved checkpoints.
    If `username` is given, only return threads belonging to that user
    (threads are created as "{username}::{uuid}" in app.py).
    """
    cursor = conn.execute("SELECT DISTINCT thread_id FROM checkpoints")
    all_ids = [row[0] for row in cursor.fetchall()]
    if username is None:
        return all_ids
    prefix = f"{username}::"
    return [t for t in all_ids if t.startswith(prefix)]


def load_thread_messages(thread_id: str) -> list[BaseMessage]:
    """
    Reload the full message history for a given thread_id directly from
    the checkpointer, so the UI can show past turns after a reconnect
    or app restart.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    if not state or not state.values:
        return []
    return state.values.get("messages", [])


def get_thread_title(thread_id: str, max_length: int = 40) -> str:
    """
    Derive a short, human-readable title for a thread from its first
    user message (e.g., "What's the weather in Ahmedabad?" -> that,
    truncated). Falls back to the raw thread_id if no messages exist.
    """
    messages = load_thread_messages(thread_id)
    for m in messages:
        if isinstance(m, HumanMessage) and m.content:
            text = m.content.strip().replace("\n", " ")
            if len(text) > max_length:
                text = text[:max_length].rstrip() + "..."
            return text
    return thread_id


def get_thread_titles(username: str) -> dict[str, str]:
    """
    Return {thread_id: title} for every thread belonging to a user,
    for populating a dropdown with readable labels.
    """
    thread_ids = list_thread_ids(username=username)
    return {tid: get_thread_title(tid) for tid in thread_ids}
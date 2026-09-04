"""
app.py
Streamlit chat UI for the LangGraph research agent, gated behind a
username/password login (JWT-based session).

Run with:
    streamlit run app.py
"""

import uuid

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from agent import chatbot, load_thread_messages, get_thread_titles
from auth import create_user, authenticate_user, generate_token, verify_token

st.set_page_config(page_title="Research Agent", page_icon="🔎")


# ---------------------------------------------------------------------------
# Auth state
# ---------------------------------------------------------------------------
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None


def current_username() -> str | None:
    """Returns the logged-in username if the session's JWT is valid, else None."""
    if not st.session_state.jwt_token:
        return None
    return verify_token(st.session_state.jwt_token)


def logout():
    st.session_state.jwt_token = None
    st.session_state.pop("thread_id", None)
    st.session_state.pop("display_messages", None)


# ---------------------------------------------------------------------------
# Login / signup screen
# ---------------------------------------------------------------------------
def render_auth_screen():
    st.title("🔎 Research Agent")
    st.caption("Log in or create an account to start chatting.")

    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            ok, message = authenticate_user(username, password)
            if ok:
                st.session_state.jwt_token = generate_token(username.strip())
                st.rerun()
            else:
                st.error(message)

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username", key="signup_username")
            new_password = st.text_input("Choose a password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Sign up")

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = create_user(new_username, new_password)
                if ok:
                    st.success(message + " You can now log in.")
                else:
                    st.error(message)


# ---------------------------------------------------------------------------
# Message display helper
# ---------------------------------------------------------------------------
def to_display_messages(messages):
    display = []
    for m in messages:
        if isinstance(m, HumanMessage) and m.content:
            display.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage) and m.content:
            display.append({"role": "assistant", "content": m.content})
    return display


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------
def stream_agent_response(user_input: str, config: dict):
    for message_chunk, metadata in chatbot.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "chat_node" and message_chunk.content:
            yield message_chunk.content


# ---------------------------------------------------------------------------
# Main chat screen (only reached once logged in)
# ---------------------------------------------------------------------------
def render_chat_screen(username: str):
    st.title("🔎 Tool-Using Research Agent")
    st.caption("Ask a research question. The agent will search the web (Tavily) "
               "or fetch live weather data, then answer with cited sources.")

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"{username}::{uuid.uuid4()}"

    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.sidebar:
        st.subheader(f"👤 {username}")
        if st.button("Log out"):
            logout()
            st.rerun()

        st.divider()
        st.subheader("Session")
        st.write(f"Thread ID: `{st.session_state.thread_id}`")

        if st.button("➕ Start new conversation"):
            st.session_state.thread_id = f"{username}::{uuid.uuid4()}"
            st.session_state.display_messages = []
            st.rerun()

        st.divider()
        st.subheader("Your past conversations")

        thread_titles = get_thread_titles(username)
        my_threads = [t for t in thread_titles if t != st.session_state.thread_id]

        if my_threads:
            selected = st.selectbox(
                "Saved threads",
                options=my_threads,
                index=None,
                placeholder="Choose a thread...",
                format_func=lambda tid: thread_titles.get(tid, tid),
            )
            if selected and st.button("Load selected thread"):
                st.session_state.thread_id = selected
                past_messages = load_thread_messages(selected)
                st.session_state.display_messages = to_display_messages(past_messages)
                st.rerun()
        else:
            st.caption("No other saved conversations yet.")

    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a research question...")

    if user_input:
        st.session_state.display_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            full_response = st.write_stream(stream_agent_response(user_input, config))

        st.session_state.display_messages.append({"role": "assistant", "content": full_response})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
username = current_username()

if username is None:
    render_auth_screen()
else:
    render_chat_screen(username)
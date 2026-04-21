"""
app.py
------
Streamlit web interface for the Evidently AI documentation agent.

Usage:
    export GROQ_API_KEY="your-key"
    streamlit run app.py
"""

import asyncio
import os
import sys

import streamlit as st

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(__file__))

import ingest
import logs
import search_agent

REPO_OWNER = "evidentlyai"
REPO_NAME = "docs"

# ---------------------------------------------------------------------------
# Cached initialisation (runs only once per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def init_agent():
    with st.spinner("📥 Downloading & indexing Evidently AI docs … (this takes ~30 s)"):
        index, _ = ingest.index_data(
            REPO_OWNER,
            REPO_NAME,
            chunk=True,
            chunking_params={"method": "sliding_window", "size": 2000, "step": 1000},
        )
    with st.spinner("🤖 Initialising agent …"):
        agent = search_agent.init_agent(index, REPO_OWNER, REPO_NAME)
    return agent


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

def stream_response(agent, prompt: str):
    """Generator that yields text deltas and logs the completed interaction."""

    async def _agen():
        async with agent.run_stream(user_prompt=prompt) as result:
            last_len = 0
            full_text = ""
            async for chunk in result.stream_output(debounce_by=0.01):
                new_text = chunk[last_len:]
                last_len = len(chunk)
                full_text = chunk
                if new_text:
                    yield new_text
            logs.log_interaction_to_file(agent, result.new_messages())
            st.session_state._last_response = full_text

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agen_obj = _agen()
    try:
        while True:
            piece = loop.run_until_complete(agen_obj.__anext__())
            yield piece
    except StopAsyncIteration:
        return


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Evidently AI Docs Agent",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Evidently AI Docs Agent")
st.caption(
    "Ask anything about [Evidently AI](https://github.com/evidentlyai/docs) — "
    "powered by Groq (Llama 3.3 70B) + text search."
)

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This agent answers questions about the **Evidently AI** library by
        searching the official documentation and generating grounded answers.

        **Stack**
        - 🦙 Groq Llama 3.3 70B
        - 🔍 minsearch text search
        - 🐍 Pydantic AI
        - 🎈 Streamlit

        **Repo:** [evidentlyai/docs](https://github.com/evidentlyai/docs)
        """
    )
    st.divider()
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# Initialise agent
agent = init_agent()

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Example prompts (shown only when chat is empty)
if not st.session_state.messages:
    st.markdown("#### 💡 Try asking:")
    examples = [
        "What is Evidently AI and what can I use it for?",
        "How do I create a data drift report?",
        "What metrics are available for classification models?",
        "How do I run regression testing for LLM outputs?",
    ]
    cols = st.columns(2)
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, use_container_width=True):
            st.session_state._prefill = example
            st.rerun()

# Handle pre-filled prompt from example buttons
prefill = st.session_state.pop("_prefill", None)

# Chat input
prompt = st.chat_input("Ask about Evidently AI …") or prefill

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_text = st.write_stream(stream_response(agent, prompt))

    final_text = getattr(st.session_state, "_last_response", response_text)
    st.session_state.messages.append({"role": "assistant", "content": final_text})

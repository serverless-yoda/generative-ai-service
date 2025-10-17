import os
import requests
import streamlit as st

# ---------- Config ----------
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
STREAM_ENDPOINT = f"{API_BASE_URL}/generate/text/stream"

# ---------- SSE reader ----------
def sse_text_chunks(prompt: str):
    """
    Connect to the FastAPI SSE endpoint and yield text chunks as they arrive.

    Expects each event line like:
      data: <text chunk>
    Optional end markers such as [DONE] are supported.
    """
    with requests.get(
        STREAM_ENDPOINT,
        params={"prompt": prompt},
        stream=True,
        timeout=(5, 600),  # (connect timeout, read timeout)
    ) as resp:
        resp.raise_for_status()

        # SSE is line-based; empty line separates events
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()

            # Skip keep-alive comments or blank lines
            if not line or line.startswith(":"):
                continue

            # Only process 'data:' fields
            if line.startswith("data:"):
                data = line[5:].strip()

                # Common end markers
                if data in {"[DONE]", "DONE"}:
                    break

                # Yield the data as plain text; if your backend sends JSON per event,
                # you can json.loads(data) here.
                yield data


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Streaming Chat", page_icon="💬", layout="centered")
st.title("💬 Streaming Chat (FastAPI SSE)")

with st.sidebar:
    st.markdown("### Connection")
    st.text_input("API base URL", value=API_BASE_URL, key="api_base_url_help", disabled=True)
    st.caption(
        "Set `API_BASE_URL` env var before `streamlit run` to change the backend URL.\n"
        f"Current endpoint: `{STREAM_ENDPOINT}`"
    )
    st.divider()
    st.markdown("### Tips")
    st.caption(
        "- Backend must be running (e.g., `uvicorn app.main:app --reload`).\n"
        "- This app expects text chunks in `data:` lines."
    )

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render prior messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_prompt = st.chat_input("Ask something…")

if user_prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Stream assistant reply token-by-token
    with st.chat_message("assistant"):
        placeholder = st.empty()
        accumulated = ""

        # Stream from SSE endpoint
        try:
            for chunk in sse_text_chunks(user_prompt):
                accumulated += chunk
                # Live update; use markdown to allow model formatting
                placeholder.markdown(accumulated)
        except requests.HTTPError as http_err:
            st.error(f"HTTP error from backend: {http_err}")
        except requests.ConnectionError:
            st.error("Cannot reach backend. Is FastAPI running at the configured URL?")
        except requests.Timeout:
            st.error("Connection timed out while waiting for stream.")
        except Exception as e:
            st.error(f"Unexpected error while streaming: {e}")

        # Persist the assistant message
        if accumulated:
            st.session_state.messages.append({"role": "assistant", "content": accumulated})
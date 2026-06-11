# frontend/app.py

import json
import requests
import streamlit as st


API_BASE = "http://localhost:8000"



def get_health() -> dict:

    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "detail": "Cannot reach API server"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def ingest_file(uploaded_file) -> dict:

    try:
        content_type = "application/pdf" if uploaded_file.name.endswith(".pdf") else "text/markdown"
        response = requests.post(
            f"{API_BASE}/ingest",
            files={"file": (uploaded_file.name, uploaded_file, content_type)},
            timeout=120,  
        )
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "error", "detail": "Cannot reach API server"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def stream_query(question: str):

    try:
        with requests.post(
            f"{API_BASE}/query",
            json={"question": question, "top_k": 5},
            stream=True,       
            timeout=120,
        ) as response:

            sources = []

            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode("utf-8")

                if not line.startswith("data: "):
                    continue

                content = line[6:]    

                if content == "[DONE]":
                    yield None, sources
                    return

                elif content.startswith("[SOURCES]"):
                    try:
                        sources = json.loads(content[9:])   
                    except json.JSONDecodeError:
                        sources = []

                else:
                    yield content, []

    except requests.exceptions.ConnectionError:
        yield "Cannot reach the API server. Is it running?", []
    except Exception as e:
        yield f"Error: {str(e)}", []



st.set_page_config(
    page_title="Personal Knowledge Base",
    layout="wide",
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Knowledge Base")
    st.divider()

    # ── Server status ──────────────────────────────────────────────
    st.subheader("Server Status")
    health = get_health()

    if health.get("status") == "ok":
        st.success("API server online")
        st.caption(f"Vectors stored: {health.get('vectors_stored', 0)}")
        st.caption(f"Model: {health.get('model', 'unknown')}")
    else:
        st.error("API server offline")
        st.caption(health.get("detail", "Unknown error"))
        st.caption("Run: `uvicorn src.api.main:app --reload`")

    st.divider()

    st.subheader("Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "md", "markdown"],           
        help="Upload a PDF or Markdown file to add it to your knowledge base",
    )

    if uploaded_file is not None:
        st.caption(f"{uploaded_file.name}")
        st.caption(f"Size: {uploaded_file.size / 1024:.1f} KB")

        if st.button("Ingest Document", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                result = ingest_file(uploaded_file)

            if result.get("status") == "success":
                st.success(
                    f"Ingested {result['chunks_stored']} chunks "
                    f"from {result['pages']} pages"
                )
                if uploaded_file.name not in st.session_state.ingested_files:
                    st.session_state.ingested_files.append(uploaded_file.name)
                st.rerun()
            else:
                st.error(f"{result.get('detail', 'Ingestion failed')}")

    st.divider()

    if st.session_state.ingested_files:
        st.subheader("Ingested This Session")
        for filename in st.session_state.ingested_files:
            st.caption(f"• {filename}")
        st.divider()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.title("Ask Your Knowledge Base")
st.caption("Upload documents in the sidebar, then ask questions about them.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.caption(
                        f"• {source['filename']} — "
                        f"page {source['page_number']} — "
                        f"score {source['score']}"
                    )


if question := st.chat_input("Ask a question about your documents..."):

    # ── Show user message immediately ──────────────────────────────
    with st.chat_message("user"):
        st.markdown(question)

    # Save to history
    st.session_state.messages.append({
        "role":    "user",
        "content": question,
        "sources": [],
    })

    # ── Stream assistant response ───────────────────────────────────
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        final_sources = []

        for token, sources in stream_query(question):
            if token is None:
                final_sources = sources
                break

            if sources:
                final_sources = sources

            full_response += token
            placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        if final_sources:
            with st.expander("Sources"):
                for source in final_sources:
                    st.caption(
                        f"• {source['filename']} — "
                        f"page {source['page_number']} — "
                        f"score {source['score']}"
                    )

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "sources": final_sources,
    })
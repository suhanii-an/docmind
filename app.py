import shutil
import tempfile
from pathlib import Path

import streamlit as st

from docmind import PROVIDERS, RAGPipeline, add_to_index, build_index, get_embeddings, get_llm, load_index
from docmind.config import settings

st.set_page_config(page_title="DocMind", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def embeddings():
    return get_embeddings()


def save_uploads(files) -> Path:
    folder = Path(tempfile.mkdtemp(prefix="docmind_"))
    for file in files:
        (folder / file.name).write_bytes(file.getbuffer())
    return folder


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("store", None)
    if st.session_state.store is None:
        try:
            st.session_state.store = load_index(embeddings())
        except FileNotFoundError:
            pass


def render_sources(sources) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for i, doc in enumerate(sources, start=1):
            page = doc.metadata.get("page")
            title = doc.metadata.get("source", "unknown") + (f" · page {page}" if page else "")
            st.markdown(f"**[{i}] {title}**")
            st.caption(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))


def sidebar() -> tuple[str, str, int]:
    with st.sidebar:
        st.header("Settings")
        provider = st.selectbox("LLM provider", PROVIDERS, index=PROVIDERS.index(settings.llm_provider))
        model = st.text_input("Model", value=settings.llm_model)
        top_k = st.slider("Chunks to retrieve", 1, 10, settings.top_k)

        st.divider()
        st.header("Documents")
        uploads = st.file_uploader(
            "Upload PDF, TXT, MD or CSV",
            type=["pdf", "txt", "md", "csv"],
            accept_multiple_files=True,
        )
        col1, col2 = st.columns(2)
        if col1.button("Add to index", disabled=not uploads, use_container_width=True):
            folder = save_uploads(uploads)
            with st.spinner("Chunking and embedding..."):
                if st.session_state.store is None:
                    st.session_state.store = build_index(folder, embeddings())
                    count = st.session_state.store.index.ntotal
                else:
                    count = add_to_index(st.session_state.store, folder)
            shutil.rmtree(folder, ignore_errors=True)
            st.success(f"Indexed {count} chunks")
        if col2.button("Rebuild from data/", use_container_width=True):
            with st.spinner("Rebuilding index..."):
                st.session_state.store = build_index(settings.data_dir, embeddings())
            st.success(f"Indexed {st.session_state.store.index.ntotal} chunks")

        if st.session_state.store is not None:
            st.caption(f"Vectors in index: {st.session_state.store.index.ntotal}")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("pipeline", None)
            st.rerun()

    return provider, model, top_k



def get_pipeline(provider: str, model: str, top_k: int) -> RAGPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None

    key = (provider, model, top_k, id(st.session_state.store))

    if st.session_state.get("pipeline_key") != key:
        old_pipeline = st.session_state.pipeline

        pipeline = RAGPipeline(
            st.session_state.store,
            get_llm(provider, model),
            top_k=top_k,
        )

        if old_pipeline is not None:
            pipeline.history = old_pipeline.history

        st.session_state.pipeline = pipeline
        st.session_state.pipeline_key = key

    return st.session_state.pipeline


def main() -> None:
    init_state()
    provider, model, top_k = sidebar()

    st.title("📚 DocMind")
    st.caption("Chat with your documents · Retrieval-Augmented Generation with LangChain + FAISS")

    if st.session_state.store is None:
        st.info("Upload documents in the sidebar or run `python cli.py ingest` to get started.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])

    question = st.chat_input("Ask something about your documents")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            pipeline = get_pipeline(provider, model, top_k)
            sources, tokens = pipeline.stream(question)
            answer = st.write_stream(tokens)
            render_sources(sources)
        except Exception as exc:
            answer, sources = f"Something went wrong: {exc}", []
            st.error(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


if __name__ == "__main__":
    main()

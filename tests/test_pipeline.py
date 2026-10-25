from pathlib import Path

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from docmind import RAGPipeline, build_index, format_docs, load_documents, load_index, split_documents

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def embeddings():
    return DeterministicFakeEmbedding(size=64)


def test_load_documents_reads_markdown():
    docs = load_documents(DATA_DIR)
    assert docs
    assert all("source" in d.metadata for d in docs)


def test_split_documents_respects_chunk_size():
    chunks = split_documents(load_documents(DATA_DIR), chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 300 for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_build_and_load_index(tmp_path, embeddings):
    store = build_index(DATA_DIR, embeddings, tmp_path)
    reloaded = load_index(embeddings, tmp_path)
    assert reloaded.index.ntotal == store.index.ntotal > 0


def test_load_index_missing_raises(tmp_path, embeddings):
    with pytest.raises(FileNotFoundError):
        load_index(embeddings, tmp_path / "missing")


def test_format_docs_numbers_sources():
    docs = [
        Document(page_content="alpha", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="beta", metadata={"source": "b.md"}),
    ]
    text = format_docs(docs)
    assert "[1] (a.pdf, page 2)" in text
    assert "[2] (b.md)" in text


def test_pipeline_answers_and_tracks_history(embeddings):
    store = FAISS.from_texts(["FAISS is a similarity search library.", "LangChain builds LLM apps."], embeddings)
    llm = FakeListChatModel(responses=["FAISS is a library [1].", "What is LangChain used for?", "It builds apps [2]."])
    pipeline = RAGPipeline(store, llm, top_k=2, search_type="similarity")

    first = pipeline.ask("What is FAISS?")
    assert first.answer == "FAISS is a library [1]."
    assert len(first.sources) == 2
    assert len(pipeline.history) == 2

    second = pipeline.ask("And LangChain?")
    assert second.standalone_question == "What is LangChain used for?"
    assert second.answer == "It builds apps [2]."
    assert len(pipeline.history) == 4

    pipeline.reset()
    assert pipeline.history == []


def test_pipeline_stream_yields_tokens(embeddings):
    store = FAISS.from_texts(["Chunking splits documents."], embeddings)
    llm = FakeListChatModel(responses=["Chunking splits text."])
    pipeline = RAGPipeline(store, llm, top_k=1, search_type="similarity")

    docs, tokens = pipeline.stream("What is chunking?")
    answer = "".join(tokens)
    assert answer == "Chunking splits text."
    assert len(docs) == 1
    assert pipeline.history[-1].content == answer

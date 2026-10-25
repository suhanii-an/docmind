from pathlib import Path

from langchain_community.document_loaders import CSVLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import settings

LOADERS = {
    ".pdf": lambda path: PyPDFLoader(path),
    ".txt": lambda path: TextLoader(path, encoding="utf-8"),
    ".md": lambda path: TextLoader(path, encoding="utf-8"),
    ".csv": lambda path: CSVLoader(path, encoding="utf-8"),
}


def supported_files(source: str | Path) -> list[Path]:
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        return [path] if path.suffix.lower() in LOADERS else []
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in LOADERS)


def load_documents(source: str | Path) -> list[Document]:
    documents = []
    for file in supported_files(source):
        loader = LOADERS[file.suffix.lower()](str(file))
        for doc in loader.load():
            doc.metadata["source"] = file.name
            if isinstance(doc.metadata.get("page"), int):
                doc.metadata["page"] = doc.metadata["page"] + 1
            documents.append(doc)
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = [c for c in splitter.split_documents(documents) if c.page_content.strip()]
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def get_embeddings(model_name: str | None = None) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name or settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(
    source: str | Path | None = None,
    embeddings: Embeddings | None = None,
    index_dir: str | Path | None = None,
) -> FAISS:
    chunks = split_documents(load_documents(source or settings.data_dir))
    if not chunks:
        raise ValueError("No supported documents found to index (pdf, txt, md, csv).")
    store = FAISS.from_documents(chunks, embeddings or get_embeddings())
    store.save_local(str(index_dir or settings.index_dir))
    return store


def add_to_index(
    store: FAISS,
    source: str | Path,
    index_dir: str | Path | None = None,
) -> int:
    chunks = split_documents(load_documents(source))
    if chunks:
        store.add_documents(chunks)
        store.save_local(str(index_dir or settings.index_dir))
    return len(chunks)


def load_index(
    embeddings: Embeddings | None = None,
    index_dir: str | Path | None = None,
) -> FAISS:
    path = Path(index_dir or settings.index_dir)
    if not (path / "index.faiss").exists():
        raise FileNotFoundError(f"No index found at {path}. Run `python cli.py ingest` first.")
    return FAISS.load_local(
        str(path),
        embeddings or get_embeddings(),
        allow_dangerous_deserialization=True,
    )

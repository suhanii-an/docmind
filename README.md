# DocMind

A small RAG project I built to learn LangChain. You put your documents (PDFs, notes, CSVs) in a folder, it indexes them, and then you can ask questions and get answers based on what's actually in those files, along with which file the answer came from.

## How it works

1. Documents are loaded and split into chunks
2. Each chunk is converted to an embedding using a sentence-transformers model (runs locally)
3. Embeddings are stored in a FAISS index
4. When you ask a question, the most relevant chunks are retrieved (using MMR so they're not all repeats of each other)
5. The chunks and your question are sent to the LLM, which answers using only that context

It also keeps track of the chat, so follow-up questions like "what about its limitations?" get rewritten into a full question before searching.

## Stack

- LangChain (LCEL chains)
- FAISS
- sentence-transformers (all-MiniLM-L6-v2)
- Groq / OpenAI / Gemini / Ollama for the LLM
- Streamlit for the UI
- pytest

## Setup

```bash
git clone https://github.com/suhanii-an/docmind.git
cd docmind
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows activate with `.venv\Scripts\activate`.

Add your API key in `.env`. I mostly used Groq since it has a free tier. If you'd rather run everything locally, install Ollama and set `LLM_PROVIDER=ollama` and `LLM_MODEL=llama3.1`.

## Usage

Build the index from the `data` folder:

```bash
python cli.py ingest data/
```

Chat in the terminal:

```bash
python cli.py chat --show-sources
```

Or ask a single question:

```bash
python cli.py ask "What is chunk overlap?"
```

Run the web app:

```bash
streamlit run app.py
```

You can also upload files directly from the sidebar in the app.

## Evaluation

I added a small script to check how well retrieval works. It runs a set of questions from `data/eval_set.json` and reports hit rate and MRR.

```bash
python cli.py eval
```

## Tests

```bash
pytest
```

The tests use fake embeddings and a fake LLM, so they run without any API key.

## Project structure

```
docmind/
  config.py     settings from .env
  ingest.py     loading, chunking, building the index
  llm.py        picks the LLM provider
  chain.py      the RAG chain and chat history
  evaluate.py   retrieval evaluation
app.py          Streamlit app
cli.py          command line
data/           sample document and eval questions
tests/
```

## Things I'd like to add

- Hybrid search (BM25 + vectors)
- Reranking the retrieved chunks
- Support for more file types like docx

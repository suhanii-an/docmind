# Retrieval-Augmented Generation Handbook

## What is RAG

Retrieval-Augmented Generation (RAG) is a technique that improves the answers of a large language model by giving it relevant information retrieved from an external knowledge source at query time. Instead of relying only on what the model memorised during training, the system first searches a document collection, then passes the most relevant passages to the model as context.

RAG reduces hallucinations, keeps answers up to date without retraining, and allows the model to cite where its information came from.

## The RAG Pipeline

A typical RAG pipeline has two phases.

The indexing phase runs offline. Documents are loaded, split into smaller chunks, converted into embedding vectors, and stored in a vector database.

The query phase runs for every question. The question is embedded with the same embedding model, the vector database returns the nearest chunks, and the language model generates an answer grounded in those chunks.

## Chunking

Chunking is the process of splitting long documents into smaller passages. Chunk size controls how much text each passage contains, and chunk overlap repeats a small amount of text between neighbouring chunks so that sentences on a boundary are not lost.

Small chunks give precise retrieval but may lack context. Large chunks carry more context but dilute the embedding. A chunk size between 500 and 1000 characters with an overlap of 10 to 20 percent is a common starting point.

The recursive character text splitter tries to split on paragraphs first, then lines, then sentences, and finally words, which keeps related text together.

## Embeddings

An embedding is a dense numerical vector that represents the meaning of a piece of text. Texts with similar meaning have vectors that are close together. Sentence-transformer models such as all-MiniLM-L6-v2 produce 384-dimensional embeddings and run quickly on a CPU.

Normalising embeddings allows cosine similarity to be computed as a simple dot product.

## Vector Stores

A vector store saves embeddings and supports nearest neighbour search. FAISS, developed by Meta AI, is an open-source library for efficient similarity search that can run fully in memory and be saved to disk. Other popular options include Chroma, Pinecone, Weaviate and Qdrant.

## Retrieval Strategies

Similarity search returns the top k chunks with the highest similarity to the query.

Maximal Marginal Relevance (MMR) balances relevance and diversity. It first fetches a larger candidate set and then selects chunks that are relevant to the query but not redundant with each other. MMR is useful when documents contain many repeated passages.

Hybrid search combines keyword search such as BM25 with vector search to capture both exact terms and semantic meaning.

## Conversational RAG

In a chat setting, follow-up questions often depend on earlier messages, for example "what about its limitations?". A question condensing step uses the language model to rewrite the follow-up into a standalone question before retrieval, so the vector search receives a complete query.

## LangChain

LangChain is a framework for building applications with language models. It provides document loaders, text splitters, embedding integrations, vector store wrappers, retrievers, prompt templates and chat model interfaces.

The LangChain Expression Language (LCEL) lets developers compose these components using the pipe operator, for example prompt | llm | parser. Chains built with LCEL support invoke, batch and stream out of the box.

## Evaluating RAG

Retrieval quality can be measured with hit rate, which is the fraction of questions where a relevant chunk appears in the top k results, and Mean Reciprocal Rank (MRR), which rewards relevant chunks that appear near the top.

Generation quality is often judged on faithfulness, meaning the answer is supported by the retrieved context, and answer relevance, meaning the answer addresses the question.

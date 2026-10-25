from dataclasses import dataclass
from operator import itemgetter
from typing import Iterator

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import VectorStore

from .config import settings

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given the conversation so far and a follow-up question, rewrite the follow-up "
            "as a standalone question that can be understood without the conversation. "
            "Do not answer it. Return only the rewritten question.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are DocMind, an assistant that answers questions strictly from the context below.\n"
            "Rules:\n"
            "- Use only the information in the context.\n"
            "- If the context does not contain the answer, reply that you could not find it in the documents.\n"
            "- Cite the sources you used with their numbers, for example [1] or [2][3].\n"
            "- Be concise and well structured.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant context found."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"{source}, page {page}" if page else source
        blocks.append(f"[{i}] ({label})\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_rag_chain(retriever: BaseRetriever, llm: BaseChatModel) -> Runnable:
    generate = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(
        docs=itemgetter("question") | retriever,
        question=itemgetter("question"),
        history=itemgetter("history"),
    ).assign(answer=generate)


@dataclass
class RAGResponse:
    answer: str
    standalone_question: str
    sources: list[Document]


class RAGPipeline:
    def __init__(
        self,
        vectorstore: VectorStore,
        llm: BaseChatModel,
        top_k: int | None = None,
        search_type: str = "mmr",
        max_history_turns: int | None = None,
    ):
        k = top_k or settings.top_k
        search_kwargs = {"k": k, "fetch_k": k * 4} if search_type == "mmr" else {"k": k}
        self.retriever = vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        self.llm = llm
        self.max_history_turns = max_history_turns or settings.max_history_turns
        self.history: list[BaseMessage] = []
        self.chain = build_rag_chain(self.retriever, llm)
        self.condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()
        self.answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    def standalone_question(self, question: str) -> str:
        if not self.history:
            return question
        rewritten = self.condense_chain.invoke({"history": self.history, "question": question}).strip()
        return rewritten or question

    def ask(self, question: str) -> RAGResponse:
        standalone = self.standalone_question(question)
        result = self.chain.invoke({"question": standalone, "history": self.history})
        self._remember(question, result["answer"])
        return RAGResponse(result["answer"], standalone, result["docs"])

    def stream(self, question: str) -> tuple[list[Document], Iterator[str]]:
        standalone = self.standalone_question(question)
        docs = self.retriever.invoke(standalone)
        inputs = {"context": format_docs(docs), "history": list(self.history), "question": standalone}

        def tokens() -> Iterator[str]:
            parts = []
            for token in self.answer_chain.stream(inputs):
                parts.append(token)
                yield token
            self._remember(question, "".join(parts))

        return docs, tokens()

    def reset(self) -> None:
        self.history = []

    def _remember(self, question: str, answer: str) -> None:
        self.history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        self.history = self.history[-self.max_history_turns * 2 :]

import json
from dataclasses import dataclass
from pathlib import Path

from langchain_core.vectorstores import VectorStore


@dataclass
class RetrievalReport:
    total: int
    hits: int
    mrr: float
    failures: list[str]

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


def evaluate_retrieval(vectorstore: VectorStore, eval_file: str | Path, k: int = 4) -> RetrievalReport:
    cases = json.loads(Path(eval_file).read_text(encoding="utf-8"))
    hits, reciprocal_ranks, failures = 0, [], []

    for case in cases:
        docs = vectorstore.similarity_search(case["question"], k=k)
        keyword = case["expected_keyword"].lower()
        rank = next((i for i, d in enumerate(docs, start=1) if keyword in d.page_content.lower()), None)
        if rank:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
            failures.append(case["question"])

    mrr = sum(reciprocal_ranks) / len(cases) if cases else 0.0
    return RetrievalReport(total=len(cases), hits=hits, mrr=mrr, failures=failures)

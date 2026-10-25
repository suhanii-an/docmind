import argparse
import sys

from docmind import RAGPipeline, build_index, get_embeddings, get_llm, load_index
from docmind.config import settings
from docmind.evaluate import evaluate_retrieval


def cmd_ingest(args: argparse.Namespace) -> None:
    store = build_index(args.source, index_dir=args.index)
    print(f"Indexed {store.index.ntotal} chunks from '{args.source}' into '{args.index}'.")


def print_sources(docs) -> None:
    print("\nSources:")
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page")
        label = f"{doc.metadata.get('source', 'unknown')}" + (f" (page {page})" if page else "")
        print(f"  [{i}] {label}")


def cmd_chat(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(
        load_index(index_dir=args.index),
        get_llm(args.provider, args.model),
        top_k=args.k,
    )
    print("DocMind chat. Type 'exit' to quit, 'reset' to clear history.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            pipeline.reset()
            print("History cleared.\n")
            continue
        docs, tokens = pipeline.stream(question)
        print("DocMind: ", end="", flush=True)
        for token in tokens:
            print(token, end="", flush=True)
        print()
        if args.show_sources:
            print_sources(docs)
        print()


def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = RAGPipeline(
        load_index(index_dir=args.index),
        get_llm(args.provider, args.model),
        top_k=args.k,
    )
    response = pipeline.ask(args.question)
    print(response.answer)
    print_sources(response.sources)


def cmd_eval(args: argparse.Namespace) -> None:
    report = evaluate_retrieval(load_index(get_embeddings(), args.index), args.file, k=args.k)
    print(f"Questions : {report.total}")
    print(f"Hit rate  : {report.hit_rate:.2%}")
    print(f"MRR       : {report.mrr:.3f}")
    for question in report.failures:
        print(f"  missed -> {question}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docmind", description="RAG over your documents with LangChain")
    parser.add_argument("--index", default=settings.index_dir)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Load, chunk and embed documents into FAISS")
    ingest.add_argument("source", nargs="?", default=settings.data_dir)
    ingest.set_defaults(func=cmd_ingest)

    for name, func, help_text in (
        ("chat", cmd_chat, "Interactive chat with memory"),
        ("ask", cmd_ask, "Ask a single question"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--provider", default=settings.llm_provider)
        p.add_argument("--model", default=settings.llm_model)
        p.add_argument("-k", type=int, default=settings.top_k)
        if name == "ask":
            p.add_argument("question")
        else:
            p.add_argument("--show-sources", action="store_true")
        p.set_defaults(func=func)

    evaluate = sub.add_parser("eval", help="Measure retrieval hit rate and MRR")
    evaluate.add_argument("--file", default="data/eval_set.json")
    evaluate.add_argument("-k", type=int, default=settings.top_k)
    evaluate.set_defaults(func=cmd_eval)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

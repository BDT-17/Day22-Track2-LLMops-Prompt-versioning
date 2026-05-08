from langsmith import traceable

from config import load_config
from qa_pairs import SAMPLE_QUESTIONS
from rag_utils import build_vectorstore, create_remote_chat_client

RETRIEVER = None
REMOTE_LLM = None


@traceable(name="rag-query", tags=["rag", "step1"])
def ask(question: str) -> str:
    docs = RETRIEVER.similarity_search(question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    return REMOTE_LLM.answer(question, context)


def main() -> None:
    global RETRIEVER, REMOTE_LLM

    config = load_config()
    vectorstore = build_vectorstore(config)
    REMOTE_LLM = create_remote_chat_client(config)
    RETRIEVER = vectorstore

    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    for index, question in enumerate(SAMPLE_QUESTIONS, start=1):
        answer = ask(question)
        print(f"[{index:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question}")
        print(f"       A: {answer[:140]}")

    print(
        f"\nSent {len(SAMPLE_QUESTIONS)} traced queries to LangSmith project "
        f"'{config.langsmith_project}'."
    )


if __name__ == "__main__":
    main()

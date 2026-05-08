import json
import warnings
from pathlib import Path

import numpy as np
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from config import DATA_DIR, load_config
from qa_pairs import QA_PAIRS
from rag_utils import (
    build_vectorstore,
    create_eval_embeddings,
    create_eval_llm,
    get_prompt_templates,
    run_prompt_with_retriever,
)


warnings.filterwarnings("ignore")
PROMPTS = get_prompt_templates()
METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


def collect_rag_outputs(vectorstore, llm, prompt_version: str) -> list[dict]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PROMPTS[prompt_version]
    results = []

    print(f"\nRunning {len(QA_PAIRS)} questions with prompt {prompt_version} ...")
    for index, qa in enumerate(QA_PAIRS, start=1):
        output = run_prompt_with_retriever(retriever, llm, prompt, qa["question"])
        results.append(
            {
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": output["answer"],
                "contexts": output["contexts"],
            }
        )
        print(f"  [{index:02d}/{len(QA_PAIRS)}] {qa['question']}")
    return results


def build_ragas_dataset(rag_results: list[dict]) -> EvaluationDataset:
    samples = [
        SingleTurnSample(
            user_input=result["question"],
            response=result["answer"],
            retrieved_contexts=result["contexts"],
            reference=result["reference"],
        )
        for result in rag_results
    ]
    return EvaluationDataset(samples=samples)


def run_ragas_eval(config, rag_results: list[dict], version: str) -> dict[str, float]:
    print(f"\nRunning RAGAS evaluation for prompt {version} ...")
    dataset = build_ragas_dataset(rag_results)
    llm_eval = create_eval_llm(config)
    emb_eval = create_eval_embeddings(config)

    result = evaluate(
        dataset=dataset,
        metrics=METRICS,
        llm=llm_eval,
        embeddings=emb_eval,
    )

    scores = {}
    for metric_name in METRIC_NAMES:
        values = [value for value in result[metric_name] if value is not None]
        scores[metric_name] = float(np.mean(values)) if values else 0.0

    for metric_name, score in scores.items():
        suffix = " target-met" if metric_name == "faithfulness" and score >= 0.8 else ""
        print(f"  {metric_name:20s}: {score:.4f}{suffix}")
    return scores


def save_report(report: dict) -> Path:
    output_path = DATA_DIR / "ragas_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    config = load_config()
    vectorstore = build_vectorstore(config)

    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    v1_results = collect_rag_outputs(vectorstore, create_eval_llm(config), "v1")
    v2_results = collect_rag_outputs(vectorstore, create_eval_llm(config), "v2")

    v1_scores = run_ragas_eval(config, v1_results, "v1")
    v2_scores = run_ragas_eval(config, v2_results, "v2")

    print("\nComparison")
    for metric_name in METRIC_NAMES:
        s1 = v1_scores[metric_name]
        s2 = v2_scores[metric_name]
        winner = "V1" if s1 >= s2 else "V2"
        print(f"  {metric_name:20s}: V1={s1:.4f}  V2={s2:.4f}  winner={winner}")

    best_faithfulness = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    target_met = best_faithfulness >= 0.8
    print(f"\nFaithfulness target met: {target_met} (best={best_faithfulness:.4f})")

    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": target_met,
    }
    output_path = save_report(report)
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()

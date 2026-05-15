"""RAGAS evaluation for the health insurance RAG system.

Runs every question in eval/test_set.json through generate_answer(),
collects the answers and retrieved contexts, then scores with RAGAS.

Usage:
    python eval/run_eval.py

Output:
    - Score table printed to stdout
    - Detailed per-question results saved to eval/results.json

Cost: ~$0.02–0.03 for 15 questions at gpt-4o-mini pricing.
"""

import json
import sys
from pathlib import Path

# Make sure src/ is importable when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
# Use the classic singleton metrics — they implement ragas.metrics.base.Metric
# which is required by evaluate(). The .collections classes use a different
# base class incompatible with the standard evaluate() pipeline.
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import faithfulness, context_precision, context_recall

from src.chat import generate_answer
from src.config import OPENAI_API_KEY, CHAT_MODEL

TEST_SET_PATH = PROJECT_ROOT / "eval" / "test_set.json"
RESULTS_PATH  = PROJECT_ROOT / "eval" / "results.json"


def run() -> None:
    print("Loading test set ...")
    test_cases = json.loads(TEST_SET_PATH.read_text())
    print(f"  {len(test_cases)} questions loaded.\n")

    questions:    list[str]       = []
    answers:      list[str]       = []
    contexts:     list[list[str]] = []
    ground_truths: list[str]      = []
    details:      list[dict]      = []

    for i, case in enumerate(test_cases, start=1):
        q  = case["question"]
        gt = case["ground_truth"]
        print(f"[{i:02d}/{len(test_cases)}] {q[:80]}...")

        result = generate_answer(q, session_id=f"eval-{i}")

        # Extract the plain text of each retrieved chunk.
        chunk_texts = [c["text"] for c in result.retrieved_chunks] or [""]

        questions.append(q)
        answers.append(result.answer)
        contexts.append(chunk_texts)
        ground_truths.append(gt)

        details.append({
            "question":     q,
            "ground_truth": gt,
            "answer":       result.answer,
            "citations":    result.citations,
            "contexts":     chunk_texts,
        })

        print(f"       Answer: {result.answer[:120]}...")
        print()

    print("Running RAGAS evaluation (this calls the LLM judge — takes ~1 min) ...\n")

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # Wire RAGAS to use the same model already configured in config.py.
    # evaluate() auto-injects the llm into each MetricWithLLM metric.
    _client = OpenAI(api_key=OPENAI_API_KEY)
    llm     = llm_factory(CHAT_MODEL, client=_client)

    scores = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall],
        llm=llm,
    )

    print("\n" + "=" * 50)
    print("RAGAS SCORES")
    print("=" * 50)
    scores_df = scores.to_pandas()
    summary = {
        "faithfulness":      round(float(scores_df["faithfulness"].mean()),      4),
        "context_precision": round(float(scores_df["context_precision"].mean()), 4),
        "context_recall":    round(float(scores_df["context_recall"].mean()),    4),
    }
    for metric, score in summary.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<22} {score:.4f}  {bar}")

    print("=" * 50)
    print()

    # Save detailed per-question results.
    output = {
        "summary": summary,
        "per_question": [
            {**detail, "ragas_scores": {
                "faithfulness":      round(float(scores_df["faithfulness"].iloc[i]),      4),
                "context_precision": round(float(scores_df["context_precision"].iloc[i]), 4),
                "context_recall":    round(float(scores_df["context_recall"].iloc[i]),    4),
            }}
            for i, detail in enumerate(details)
        ],
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Detailed results saved to {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()

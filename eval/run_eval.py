"""RAGAS evaluation for the health insurance RAG system.

Runs every question in eval/test_set.json through generate_answer(),
scores with RAGAS (faithfulness, context_precision, context_recall,
answer_relevancy), and prints a per-question table plus a summary.

Usage:
    python eval/run_eval.py

Output:
    - Per-question score table printed to stdout
    - Summary scores with pass/fail indicators
    - Detailed results saved to eval/results.json

Cost: ~$0.05–0.10 for 35 questions at gpt-4o-mini pricing.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.llms import llm_factory
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from ragas.metrics import (
        faithfulness,
        context_precision,
        context_recall,
        answer_relevancy,
    )

from src.chat import generate_answer
from src.config import OPENAI_API_KEY, CHAT_MODEL

TEST_SET_PATH = PROJECT_ROOT / "eval" / "test_set.json"
RESULTS_PATH  = PROJECT_ROOT / "eval" / "results.json"

PASS_THRESHOLD = 0.60   # per-metric score below this = FAIL
METRICS = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]

# ANSI colours (disabled if not a tty)
_tty = sys.stdout.isatty()
RED    = "\033[91m" if _tty else ""
GREEN  = "\033[92m" if _tty else ""
YELLOW = "\033[93m" if _tty else ""
BOLD   = "\033[1m"  if _tty else ""
RESET  = "\033[0m"  if _tty else ""


def bar(score: float, width: int = 16) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def colour(score: float) -> str:
    if score >= 0.75:
        return GREEN
    if score >= PASS_THRESHOLD:
        return YELLOW
    return RED


def run() -> None:
    print(f"\n{BOLD}Loading test set …{RESET}")
    test_cases = json.loads(TEST_SET_PATH.read_text())
    n = len(test_cases)
    print(f"  {n} questions loaded.\n")

    questions:     list[str]       = []
    answers:       list[str]       = []
    contexts:      list[list[str]] = []
    ground_truths: list[str]       = []
    details:       list[dict]      = []

    for i, case in enumerate(test_cases, start=1):
        q  = case["question"]
        gt = case["ground_truth"]
        print(f"[{i:02d}/{n}] {q[:90]}")

        result = generate_answer(q, session_id=f"eval-{i}")
        chunk_texts = [c["text"] for c in result.retrieved_chunks] or [""]

        questions.append(q)
        answers.append(result.answer)
        contexts.append(chunk_texts)
        ground_truths.append(gt)

        status = "✓" if result.retrieved_chunks else "✗ NO CHUNKS"
        print(f"       {status}  {result.answer[:110]}…\n")

        details.append({
            "question":     q,
            "ground_truth": gt,
            "answer":       result.answer,
            "citations":    result.citations,
            "contexts":     chunk_texts,
        })

    print(f"{BOLD}Running RAGAS (LLM judge — ~1 min) …{RESET}\n")

    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    _client = OpenAI(api_key=OPENAI_API_KEY)
    llm     = llm_factory(CHAT_MODEL, client=_client)

    scores = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall, answer_relevancy],
        llm=llm,
    )
    df = scores.to_pandas()

    # ── Per-question table ────────────────────────────────────────────────
    col_w = 14
    header = f"{'Question':<55} " + "  ".join(f"{m[:col_w]:<{col_w}}" for m in METRICS)
    print("\n" + "=" * len(header))
    print(f"{BOLD}PER-QUESTION SCORES{RESET}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    fails: list[dict] = []

    for i, (detail, row) in enumerate(zip(details, df.itertuples())):
        q_short = detail["question"][:54]
        row_scores = {
            "faithfulness":      float(getattr(row, "faithfulness",      0) or 0),
            "context_precision": float(getattr(row, "context_precision", 0) or 0),
            "context_recall":    float(getattr(row, "context_recall",    0) or 0),
            "answer_relevancy":  float(getattr(row, "answer_relevancy",  0) or 0),
        }
        score_cells = "  ".join(
            f"{colour(v)}{v:.2f}{RESET}{'  ':>{col_w - 4}}" for v in row_scores.values()
        )
        print(f"{q_short:<55} {score_cells}")

        if any(v < PASS_THRESHOLD for v in row_scores.values()):
            fails.append({"index": i + 1, "question": detail["question"], "scores": row_scores})

        details[i]["ragas_scores"] = {k: round(v, 4) for k, v in row_scores.items()}

    print("=" * len(header))

    # ── Summary ───────────────────────────────────────────────────────────
    summary = {m: round(float(df[m].mean()), 4) for m in METRICS}

    print(f"\n{BOLD}SUMMARY{RESET}")
    print("─" * 52)
    for metric, score in summary.items():
        c = colour(score)
        print(f"  {metric:<24} {c}{score:.4f}{RESET}  {bar(score)}")
    print("─" * 52)
    print(f"  Questions:  {n}   Failures (any metric < {PASS_THRESHOLD}): {len(fails)}")

    # ── Failures ──────────────────────────────────────────────────────────
    if fails:
        print(f"\n{BOLD}{RED}FAILED QUESTIONS{RESET}")
        for f in fails:
            print(f"\n  [{f['index']:02d}] {f['question']}")
            for m, v in f["scores"].items():
                if v < PASS_THRESHOLD:
                    print(f"        {RED}{m}: {v:.4f}{RESET}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {"summary": summary, "per_question": details}
    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nDetailed results → {RESULTS_PATH.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    run()

import json
import os
from typing import Any, Dict, List

from eval_metrics import evaluate_single, to_dict
from pipeline import RagPipeline


# Infer project root from this file location: .../src/run_eval.py → go two levels up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
EVAL_QUESTIONS_PATH = os.path.join(DATA_DIR, "eval_questions.json")


def load_eval_questions(path: str) -> List[Dict[str, Any]]:
    """
    Load evaluation questions from a JSON file.

    Expected format: list of objects with fields like:
      - id
      - question
      - expected_keywords
      - must_be_grounded_in
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_corpus_docs(doc_ids: List[str]) -> str:
    """
    Load the "gold" context for evaluation by concatenating the texts
    of all documents whose filenames are listed in doc_ids.
    """
    parts: List[str] = []
    for doc_id in doc_ids:
        path = os.path.join(CORPUS_DIR, doc_id)
        if not os.path.exists(path):
            # If a listed document is missing, silently skip it
            continue
        with open(path, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n".join(parts)


def main() -> None:
    # Load the evaluation set and initialize the RAG pipeline
    questions = load_eval_questions(EVAL_QUESTIONS_PATH)
    pipeline = RagPipeline(corpus_dir=CORPUS_DIR)

    results: List[Dict[str, Any]] = []

    for q in questions:
        qid = q["id"]
        question = q["question"]
        expected_keywords = q.get("expected_keywords", [])
        must_be_grounded_in = q.get("must_be_grounded_in", [])

        # Run the RAG pipeline: retrieve context and produce a stub answer
        answer, contexts = pipeline.run(question, top_k=3)

        # Reference ("gold") context used for overlap evaluation:
        # built from must_be_grounded_in defined in eval_questions.json
        gold_context = load_corpus_docs(must_be_grounded_in)

        # Compute metrics for this (question, answer) pair
        eval_result = evaluate_single(
            question_id=qid,
            answer=answer,
            expected_keywords=expected_keywords,
            context_text=gold_context,
            alpha=0.5,
        )

        # Collect a detailed record for this question
        record: Dict[str, Any] = {
            "id": qid,
            "question": question,
            "answer": answer,
            "expected_keywords": expected_keywords,
            "must_be_grounded_in": must_be_grounded_in,
            "metrics": to_dict(eval_result),
            "retrieved_docs": [
                {
                    "doc_id": ctx.doc_id,
                    "score": ctx.score,
                }
                for ctx in contexts
            ],
        }
        results.append(record)

    # Print a short summary to stdout
    print("=== RAG answer evaluation summary ===")
    for r in results:
        m = r["metrics"]
        print(
            f"{r['id']}: score={m['score']:.3f} "
            f"(coverage={m['keyword_coverage']:.3f}, overlap={m['context_overlap']:.3f})"
        )

    # Save the detailed JSON report with all questions and metrics
    out_path = os.path.join(PROJECT_ROOT, "rag_eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    # Simple tokenizer:
    # - lowercase
    # - split on whitespace
    # - keep only alphanumeric characters in each token
    tokens = []
    for raw in text.lower().split():
        token = "".join(ch for ch in raw if ch.isalnum())
        if token:
            tokens.append(token)
    return tokens


def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """
    Fraction of expected keywords that are actually mentioned in the answer.
    Uses a case-insensitive substring check.
    """
    if not expected_keywords:
        return 0.0

    answer_l = answer.lower()
    hits = 0
    for kw in expected_keywords:
        if kw.lower() in answer_l:
            hits += 1
    return hits / len(expected_keywords)


def context_overlap(answer: str, context_text: str) -> float:
    """
    Simple "non-hallucination" proxy:
    fraction of answer tokens that also appear in the context.
    """
    ans_tokens = _tokenize(answer)
    ctx_tokens = set(_tokenize(context_text))

    if not ans_tokens:
        return 0.0

    hits = sum(1 for t in ans_tokens if t in ctx_tokens)
    return hits / len(ans_tokens)


@dataclass
class EvalResult:
    question_id: str
    coverage: float
    overlap: float
    score: float


def compute_score(coverage: float, overlap: float, alpha: float = 0.5) -> float:
    """
    Final score = alpha * coverage + (1 - alpha) * overlap.
    By default alpha = 0.5 (balanced weighting).
    """
    return alpha * coverage + (1.0 - alpha) * overlap


def evaluate_single(
    question_id: str,
    answer: str,
    expected_keywords: List[str],
    context_text: str,
    alpha: float = 0.5,
) -> EvalResult:
    """
    Evaluate a single (question, answer) pair:
    - compute keyword coverage,
    - compute context overlap,
    - combine them into a single score.
    """
    cov = keyword_coverage(answer, expected_keywords)
    ov = context_overlap(answer, context_text)
    score = compute_score(cov, ov, alpha=alpha)
    return EvalResult(
        question_id=question_id,
        coverage=cov,
        overlap=ov,
        score=score,
    )


def to_dict(res: EvalResult) -> Dict[str, float]:
    # Convert EvalResult dataclass to a plain dict (useful for JSON serialization)
    return {
        "question_id": res.question_id,
        "keyword_coverage": res.coverage,
        "context_overlap": res.overlap,
        "score": res.score,
    }

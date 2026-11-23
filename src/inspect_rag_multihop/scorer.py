from __future__ import annotations

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from .metrics import evaluate_retrieval, extract_answer, qa_correct


def _retrieval_metrics(state: TaskState) -> dict[str, float]:
    retrieved = state.metadata.get("retrieved", [])
    evidence = state.metadata.get("evidence", [])
    return evaluate_retrieval(
        [item["text"] for item in retrieved],
        [item["fact"] for item in evidence],
    )


def _score_metric(state: TaskState, key: str) -> Score:
    metrics = _retrieval_metrics(state)
    return Score(
        value=metrics[key],
        explanation=", ".join(f"{name}={value:.4f}" for name, value in metrics.items()),
    )


@scorer(metrics=[mean(), stderr()])
def hits_at_4() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return _score_metric(state, "hits@4")

    return score


@scorer(metrics=[mean(), stderr()])
def hits_at_10() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return _score_metric(state, "hits@10")

    return score


@scorer(metrics=[mean(), stderr()])
def map_at_10() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return _score_metric(state, "map@10")

    return score


@scorer(metrics=[mean(), stderr()])
def mrr_at_10() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        return _score_metric(state, "mrr@10")

    return score


@scorer(metrics=[accuracy(), stderr()])
def qa_accuracy() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        prediction = extract_answer(state.output.completion)
        correct = qa_correct(prediction, target.text)
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=prediction,
            explanation=f"gold={target.text!r}",
        )

    return score

from __future__ import annotations

import re
from collections.abc import Sequence

_ANSWER_RE = re.compile(r'The answer to the question is "(.*?)"')


def _normalize(text: str) -> str:
    return text.replace(" ", "").replace("\n", "")


def evaluate_retrieval(
    retrieved_texts: Sequence[str],
    gold_facts: Sequence[str],
) -> dict[str, float]:
    """Hits@4, Hits@10, MAP@10 and MRR@10 as defined by the MultiHop-RAG authors.

    A retrieved passage is relevant when it contains a gold evidence fact as a
    substring (whitespace and newlines ignored). Each gold fact counts once.
    """
    if not gold_facts:
        raise ValueError("gold_facts must not be empty")

    gold = [_normalize(fact) for fact in gold_facts]
    retrieved = [_normalize(text) for text in retrieved_texts]

    hits_at_4 = False
    hits_at_10 = False
    first_relevant_rank: int | None = None
    average_precision = 0.0
    found: list[str] = []

    for rank, passage in enumerate(retrieved[:10], start=1):
        new_matches = [
            fact for fact in gold if fact and fact in passage and fact not in found
        ]
        if not new_matches:
            continue
        hits_at_10 = True
        if first_relevant_rank is None:
            first_relevant_rank = rank
        if rank <= 4:
            hits_at_4 = True
        found.extend(new_matches)
        average_precision += len(new_matches) / rank

    return {
        "hits@4": float(hits_at_4),
        "hits@10": float(hits_at_10),
        "map@10": average_precision / min(len(gold), 10),
        "mrr@10": 1.0 / first_relevant_rank if first_relevant_rank else 0.0,
    }


def extract_answer(text: str) -> str:
    match = _ANSWER_RE.search(text)
    return match.group(1) if match else text


def has_intersection(prediction: str, gold: str) -> bool:
    return bool(set(prediction.split()).intersection(gold.split()))


def qa_correct(prediction: str, gold: str) -> bool:
    return has_intersection(prediction.lower(), gold.lower())

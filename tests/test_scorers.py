import asyncio

from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState

from inspect_rag_multihop.scorer import (
    hits_at_4,
    hits_at_10,
    map_at_10,
    mrr_at_10,
    qa_accuracy,
)

TARGET = Target("Alice")


def _state(completion: str = "") -> TaskState:
    return TaskState(
        model="mockllm/model",
        sample_id="1",
        epoch=1,
        input="Who founded the company?",
        messages=[],
        target=TARGET,
        output=ModelOutput.from_content(model="mockllm/model", content=completion),
        metadata={
            "retrieved": [{"title": "Doc A", "text": "Alice founded the company."}],
            "evidence": [{"title": "Doc A", "fact": "Alice founded the company."}],
        },
    )


def test_hits_at_4_scores_top_rank_hit():
    score = asyncio.run(hits_at_4()(_state(), TARGET))

    assert score.value == 1.0


def test_hits_at_10_scores_top_rank_hit():
    score = asyncio.run(hits_at_10()(_state(), TARGET))

    assert score.value == 1.0


def test_map_at_10_scores_top_rank_hit():
    score = asyncio.run(map_at_10()(_state(), TARGET))

    assert score.value == 1.0


def test_mrr_at_10_scores_top_rank_hit():
    score = asyncio.run(mrr_at_10()(_state(), TARGET))

    assert score.value == 1.0


def test_qa_accuracy_scores_correct_answer():
    state = _state('The answer to the question is "Alice"')

    score = asyncio.run(qa_accuracy()(state, TARGET))

    assert score.value == CORRECT
    assert score.answer == "Alice"


def test_qa_accuracy_scores_wrong_answer():
    state = _state('The answer to the question is "Bob"')

    score = asyncio.run(qa_accuracy()(state, TARGET))

    assert score.value == INCORRECT

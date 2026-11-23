import asyncio

from inspect_ai.model import ModelOutput
from inspect_ai.solver import TaskState

from inspect_rag_multihop.dataset import CorpusDocument
from inspect_rag_multihop.retrieval import BM25Index
from inspect_rag_multihop.tasks import bm25_retrieval, retrieve_then_answer


def _index():
    corpus = [
        CorpusDocument(
            title="Doc A",
            body="Alice founded the company. Bob joined later.",
            source="Src",
            published_at="2023-01-01",
            url="https://example.com/a",
        )
    ]
    return BM25Index.from_corpus(corpus, chunk_size=50, chunk_overlap=0)


def _state(query: str = "Who founded the company?") -> TaskState:
    return TaskState(
        model="mockllm/model",
        sample_id="1",
        epoch=1,
        input=query,
        messages=[],
    )


async def _forbidden_generate(state: TaskState) -> TaskState:
    raise AssertionError("bm25_retrieval must not call generate")


async def _fake_generate(state: TaskState) -> TaskState:
    state.output = ModelOutput.from_content(
        model="mockllm/model",
        content='The answer to the question is "Alice"',
    )
    return state


def test_bm25_retrieval_solver_records_passages():
    solver = bm25_retrieval(_index(), top_k=10)

    state = asyncio.run(solver(_state(), _forbidden_generate))

    assert state.metadata["retrieved"][0]["title"] == "Doc A"
    assert "Doc A" in state.output.completion


def test_retrieve_then_answer_solver_builds_prompt_and_calls_model():
    solver = retrieve_then_answer(_index(), top_k=10)

    state = asyncio.run(solver(_state(), _fake_generate))

    prompt = state.messages[-1].content
    assert "Question: Who founded the company?" in prompt
    assert "Alice founded the company." in prompt
    assert state.output.completion == 'The answer to the question is "Alice"'

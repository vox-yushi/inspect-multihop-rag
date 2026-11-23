from __future__ import annotations

from collections.abc import Sequence

from inspect_ai import Task, task
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.solver import Generate, Solver, TaskState, solver

from inspect_rag_multihop.dataset import (
    CorpusDocument,
    QueryRecord,
    build_dataset,
    load_corpus,
    load_queries,
)
from inspect_rag_multihop.retrieval import BM25Index, Chunk
from inspect_rag_multihop.scorer import (
    hits_at_4,
    hits_at_10,
    map_at_10,
    mrr_at_10,
    qa_accuracy,
)

TASK_VERSION = "1.0.0"

CONTEXT_SEPARATOR = "\n\n--------------\n\n"

QA_PROMPT = (
    "Below is a question followed by some context from different sources. "
    "Please answer the question based on the context. The answer to the "
    "question is a word or entity. If the provided information is "
    "insufficient to answer the question, respond 'Insufficient Information'. "
    "Answer directly without explanation."
)


def _retrieved_records(results: Sequence[tuple[Chunk, float]]) -> list[dict[str, object]]:
    return [
        {
            "title": chunk.title,
            "text": chunk.text,
            "score": score,
            "chunk_index": chunk.chunk_index,
        }
        for chunk, score in results
    ]


@solver
def bm25_retrieval(index: BM25Index, top_k: int = 10) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        results = index.search(state.input_text, top_k=top_k)
        state.metadata["retrieved"] = _retrieved_records(results)
        state.output = ModelOutput.from_content(
            model="bm25",
            content="\n".join(
                f"[{rank}] {chunk.title}" for rank, (chunk, _) in enumerate(results, start=1)
            ),
        )
        return state

    return solve


@solver
def retrieve_then_answer(index: BM25Index, top_k: int = 10) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        results = index.search(state.input_text, top_k=top_k)
        state.metadata["retrieved"] = _retrieved_records(results)
        context = CONTEXT_SEPARATOR.join(chunk.text for chunk, _ in results)
        prompt = f"{QA_PROMPT}\n\nQuestion: {state.input_text}\n\nContext:\n\n{context}"
        state.messages = [ChatMessageUser(content=prompt)]
        return await generate(state)

    return solve


def _build_retrieval_task(
    corpus: Sequence[CorpusDocument],
    queries: Sequence[QueryRecord],
    top_k: int = 10,
    chunk_size: int = 256,
    chunk_overlap: int = 25,
) -> Task:
    if top_k < 10:
        raise ValueError("top_k must be at least 10 because the metrics use Hits@10/MAP@10")
    index = BM25Index.from_corpus(corpus, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return Task(
        dataset=build_dataset(queries, include_null_queries=False),
        solver=bm25_retrieval(index, top_k=top_k),
        scorer=[hits_at_4(), hits_at_10(), map_at_10(), mrr_at_10()],
        version=TASK_VERSION,
    )


def _build_qa_task(
    corpus: Sequence[CorpusDocument],
    queries: Sequence[QueryRecord],
    top_k: int = 10,
    chunk_size: int = 256,
    chunk_overlap: int = 25,
) -> Task:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    index = BM25Index.from_corpus(corpus, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return Task(
        dataset=build_dataset(queries, include_null_queries=True),
        solver=retrieve_then_answer(index, top_k=top_k),
        scorer=qa_accuracy(),
        version=TASK_VERSION,
    )


@task
def multihop_rag_retrieval(
    top_k: int = 10,
    chunk_size: int = 256,
    chunk_overlap: int = 25,
) -> Task:
    """BM25 retrieval over the MultiHop-RAG corpus (null queries excluded)."""
    return _build_retrieval_task(
        load_corpus(),
        load_queries(),
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@task
def multihop_rag_qa(
    top_k: int = 10,
    chunk_size: int = 256,
    chunk_overlap: int = 25,
) -> Task:
    """Retrieve then answer MultiHop-RAG queries (including null queries)."""
    return _build_qa_task(
        load_corpus(),
        load_queries(),
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

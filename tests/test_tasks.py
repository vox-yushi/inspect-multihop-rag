import pytest

from inspect_rag_multihop.dataset import CorpusDocument, Evidence, QueryRecord
from inspect_rag_multihop.tasks import _build_qa_task, _build_retrieval_task

CORPUS = [
    CorpusDocument(
        title="Doc A",
        body="Alice founded the company. Bob joined later.",
        source="Src",
        published_at="2023-01-01",
        url="https://example.com/a",
    ),
    CorpusDocument(
        title="Doc B",
        body="The company makes widgets.",
        source="Src",
        published_at="2023-02-01",
        url="https://example.com/b",
    ),
]

QUERIES = [
    QueryRecord(
        query="Who founded the company?",
        answer="Alice",
        question_type="inference_query",
        evidence=(Evidence(title="Doc A", fact="Alice founded the company."),),
    ),
    QueryRecord(
        query="Who bought the company?",
        answer="Insufficient information.",
        question_type="null_query",
        evidence=(),
    ),
]


def test_retrieval_task_uses_non_null_samples_and_four_scorers():
    task = _build_retrieval_task(CORPUS, QUERIES)

    assert len(task.dataset) == 1
    assert len(task.scorer) == 4


def test_qa_task_includes_null_samples():
    task = _build_qa_task(CORPUS, QUERIES)

    assert len(task.dataset) == 2
    assert len(task.scorer) == 1


def test_retrieval_requires_top_k_at_least_ten():
    with pytest.raises(ValueError, match="top_k"):
        _build_retrieval_task(CORPUS, QUERIES, top_k=5)

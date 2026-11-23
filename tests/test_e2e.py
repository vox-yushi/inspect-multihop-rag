from inspect_ai import eval

from inspect_rag_multihop.dataset import CorpusDocument, Evidence, QueryRecord
from inspect_rag_multihop.tasks import _build_retrieval_task

CORPUS = [
    CorpusDocument(
        title="Doc A",
        body="Alice founded the company. Bob joined later.",
        source="Src",
        published_at="2023-01-01",
        url="https://example.com/a",
    )
]

QUERIES = [
    QueryRecord(
        query="Who founded the company?",
        answer="Alice",
        question_type="inference_query",
        evidence=(Evidence(title="Doc A", fact="Alice founded the company."),),
    )
]


def test_retrieval_task_runs_end_to_end(tmp_path):
    task = _build_retrieval_task(CORPUS, QUERIES)

    logs = eval(
        task,
        model="mockllm/model",
        limit=1,
        display="none",
        log_dir=str(tmp_path),
    )

    assert logs[0].status == "success"
    assert logs[0].results is not None

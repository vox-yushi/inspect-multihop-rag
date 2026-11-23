from inspect_rag_multihop.dataset import (
    Evidence,
    QueryRecord,
    build_dataset,
)


def _queries():
    return [
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


def test_build_dataset_excludes_null_queries():
    dataset = build_dataset(_queries())

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample.id == "multihop-rag-0"
    assert sample.target == "Alice"
    assert sample.metadata["question_type"] == "inference_query"
    assert sample.metadata["evidence"] == [
        {"title": "Doc A", "fact": "Alice founded the company."}
    ]


def test_build_dataset_can_include_null_queries():
    dataset = build_dataset(_queries(), include_null_queries=True)

    assert len(dataset) == 2
    assert dataset[1].id == "multihop-rag-1"
    assert dataset[1].metadata["evidence"] == []

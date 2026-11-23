import pytest

from inspect_rag_multihop.dataset import CorpusDocument
from inspect_rag_multihop.retrieval import BM25Index, chunk_document


def _doc(title="Doc A", body="Alpha is here. Beta is there. Gamma follows. Delta ends."):
    return CorpusDocument(
        title=title,
        body=body,
        source="Src",
        published_at="2023-01-01",
        url="https://example.com",
    )


def test_chunks_include_metadata_header():
    chunks = chunk_document(_doc(), chunk_size=4, chunk_overlap=1)

    assert len(chunks) >= 2
    assert all(chunk.text.startswith("Title: Doc A") for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_invalid_chunk_parameters_raise():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document(_doc(), chunk_size=0)
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document(_doc(), chunk_size=10, chunk_overlap=10)


def test_bm25_ranks_matching_document_first():
    corpus = [
        _doc("Doc A", "Alpha beta gamma."),
        _doc("Doc B", "Delta epsilon zeta."),
    ]
    index = BM25Index.from_corpus(corpus, chunk_size=10, chunk_overlap=0)

    results = index.search("alpha", top_k=2)

    assert results[0][0].title == "Doc A"
    assert results[0][1] >= results[1][1]


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Index([])

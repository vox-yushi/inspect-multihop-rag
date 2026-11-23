import pytest

from inspect_rag_multihop.metrics import evaluate_retrieval, extract_answer, qa_correct


def test_hits_and_map_match_official_definition():
    retrieved = ["x a y", "nothing here", "b c"] + ["z"] * 7

    metrics = evaluate_retrieval(retrieved, ["a", "b"])

    assert metrics["hits@4"] == 1.0
    assert metrics["hits@10"] == 1.0
    assert metrics["mrr@10"] == 1.0
    assert metrics["map@10"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)


def test_relevant_passage_below_rank_four():
    retrieved = ["z", "z", "z", "z", "a"] + ["z"] * 5

    metrics = evaluate_retrieval(retrieved, ["a"])

    assert metrics["hits@4"] == 0.0
    assert metrics["hits@10"] == 1.0
    assert metrics["mrr@10"] == pytest.approx(0.2)
    assert metrics["map@10"] == pytest.approx(0.2)


def test_gold_fact_is_counted_once():
    retrieved = ["a b", "a again"] + ["z"] * 8

    metrics = evaluate_retrieval(retrieved, ["a", "b"])

    assert metrics["map@10"] == pytest.approx(1.0)


def test_whitespace_is_ignored_when_matching():
    metrics = evaluate_retrieval(["foo NewYork bar"], ["New York"])

    assert metrics["hits@4"] == 1.0


def test_empty_gold_facts_raise():
    with pytest.raises(ValueError, match="gold_facts"):
        evaluate_retrieval(["x"], [])


def test_extract_answer():
    text = 'blah The answer to the question is "Sam Bankman-Fried" done'

    assert extract_answer(text) == "Sam Bankman-Fried"
    assert extract_answer("raw output") == "raw output"


def test_qa_correct_uses_word_intersection():
    assert qa_correct("The answer is Sam Bankman-Fried", "Sam Bankman-Fried")
    assert qa_correct("insufficient information", "Insufficient information.")
    assert not qa_correct("Paris", "Sam Bankman-Fried")

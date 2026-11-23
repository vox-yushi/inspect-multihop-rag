from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download
from inspect_ai.dataset import Dataset, MemoryDataset, Sample

DATASET_REPO = "yixuantt/MultiHopRAG"
# Last revision of the dataset repository (2024-01-30).
DATASET_REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"

QUERIES_FILE = "MultiHopRAG.json"
CORPUS_FILE = "corpus.json"

_SHA256 = {
    QUERIES_FILE: "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff",
    CORPUS_FILE: "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f",
}


@dataclass(frozen=True)
class Evidence:
    title: str
    fact: str


@dataclass(frozen=True)
class QueryRecord:
    query: str
    answer: str
    question_type: str
    evidence: tuple[Evidence, ...]


@dataclass(frozen=True)
class CorpusDocument:
    title: str
    body: str
    source: str
    published_at: str
    url: str


def _download(filename: str) -> Path:
    path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=filename,
            revision=DATASET_REVISION,
            repo_type="dataset",
        )
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _SHA256[filename]:
        raise ValueError(
            f"SHA256 mismatch for {DATASET_REPO}/{filename} at {DATASET_REVISION}: "
            f"expected {_SHA256[filename]}, got {digest}"
        )
    return path


def load_corpus() -> list[CorpusDocument]:
    raw = json.loads(_download(CORPUS_FILE).read_text(encoding="utf-8"))
    return [
        CorpusDocument(
            title=item["title"],
            body=item["body"],
            source=item.get("source", ""),
            published_at=item.get("published_at", ""),
            url=item.get("url", ""),
        )
        for item in raw
    ]


def load_queries() -> list[QueryRecord]:
    raw = json.loads(_download(QUERIES_FILE).read_text(encoding="utf-8"))
    return [
        QueryRecord(
            query=item["query"],
            answer=item["answer"],
            question_type=item["question_type"],
            evidence=tuple(
                Evidence(title=evidence["title"], fact=evidence["fact"])
                for evidence in item["evidence_list"]
            ),
        )
        for item in raw
    ]


def build_dataset(
    queries: Sequence[QueryRecord],
    include_null_queries: bool = False,
) -> Dataset:
    samples = []
    for index, record in enumerate(queries):
        if record.question_type == "null_query" and not include_null_queries:
            continue
        samples.append(
            Sample(
                id=f"multihop-rag-{index}",
                input=record.query,
                target=record.answer,
                metadata={
                    "question_type": record.question_type,
                    "query": record.query,
                    "evidence": [
                        {"title": evidence.title, "fact": evidence.fact}
                        for evidence in record.evidence
                    ],
                },
            )
        )
    return MemoryDataset(samples)

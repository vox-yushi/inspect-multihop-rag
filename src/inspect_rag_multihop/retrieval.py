from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .dataset import CorpusDocument

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    text: str
    title: str
    chunk_index: int


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_RE.split(text.strip()) if sentence.strip()]


def _pack_sentences(
    sentences: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    chunks: list[str] = []
    start = 0
    total = len(sentences)

    while start < total:
        end = start
        size = 0
        while end < total:
            sentence_size = len(sentences[end].split())
            if end > start and size + sentence_size > chunk_size:
                break
            size += sentence_size
            end += 1

        chunks.append(" ".join(sentences[start:end]))
        if end >= total:
            break

        next_start = end
        overlap_size = 0
        while (
            next_start - 1 >= start
            and overlap_size + len(sentences[next_start - 1].split()) <= chunk_overlap
        ):
            next_start -= 1
            overlap_size += len(sentences[next_start].split())

        start = max(next_start, start + 1)

    return chunks


def chunk_document(
    document: CorpusDocument,
    chunk_size: int = 256,
    chunk_overlap: int = 25,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    body_chunks = _pack_sentences(_sentences(document.body), chunk_size, chunk_overlap)
    if not body_chunks:
        body_chunks = [""]

    header = (
        f"Title: {document.title}\n"
        f"Source: {document.source}\n"
        f"Published at: {document.published_at}\n\n"
    )
    return [
        Chunk(text=header + body, title=document.title, chunk_index=index)
        for index, body in enumerate(body_chunks)
    ]


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a BM25 index over an empty corpus")
        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    @classmethod
    def from_corpus(
        cls,
        corpus: Sequence[CorpusDocument],
        chunk_size: int = 256,
        chunk_overlap: int = 25,
    ) -> BM25Index:
        chunks = [
            chunk
            for document in corpus
            for chunk in chunk_document(document, chunk_size, chunk_overlap)
        ]
        return cls(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def search(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]
        return [(self._chunks[index], float(scores[index])) for index in ranked]

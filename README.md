# inspect-multihop-rag

MultiHop-RAG evaluation tasks for [Inspect AI](https://inspect.aisi.org.uk/).
The benchmark contains 2,556 queries over a corpus of 609 news articles; each
answer requires evidence spread across 2-4 documents, and 301 queries are
unanswerable (`null_query`).

This package provides two tasks:

| Task | Samples | Scoring |
| --- | --- | --- |
| `multihop_rag_retrieval` | 2,255 (null queries excluded) | Hits@4, Hits@10, MAP@10, MRR@10 |
| `multihop_rag_qa` | 2,556 (null queries included) | Answer accuracy |

Retrieval is performed with BM25 over sentence-packed chunks. The QA task
retrieves the top 10 chunks, then asks the model to answer from that context.

## Installation

```bash
uv sync
```

## Usage

```bash
# Retrieval metrics (no model calls are needed; the model flag is still required)
uv run inspect eval src/inspect_rag_multihop/tasks.py@multihop_rag_retrieval \
    --model mockllm/model

# Retrieval + QA
uv run inspect eval src/inspect_rag_multihop/tasks.py@multihop_rag_qa \
    --model openai/gpt-4o-mini \
    --limit 20
```

Task options:

| Option | Default | Meaning |
| --- | --- | --- |
| `top_k` | `10` | Passages retrieved per query (`>= 10` for the retrieval task) |
| `chunk_size` | `256` | Maximum chunk size in words |
| `chunk_overlap` | `25` | Sentence overlap between adjacent chunks, in words |

Pass options with `-T`, for example `-T top_k=20 -T chunk_size=512`.

## Dataset

The tasks download `yixuantt/MultiHopRAG` from Hugging Face at a pinned revision
and verify the SHA256 of both files:

- Revision: `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`
- `MultiHopRAG.json`: `03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff`
- `corpus.json`: `20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f`
- License: ODC-BY

## Metrics

Retrieval metrics follow `retrieval_evaluate.py` in the
[official repository](https://github.com/yixuantt/MultiHop-RAG):

- Whitespace and newlines are removed from retrieved passages and gold facts
  before matching.
- A passage is relevant when it contains a gold evidence fact as a substring.
  Each gold fact is counted at most once per query.
- `Hits@4` / `Hits@10`: at least one relevant passage in the top 4 / top 10.
- Average precision at rank `r` is `new_gold_facts_at_r / r`, and `MAP@10` is
  the mean of `sum(AP) / min(num_gold_facts, 10)` over queries.
- `MRR@10` is the reciprocal rank of the first relevant passage.
- `null_query` samples have no evidence and are excluded from the retrieval
  task rather than scored as zero.

QA accuracy mirrors `qa_evaluate.py`: the answer is extracted from
`The answer to the question is "..."` when present, and a prediction is correct
when it shares at least one whitespace-delimited word with the gold answer.
Null queries count in the QA task; their gold answer is
`Insufficient information.`

## Baseline

BM25 retrieval with the default chunking (`chunk_size=256`,
`chunk_overlap=25`, `top_k=10`) on all 2,255 non-null queries:

| Metric | Score |
| --- | --- |
| Hits@4 | 0.753 |
| Hits@10 | 0.896 |
| MAP@10 | 0.315 |
| MRR@10 | 0.605 |

Produced on 2026-11-23 with Inspect AI 0.3.291 using
`inspect eval src/inspect_rag_multihop/tasks.py@multihop_rag_retrieval`.
These numbers validate this implementation; they are not a reproduction of the
paper's token-based chunking or dense retrieval baselines.

## Registering with Inspect Evals

The repository follows the single-eval layout used by the
[Inspect Evals register](https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/register/README.md).
Registration is one issue per task:

1. Push the repository to public GitHub and note a full 40-character commit SHA.
2. Open a **Register Eval Submission** issue with:
   - arXiv URL: `https://arxiv.org/abs/2401.15391v1`
   - Source URL:
     `https://github.com/vox-yushi/inspect-multihop-rag/blob/<sha>/src/inspect_rag_multihop/tasks.py#L116`
     for `multihop_rag_retrieval`, or `#L132` for `multihop_rag_qa`.
3. After the bot opens the register PR, upload full-run logs of two models:
   2,255 samples for retrieval and 2,556 for QA. Small, inexpensive models are
   acceptable.

## Implementation notes

- Chunks are built by packing whole sentences up to `chunk_size` words, with
  sentence-level overlap. This approximates the `SentenceSplitter` used by the
  official BM25 baseline without depending on LlamaIndex.
- Each chunk is prefixed with `Title`, `Source` and `Published at`, matching the
  metadata-inclusive text indexed by the official baseline.
- Only BM25 is implemented. The paper also evaluates dense retrievers and
  rerankers; those are out of scope for this package.
- Retrieval metrics are deterministic, so the retrieval task does not call a
  model. The QA task uses the official prompt from `qa_llama.py`.

## Development

```bash
uv run pytest
uv run ruff check .
```

## Citation

```bibtex
@misc{tang2024multihoprag,
      title={MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries},
      author={Yixuan Tang and Yi Yang},
      year={2024},
      eprint={2401.15391},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
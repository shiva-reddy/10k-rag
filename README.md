# 10k-rag

Hierarchical retrieval-augmented question-answering over 35 SEC 10-K filings: Microsoft, Apple, Alphabet, Amazon, Meta, Nvidia, and Tesla, fiscal years 2021 through 2025.

**Full writeup:** [https://shiva-reddy.github.io/10k-rag/](https://shiva-reddy.github.io/10k-rag/)

The Pages site walks through the corpus, the four-stage pipeline (parse, chunk, embed, retrieve), the metadata-augmented embedding strategy that makes the index entity-aware, parameter comparisons across six tunable knobs, generalization tests on three non-MAG7 filers, sample queries with retrieved children and LLM answers, and a complete configuration reference.

## Setup

```bash
git clone https://github.com/shiva-reddy/10k-rag.git
cd 10k-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Running

The repo ships with the raw HTML, cleaned Markdown, and chunked JSONL committed. Only the HNSW index needs to be rebuilt before the first query:

```bash
python -m src.embed                                    # ~2 min on CPU
python -m src.rag --question "Apple's iPhone revenue in fiscal 2024"
```

To rebuild the entire pipeline from raw HTML:

```bash
python -m src.parse_10k --manifest data/raw-html/manifest.json \
                        --source-dir data/raw-html
python -m src.chunker
python -m src.embed
```

To re-run the parameter comparisons documented in Section 6 of the writeup:

```bash
python scripts/run_all_sweeps.py
```

Tests:

```bash
pytest tests/
```

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Pipeline modules: `parse_10k.py`, `chunker.py`, `embed.py`, `rag.py`, `config_loader.py` |
| `config/default.yaml` | Every tunable parameter for the pipeline |
| `data/raw-html/` | 35 SEC 10-K filings as committed HTML (113 MB) |
| `data/cleaned/` | Parsed Markdown plus per-filing metadata and lineage JSON |
| `data/chunks/` | `children.jsonl` and `parents.jsonl` produced by the chunker |
| `data/chroma/` | Persistent ChromaDB HNSW index (gitignored, regenerable in ~2 min) |
| `data/sweeps/results.json` | Output of the six-knob parameter comparison |
| `data/generalization/results.json` | Parse statistics on BRK.A, JPM, and DDOG |
| `scripts/` | Fetchers (`fetch_*`), the comparison runner (`run_all_sweeps.py`), and the demo-answer capture |
| `docs/` | The Pages site (Jekyll, just-the-docs theme) |
| `tests/` | Tests covering chunker invariants and config validation |

## Reference

For the full discussion of the pipeline, design choices, and parameter studies, see [the Pages site](https://shiva-reddy.github.io/10k-rag/).

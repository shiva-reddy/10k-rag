"""Unified parameter-sweep runner for the 10k-rag pipeline.

Six sweeps, holding every other knob at the default. Five fixed eval queries.

Sweeps that require re-chunk + re-embed:
  1. embed_metadata_prefix ∈ {true, false}
  2. child_max_chars       ∈ {600, 1200, 2400}
  3. table_embedding_text  ∈ {caption_only, caption_first_row_first_col, full_text}

Sweeps that are free at query time (same default index):
  4. hnsw_search_ef        ∈ {10, 50, 200}
  5. top_k_children        ∈ {3, 8, 16}
  6. max_unique_parents    ∈ {2, 4, 8}

Output: data/sweeps/results.json (full top-3 captures) and a printed summary
of top-1 similarities + entity-correctness per (sweep, setting, query).
"""
from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import asdict
from pathlib import Path

import chromadb

from src.chunker import chunk_filing
from src.config_loader import load_config
from src.embed import build_collection
from src.rag import RAGSystem

REPO_ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = REPO_ROOT / "data" / "cleaned"
SWEEP_ROOT = REPO_ROOT / "data" / "sweeps"
SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

EVAL_QUERIES = [
    ("Q1_msft_ai_risk",        "How does Microsoft frame AI risk in its risk factors?",       "MSFT"),
    ("Q2_aapl_iphone_revenue", "Apple's iPhone revenue in fiscal 2024",                       "AAPL"),
    ("Q3_tsla_key_person",     "Tesla's key-person succession risk",                          "TSLA"),
    ("Q4_googl_cloud_segment", "Google Cloud segment performance",                            "GOOGL"),
    ("Q5_cyber_mag7",          "Cybersecurity governance disclosures across MAG7",            None),  # cross-company, no expected entity
]


def with_chunk(cfg, **kw):
    new = dataclasses.replace(cfg.chunking, **kw)
    return dataclasses.replace(cfg, chunking=new)


def with_vdb(cfg, **kw):
    new = dataclasses.replace(cfg.vector_db, **kw)
    return dataclasses.replace(cfg, vector_db=new)


def with_retrieval(cfg, **kw):
    new = dataclasses.replace(cfg.retrieval, **kw)
    return dataclasses.replace(cfg, retrieval=new)


def chunk_all(cfg, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    parents_path = out_dir / "parents.jsonl"
    children_path = out_dir / "children.jsonl"
    n_parents = n_children = n_prose = n_tables = 0
    with open(parents_path, "w") as pf, open(children_path, "w") as cf:
        for md in sorted(CLEANED_DIR.glob("*.md")):
            meta = md.with_suffix(".meta.json")
            if not meta.exists():
                continue
            parents, children = chunk_filing(md, meta, cfg)
            for p in parents:
                pf.write(json.dumps(asdict(p)) + "\n")
            for c in children:
                cf.write(json.dumps(asdict(c)) + "\n")
                if c.chunk_type == "prose":
                    n_prose += 1
                else:
                    n_tables += 1
            n_parents += len(parents)
            n_children += len(children)
    return {"parents": n_parents, "children": n_children, "prose": n_prose, "tables": n_tables}


def query_one(rag: RAGSystem, query: str, top_k: int, where: dict | None = None) -> list[dict]:
    out = []
    for r in rag.retrieve(query, top_k=top_k, where=where):
        out.append({
            "chunk_id": r.chunk_id,
            "ticker": r.metadata.get("ticker"),
            "fy": r.metadata.get("fy_label"),
            "item": r.metadata.get("item"),
            "sub_heading": r.metadata.get("sub_heading") or "",
            "chunk_type": r.metadata.get("chunk_type"),
            "similarity": round(r.similarity, 4),
            "snippet": r.text[:160].replace("\n", " "),
        })
    return out


def run_query_set(rag: RAGSystem, top_k: int) -> list[dict]:
    rows = []
    for qid, q, expected_ticker in EVAL_QUERIES:
        top3 = query_one(rag, q, top_k=top_k)
        rows.append({
            "qid": qid,
            "query": q,
            "expected_ticker": expected_ticker,
            "top3": top3[:3],
            "top1_similarity": top3[0]["similarity"] if top3 else None,
            "top3_entity_match": (
                None if expected_ticker is None
                else sum(1 for c in top3[:3] if c["ticker"] == expected_ticker)
            ),
        })
    return rows


# ─── Sweep 1: embed_metadata_prefix ─────────────────────────────────────────


def sweep_embed_prefix() -> list[dict]:
    rows = []
    for value in [True, False]:
        label = f"S1_embed_prefix_{str(value).lower()}"
        print(f"\n[{label}]  embed_metadata_prefix={value}")
        chunks_dir = SWEEP_ROOT / "chunks" / label
        chroma_dir = SWEEP_ROOT / "chroma" / label
        cfg = load_config()
        cfg = with_chunk(cfg, embed_metadata_prefix=value)
        cfg = with_vdb(cfg, collection_name=f"sweep_{label}")

        t0 = time.time()
        stats = chunk_all(cfg, chunks_dir)
        build_collection(cfg, chunks_dir=chunks_dir, chroma_dir=chroma_dir, force=True, progress=False)
        build_seconds = round(time.time() - t0, 1)

        rag = RAGSystem(cfg=cfg, chroma_dir=chroma_dir, chunks_dir=chunks_dir, llm_backend="none")
        retrievals = run_query_set(rag, top_k=3)
        rows.append({
            "sweep": "embed_metadata_prefix",
            "setting": value,
            "label": label,
            "build_seconds": build_seconds,
            "chunk_stats": stats,
            "retrievals": retrievals,
        })
        print(f"  {label} ✓  ({build_seconds}s)")
    return rows


# ─── Sweep 2: child_max_chars ───────────────────────────────────────────────


def sweep_child_max_chars() -> list[dict]:
    rows = []
    for value in [600, 1200, 2400]:
        label = f"S2_child_max_{value}"
        print(f"\n[{label}]  child_max_chars={value}")
        chunks_dir = SWEEP_ROOT / "chunks" / label
        chroma_dir = SWEEP_ROOT / "chroma" / label
        cfg = load_config()
        cfg = with_chunk(cfg, child_max_chars=value)
        cfg = with_vdb(cfg, collection_name=f"sweep_{label}")

        t0 = time.time()
        stats = chunk_all(cfg, chunks_dir)
        build_collection(cfg, chunks_dir=chunks_dir, chroma_dir=chroma_dir, force=True, progress=False)
        build_seconds = round(time.time() - t0, 1)

        rag = RAGSystem(cfg=cfg, chroma_dir=chroma_dir, chunks_dir=chunks_dir, llm_backend="none")
        retrievals = run_query_set(rag, top_k=3)
        rows.append({
            "sweep": "child_max_chars",
            "setting": value,
            "label": label,
            "build_seconds": build_seconds,
            "chunk_stats": stats,
            "retrievals": retrievals,
        })
        print(f"  {label} ✓  ({build_seconds}s)  children={stats['children']}")
    return rows


# ─── Sweep 3: table_embedding_text ──────────────────────────────────────────


def sweep_table_emb_text() -> list[dict]:
    rows = []
    for value in ["caption_only", "caption_first_row_first_col", "full_text"]:
        label = f"S3_tabletxt_{value}"
        print(f"\n[{label}]  table_embedding_text={value}")
        chunks_dir = SWEEP_ROOT / "chunks" / label
        chroma_dir = SWEEP_ROOT / "chroma" / label
        cfg = load_config()
        cfg = with_chunk(cfg, table_embedding_text=value)
        cfg = with_vdb(cfg, collection_name=f"sweep_{label}")

        t0 = time.time()
        stats = chunk_all(cfg, chunks_dir)
        build_collection(cfg, chunks_dir=chunks_dir, chroma_dir=chroma_dir, force=True, progress=False)
        build_seconds = round(time.time() - t0, 1)

        rag = RAGSystem(cfg=cfg, chroma_dir=chroma_dir, chunks_dir=chunks_dir, llm_backend="none")
        # Capture both unrestricted and table-only retrievals
        retr_unrestricted = run_query_set(rag, top_k=3)
        retr_table_only = []
        for qid, q, expected in EVAL_QUERIES:
            top3 = query_one(rag, q, top_k=3, where={"chunk_type": "table"})
            retr_table_only.append({
                "qid": qid, "query": q, "expected_ticker": expected,
                "top3": top3, "top1_similarity": top3[0]["similarity"] if top3 else None,
            })
        rows.append({
            "sweep": "table_embedding_text",
            "setting": value,
            "label": label,
            "build_seconds": build_seconds,
            "chunk_stats": stats,
            "retrievals_unrestricted": retr_unrestricted,
            "retrievals_table_only": retr_table_only,
        })
        print(f"  {label} ✓  ({build_seconds}s)")
    return rows


# ─── Sweep 4: hnsw_search_ef ────────────────────────────────────────────────


def sweep_hnsw_search_ef() -> list[dict]:
    """Same default index, modify search_ef per setting."""
    rows = []
    base = load_config()
    default_chunks = REPO_ROOT / "data" / "chunks"
    default_chroma = REPO_ROOT / "data" / "chroma"
    client = chromadb.PersistentClient(path=str(default_chroma))
    coll = client.get_collection(base.vector_db.collection_name)

    for value in [10, 50, 200]:
        label = f"S4_search_ef_{value}"
        print(f"\n[{label}]  hnsw_search_ef={value}")
        coll.modify(metadata={"hnsw:search_ef": value})
        cfg = with_vdb(base, hnsw_search_ef=value)
        rag = RAGSystem(cfg=cfg, chroma_dir=default_chroma, chunks_dir=default_chunks, llm_backend="none")
        retrievals = run_query_set(rag, top_k=3)
        rows.append({
            "sweep": "hnsw_search_ef",
            "setting": value,
            "label": label,
            "retrievals": retrievals,
        })
        print(f"  {label} ✓")

    coll.modify(metadata={"hnsw:search_ef": base.vector_db.hnsw_search_ef})
    return rows


# ─── Sweep 5: top_k_children ────────────────────────────────────────────────


def sweep_top_k_children() -> list[dict]:
    """Same default index, vary top-k at query time."""
    rows = []
    base = load_config()
    default_chunks = REPO_ROOT / "data" / "chunks"
    default_chroma = REPO_ROOT / "data" / "chroma"

    for value in [3, 8, 16]:
        label = f"S5_top_k_{value}"
        print(f"\n[{label}]  top_k_children={value}")
        cfg = with_retrieval(base, top_k_children=value)
        rag = RAGSystem(cfg=cfg, chroma_dir=default_chroma, chunks_dir=default_chunks, llm_backend="none")
        # capture the full top-k for each query (not just top-3)
        per_query = []
        for qid, q, expected in EVAL_QUERIES:
            full = query_one(rag, q, top_k=value)
            # Also track unique parents that emerge after expansion
            unique_parents = []
            seen = set()
            for c in full:
                # parent_id isn't in the metadata returned by retrieve(); reconstruct
                # via chunk_id prefix (parent_id is everything before last "-pN" or "-tN")
                cid = c["chunk_id"]
                # remove suffixes like "-p11", "-t3", "-part1", etc.
                parts = cid.rsplit("-", 1)
                pid_guess = parts[0] if parts and (parts[1].startswith("p") or parts[1].startswith("t")) else cid
                if pid_guess not in seen:
                    seen.add(pid_guess)
                    unique_parents.append(pid_guess)
            per_query.append({
                "qid": qid, "query": q, "expected_ticker": expected,
                "top3": full[:3],
                "top_k": value,
                "unique_parents_estimated": len(unique_parents),
                "top1_similarity": full[0]["similarity"] if full else None,
            })
        rows.append({
            "sweep": "top_k_children",
            "setting": value,
            "label": label,
            "retrievals": per_query,
        })
        print(f"  {label} ✓")
    return rows


# ─── Sweep 6: max_unique_parents (free, but we capture it as a config knob) ─


def sweep_max_unique_parents() -> list[dict]:
    """Same default index. Tests how many parents would be selected after expand."""
    rows = []
    base = load_config()
    default_chunks = REPO_ROOT / "data" / "chunks"
    default_chroma = REPO_ROOT / "data" / "chroma"

    for value in [2, 4, 8]:
        label = f"S6_max_parents_{value}"
        print(f"\n[{label}]  max_unique_parents={value}")
        cfg = with_retrieval(base, max_unique_parents=value)
        rag = RAGSystem(cfg=cfg, chroma_dir=default_chroma, chunks_dir=default_chunks, llm_backend="none")
        per_query = []
        for qid, q, expected in EVAL_QUERIES:
            children = rag.retrieve(q, top_k=base.retrieval.top_k_children)
            parents = rag.expand_parents(children, max_unique=value)
            total_chars = sum(len(p.text) for p in parents)
            per_query.append({
                "qid": qid, "query": q, "expected_ticker": expected,
                "top_k_children": base.retrieval.top_k_children,
                "max_unique_parents": value,
                "n_parents": len(parents),
                "total_parent_chars": total_chars,
                "parents": [
                    {
                        "parent_id": p.parent_id,
                        "ticker": p.metadata.get("ticker"),
                        "fy": p.metadata.get("fy_label"),
                        "item": p.metadata.get("item"),
                        "sub_heading": p.metadata.get("sub_heading") or "",
                        "best_similarity": round(p.best_similarity, 4),
                        "len": len(p.text),
                    } for p in parents
                ],
            })
        rows.append({
            "sweep": "max_unique_parents",
            "setting": value,
            "label": label,
            "retrievals": per_query,
        })
        print(f"  {label} ✓")
    return rows


def main():
    print("=" * 60)
    print("Running 6 sweeps × 5 eval queries")
    print("=" * 60)
    all_rows = []
    all_rows.extend(sweep_embed_prefix())
    all_rows.extend(sweep_child_max_chars())
    all_rows.extend(sweep_table_emb_text())
    all_rows.extend(sweep_hnsw_search_ef())
    all_rows.extend(sweep_top_k_children())
    all_rows.extend(sweep_max_unique_parents())

    out_path = SWEEP_ROOT / "results.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\nWrote {out_path.relative_to(REPO_ROOT)}  ({len(all_rows)} configurations)")


if __name__ == "__main__":
    main()

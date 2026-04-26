"""embed.py — embed children, write to ChromaDB.

Pipeline stage 3. Reads data/chunks/children.jsonl, embeds each child's
`embedding_text` with sentence-transformers, stores in a persistent
ChromaDB collection at data/chroma/.

Parents are NOT embedded — they're loaded from parents.jsonl at retrieval
time and looked up by parent_id. This is the assignment's "embed children
only" rule, made literal at the storage layer.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.config_loader import Config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
DEFAULT_CHROMA_DIR = REPO_ROOT / "data" / "chroma"


def _load_children(path: Path) -> list[dict]:
    return [json.loads(L) for L in path.read_text().splitlines() if L.strip()]


def _flatten_metadata(meta: dict) -> dict:
    """ChromaDB metadata must be {str: str|int|float|bool}. Convert None to
    sentinel and stringify any unsupported types."""
    out = {}
    for k, v in meta.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def build_collection(
    cfg: Config,
    chunks_dir: Path = DEFAULT_CHUNKS_DIR,
    chroma_dir: Path = DEFAULT_CHROMA_DIR,
    force: bool = True,
    progress: bool = True,
) -> None:
    """Embed all children and (re)create the ChromaDB collection."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    children_path = chunks_dir / "children.jsonl"
    children = _load_children(children_path)
    if progress:
        print(f"Loaded {len(children):,} children from {children_path}")

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    name = cfg.vector_db.collection_name
    if force:
        try:
            client.delete_collection(name)
            if progress:
                print(f"Deleted existing collection {name!r}")
        except Exception:
            pass

    collection = client.create_collection(
        name=name,
        metadata={
            "hnsw:M": cfg.vector_db.hnsw_M,
            "hnsw:construction_ef": cfg.vector_db.hnsw_construction_ef,
            "hnsw:search_ef": cfg.vector_db.hnsw_search_ef,
            "hnsw:space": cfg.vector_db.distance,
        },
    )
    if progress:
        print(
            f"Created collection {name!r} with HNSW(M={cfg.vector_db.hnsw_M}, "
            f"ef_c={cfg.vector_db.hnsw_construction_ef}, "
            f"ef_s={cfg.vector_db.hnsw_search_ef}, "
            f"space={cfg.vector_db.distance})"
        )

    if progress:
        print(f"Loading embedding model {cfg.embedding.model!r}…")
    model = SentenceTransformer(cfg.embedding.model)

    batch_size = cfg.embedding.batch_size
    n = len(children)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = children[start:end]
        texts = [c["embedding_text"] for c in batch]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=cfg.embedding.normalize_embeddings,
            show_progress_bar=False,
        )
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=[emb.tolist() for emb in embeddings],
            metadatas=[_flatten_metadata({**c["metadata"],
                                          "chunk_type": c["chunk_type"],
                                          "parent_id": c["parent_id"]})
                       for c in batch],
            documents=[c["text"] for c in batch],
        )
        if progress:
            print(f"  embedded + indexed {end:,}/{n:,}", end="\r")

    if progress:
        print()
        # final stats
        count = collection.count()
        print(f"Collection {name!r} now contains {count:,} children")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Embed children and build ChromaDB index.")
    ap.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    ap.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--no-force", action="store_true",
        help="Don't drop existing collection (will error if it exists)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    build_collection(
        cfg=cfg,
        chunks_dir=args.chunks_dir,
        chroma_dir=args.chroma_dir,
        force=not args.no_force,
    )


if __name__ == "__main__":
    main()

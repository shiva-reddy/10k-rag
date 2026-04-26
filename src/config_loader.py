"""Config loader for the 10k-rag pipeline.

Loads YAML config files into typed dataclasses, validates ranges, and
supports in-code overrides for notebook experimentation.

Usage:
    from src.config_loader import load_config

    cfg = load_config()                          # default config
    cfg = load_config("config/strict.yaml")      # variant config
    cfg = load_config(overrides={"chunking": {"child_target_chars": 600}})

    print(cfg.chunking.child_target_chars)       # typed attribute access
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


# ─── Section dataclasses ──────────────────────────────────────────────────


@dataclass
class CleaningConfig:
    items_kept: list[str]
    items_dropped: list[str]
    min_item_chars: int

    table_min_rows: int
    table_min_cols: int
    table_min_text_chars: int
    table_max_nested_depth: int
    table_multirow_header_separator: str
    table_empty_cell_marker: str

    subheading_strategies: list[str]
    bold_subheading_max_chars: int
    bold_subheading_must_be_sole_content: bool
    bold_subheading_must_be_followed_by_para: bool

    drop_xbrl_tags: bool
    drop_hidden_elements: bool
    drop_navigation_chrome: bool


@dataclass
class ChunkingConfig:
    child_target_chars: int
    child_min_chars: int
    child_max_chars: int

    parent_max_chars: int
    use_subheadings_as_parents: bool

    table_as_child: bool
    table_embedding_text: str
    table_oversized_threshold: int

    embed_metadata_prefix: bool = True


@dataclass
class EmbeddingConfig:
    model: str
    batch_size: int
    normalize_embeddings: bool


@dataclass
class VectorDBConfig:
    collection_name: str
    hnsw_M: int
    hnsw_construction_ef: int
    hnsw_search_ef: int
    distance: str


@dataclass
class RetrievalConfig:
    top_k_children: int
    max_unique_parents: int
    metadata_filters: dict[str, Any]


@dataclass
class GenerationConfig:
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int
    citation_style: str


@dataclass
class Config:
    cleaning: CleaningConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_db: VectorDBConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig

    source_path: Path | None = None

    def __repr__(self) -> str:
        src = self.source_path.name if self.source_path else "in-memory"
        return f"Config(source={src})"


SECTION_TYPES = {
    "cleaning": CleaningConfig,
    "chunking": ChunkingConfig,
    "embedding": EmbeddingConfig,
    "vector_db": VectorDBConfig,
    "retrieval": RetrievalConfig,
    "generation": GenerationConfig,
}


# ─── Loader ───────────────────────────────────────────────────────────────


def load_config(
    path: str | Path | None = None,
    overrides: dict | None = None,
) -> Config:
    """Load YAML config and return a validated, typed Config object.

    Args:
        path: Path to YAML. Defaults to config/default.yaml.
        overrides: Dict of section-keyed overrides applied AFTER the file load.
                   Example: {"chunking": {"child_target_chars": 600}}.

    Raises:
        ValueError on schema mismatch or out-of-range values.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    if overrides:
        raw = _deep_merge(raw, overrides)

    sections = {}
    for name, section_cls in SECTION_TYPES.items():
        if name not in raw:
            raise ValueError(f"Config missing required section: {name!r}")
        try:
            sections[name] = _construct_dataclass(section_cls, raw[name])
        except TypeError as e:
            raise ValueError(f"Section {name!r} has invalid fields: {e}") from e

    cfg = Config(**sections, source_path=cfg_path)
    _validate(cfg)
    return cfg


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base. Returns a NEW dict; base unchanged."""
    out = dict(base)
    for k, v in overrides.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _construct_dataclass(cls, data: dict):
    """Construct a dataclass, raising on extra/missing keys."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    expected = {f.name for f in fields(cls)}
    given = set(data.keys())
    missing = expected - given
    extra = given - expected
    if missing:
        raise TypeError(f"missing fields: {sorted(missing)}")
    if extra:
        raise TypeError(f"unexpected fields: {sorted(extra)}")
    return cls(**data)


# ─── Validation ───────────────────────────────────────────────────────────


def _validate(cfg: Config) -> None:
    """Cross-field validation. Raises ValueError for bad combinations."""

    # Items kept/dropped must not overlap
    kept = set(cfg.cleaning.items_kept)
    dropped = set(cfg.cleaning.items_dropped)
    overlap = kept & dropped
    if overlap:
        raise ValueError(
            f"items_kept and items_dropped overlap: {sorted(overlap)}"
        )

    # Child size invariants
    c = cfg.chunking
    if not (c.child_min_chars < c.child_target_chars < c.child_max_chars):
        raise ValueError(
            f"chunking child sizes must satisfy "
            f"child_min_chars ({c.child_min_chars}) "
            f"< child_target_chars ({c.child_target_chars}) "
            f"< child_max_chars ({c.child_max_chars})"
        )
    if c.parent_max_chars < c.child_max_chars:
        raise ValueError(
            f"parent_max_chars ({c.parent_max_chars}) "
            f"must be >= child_max_chars ({c.child_max_chars})"
        )

    # Table thresholds
    cl = cfg.cleaning
    if cl.table_min_rows < 1 or cl.table_min_cols < 1:
        raise ValueError("table_min_rows and table_min_cols must be >= 1")
    if cl.table_min_text_chars < 0:
        raise ValueError("table_min_text_chars must be >= 0")

    # Sub-heading strategies must be from the allowed set
    allowed_strategies = {"html_heading_tags", "bold_isolated", "font_size_hints"}
    bad = set(cl.subheading_strategies) - allowed_strategies
    if bad:
        raise ValueError(
            f"unknown subheading_strategies: {sorted(bad)}; "
            f"allowed: {sorted(allowed_strategies)}"
        )

    # Embedding text strategy
    allowed_emb = {
        "caption_only",
        "caption_first_row",
        "caption_first_row_first_col",
        "full_text",
    }
    if c.table_embedding_text not in allowed_emb:
        raise ValueError(
            f"unknown table_embedding_text: {c.table_embedding_text!r}; "
            f"allowed: {sorted(allowed_emb)}"
        )

    # HNSW params
    v = cfg.vector_db
    if v.hnsw_M < 4 or v.hnsw_M > 64:
        raise ValueError(f"hnsw_M={v.hnsw_M} should be in [4, 64]")
    if v.hnsw_construction_ef < v.hnsw_search_ef:
        raise ValueError(
            "hnsw_construction_ef should be >= hnsw_search_ef "
            "(build is one-time; search is per-query)"
        )
    if v.distance not in {"cosine", "l2", "ip"}:
        raise ValueError(f"unknown distance metric: {v.distance!r}")

    # Retrieval
    r = cfg.retrieval
    if r.top_k_children < 1:
        raise ValueError("top_k_children must be >= 1")
    if r.max_unique_parents < 1:
        raise ValueError("max_unique_parents must be >= 1")
    if r.max_unique_parents > r.top_k_children:
        # Not an error, but unusual — log a hint by raising informatively
        # Actually this is fine: parents can dedupe to fewer than k. Skip.
        pass

    # Generation
    g = cfg.generation
    if not (0.0 <= g.llm_temperature <= 2.0):
        raise ValueError(f"llm_temperature={g.llm_temperature} should be in [0, 2]")
    if g.llm_max_tokens < 1:
        raise ValueError("llm_max_tokens must be >= 1")
    allowed_cite = {"inline_compact", "inline_full", "footnote_numbered"}
    if g.citation_style not in allowed_cite:
        raise ValueError(
            f"unknown citation_style: {g.citation_style!r}; "
            f"allowed: {sorted(allowed_cite)}"
        )


# ─── CLI smoke test ───────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    print(cfg)
    print(f"  child_target_chars: {cfg.chunking.child_target_chars}")
    print(f"  parent_max_chars:   {cfg.chunking.parent_max_chars}")
    print(f"  embedding model:    {cfg.embedding.model}")
    print(f"  HNSW (M, ef_c, ef_s): "
          f"({cfg.vector_db.hnsw_M}, {cfg.vector_db.hnsw_construction_ef}, "
          f"{cfg.vector_db.hnsw_search_ef})")
    print(f"  items kept:         {cfg.cleaning.items_kept}")

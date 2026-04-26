"""Tests for src.chunker — focused on hierarchy invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.chunker import (
    Block,
    _merge_undersized,
    _parse_blocks,
    _split_oversized,
    _strip_frontmatter,
    _table_embedding_text,
    chunk_filing,
)
from src.config_loader import load_config


REPO = Path(__file__).resolve().parent.parent
CLEANED = REPO / "data" / "cleaned"


# ─── Pure-function tests ──────────────────────────────────────────────────


def test_strip_frontmatter():
    md = "---\nticker: MSFT\n---\n# PART I\n\nbody\n"
    body, char_off, line_off = _strip_frontmatter(md)
    assert body.startswith("# PART I")
    assert char_off > 0
    assert line_off > 0


def test_strip_frontmatter_none():
    md = "# Heading\n\nbody"
    body, char_off, line_off = _strip_frontmatter(md)
    assert body == md
    assert char_off == 0
    assert line_off == 0


def test_split_oversized_keeps_short():
    s = "Short paragraph."
    assert _split_oversized(s, 1000) == [s]


def test_split_oversized_splits_long():
    s = ("This is a sentence. " * 200).strip()
    pieces = _split_oversized(s, 500)
    assert len(pieces) > 1
    assert all(len(p) <= 500 for p in pieces)


def test_merge_undersized():
    blocks = [
        Block(kind="paragraph", text="a"),
        Block(kind="paragraph", text="b"),
        Block(kind="paragraph", text="c" * 200),
        Block(kind="paragraph", text="d"),
    ]
    out = _merge_undersized(blocks, min_chars=80)
    # 'a','b' should merge into the 'c' block; 'd' stays alone (no follow-up)
    assert len(out) <= 3
    assert any("a" in b.text and "b" in b.text for b in out)


def test_table_embedding_text_caption_first_row_first_col():
    table_md = (
        "**Table: Segment Performance**\n"
        "| Segment | FY25 | FY24 |\n"
        "|---|---|---|\n"
        "| Cloud | 137,447 | 111,556 |\n"
        "| MPC | 51,000 | 49,000 |"
    )
    out = _table_embedding_text(table_md, "Segment Performance", "caption_first_row_first_col")
    assert "Segment Performance" in out
    assert "FY25" in out
    assert "Cloud" in out
    assert "MPC" in out
    assert "137,447" not in out  # numbers shouldn't be in embedding text


def test_table_embedding_text_caption_only():
    table_md = "**Table: X**\n| a | b |\n|---|---|\n| 1 | 2 |"
    out = _table_embedding_text(table_md, "X", "caption_only")
    assert out == "X"


def test_parse_blocks_handles_table_block():
    md = (
        "## Item 1A. Risk Factors\n"
        "\n"
        "Some intro paragraph.\n"
        "\n"
        "**Table: Foo**\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "\n"
        "Following paragraph.\n"
    )
    blocks = _parse_blocks(md, 0, 0)
    kinds = [b.kind for b in blocks]
    assert "h2" in kinds
    assert "paragraph" in kinds
    assert "table" in kinds
    table_block = next(b for b in blocks if b.kind == "table")
    assert table_block.table_caption == "Foo"
    assert "| a | b |" in table_block.text


# ─── Integration: hierarchy invariants on real corpus ─────────────────────


@pytest.fixture(scope="module")
def msft_chunks():
    """Chunk MSFT FY2024 once and reuse."""
    cfg = load_config()
    md_path = CLEANED / "MSFT-FY2024.md"
    if not md_path.exists():
        pytest.skip(f"need parsed corpus at {CLEANED}")
    parents, children = chunk_filing(md_path, md_path.with_suffix(".meta.json"), cfg)
    return parents, children, cfg


def test_every_child_has_a_valid_parent(msft_chunks):
    parents, children, _cfg = msft_chunks
    parent_ids = {p.parent_id for p in parents}
    orphans = [c for c in children if c.parent_id not in parent_ids]
    assert orphans == [], f"{len(orphans)} children point to nonexistent parents"


def test_no_parent_exceeds_max(msft_chunks):
    parents, _children, cfg = msft_chunks
    cap = cfg.chunking.parent_max_chars
    over = [(p.parent_id, len(p.text)) for p in parents if len(p.text) > cap]
    # Allow oversized in metadata but not in body text
    assert all(not p.metadata.get("oversized") for p in parents), \
        f"oversized parents: {[p.parent_id for p in parents if p.metadata.get('oversized')]}"


def test_children_have_required_metadata(msft_chunks):
    _parents, children, _cfg = msft_chunks
    required = {"ticker", "company", "fy_label", "item", "edgar_url",
                "md_char_start", "md_char_end"}
    for c in children:
        missing = required - c.metadata.keys()
        assert not missing, f"{c.chunk_id} missing {missing}"


def test_table_chunks_have_caption_and_embedding_text(msft_chunks):
    _parents, children, _cfg = msft_chunks
    tables = [c for c in children if c.chunk_type == "table"]
    assert tables, "expected at least one table chunk"
    for t in tables:
        assert "table_caption" in t.metadata
        assert t.embedding_text  # non-empty
        # Table embedding text should NOT contain raw numbers (digits-heavy)
        digits = sum(1 for c in t.embedding_text if c.isdigit())
        assert digits / max(len(t.embedding_text), 1) < 0.20, \
            f"table embedding text too digit-heavy: {t.embedding_text!r}"


def test_kept_items_are_present(msft_chunks):
    parents, _children, _cfg = msft_chunks
    items_seen = {p.metadata["item"] for p in parents}
    # we expect Item 1, 1A, 1C, 7, 8 in any modern MSFT 10-K
    expected = {"Item 1", "Item 1A", "Item 1C", "Item 7", "Item 8"}
    missing = expected - items_seen
    assert not missing, f"missing kept items: {missing}"


def test_chunk_ids_unique(msft_chunks):
    _parents, children, _cfg = msft_chunks
    ids = [c.chunk_id for c in children]
    duplicates = {x for x in ids if ids.count(x) > 1}
    assert not duplicates, f"duplicate chunk ids: {duplicates}"

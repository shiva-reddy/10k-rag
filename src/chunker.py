"""chunker.py — read clean Markdown and emit hierarchical children + parents.

Pipeline stage 2. Reads Markdown produced by parse_10k.py, walks block-by-
block, and emits two JSONL files:

  data/chunks/children.jsonl   — paragraphs + tables (one record per child)
  data/chunks/parents.jsonl    — full Item or sub-section text (one per parent)

Hierarchy contract (the assignment's deliverable):
  • Each CHILD has exactly one parent_id.
  • Children are embedded; parents are not.
  • Parent = the most-specific containing section: a sub-heading region if
    one exists, otherwise the whole Item.
  • At retrieval time the system does ONE hop child→parent. Sub-heading
    nesting in the source document collapses to two levels at retrieval.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config_loader import Config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

ITEM_REGEX = re.compile(r"^Item\s+(\d+[A-C]?)\.?\s*(.*)$", re.IGNORECASE)


# ─── Block model (intermediate representation) ────────────────────────────


@dataclass
class Block:
    """A discrete logical block parsed from clean Markdown."""
    kind: str               # "h1" | "h2" | "h3" | "h4" | "paragraph" | "table"
    text: str               # heading text, paragraph text, or full table markdown
    table_caption: str = ""
    char_start: int = 0
    char_end: int = 0
    line_start: int = 1
    line_end: int = 1


# ─── Output records (what we serialize) ───────────────────────────────────


@dataclass
class Parent:
    parent_id: str
    text: str
    metadata: dict


@dataclass
class Child:
    chunk_id: str
    parent_id: str
    chunk_type: str         # "prose" | "table"
    text: str               # the chunk's content
    embedding_text: str     # what to embed (may differ for tables)
    metadata: dict


# ─── Markdown → Blocks ────────────────────────────────────────────────────


def _strip_frontmatter(md: str) -> tuple[str, int, int]:
    """Skip the YAML frontmatter. Return (body, char_offset, line_offset)."""
    if not md.startswith("---"):
        return md, 0, 0
    end = md.find("\n---\n", 3)
    if end < 0:
        return md, 0, 0
    body_start = end + len("\n---\n")
    return md[body_start:], body_start, md[:body_start].count("\n")


def _line_starts(lines: list[str]) -> list[int]:
    """Char offset of each line (relative to body start)."""
    starts = [0]
    for L in lines[:-1]:
        starts.append(starts[-1] + len(L) + 1)
    return starts


def _parse_blocks(body: str, char_off: int, line_off: int) -> list[Block]:
    """Split clean Markdown into ordered Blocks.

    Recognized block patterns (each separated by a blank line):
      `# X`           heading level 1
      `## X`          heading level 2 (Item N)
      `### X`         heading level 3 (sub-heading)
      `**Table: X**` followed immediately by `| ... |` rows  → table block
      `(omitted: ...)`                                       → ignored
      anything else                                          → paragraph
    """
    lines = body.split("\n")
    starts = _line_starts(lines)
    blocks: list[Block] = []

    i = 0
    n = len(lines)

    def char_range(start_i: int, end_i: int) -> tuple[int, int]:
        cs = char_off + starts[start_i]
        if end_i <= 0 or end_i > len(starts):
            ce = char_off + len(body)
        else:
            last = end_i - 1
            ce = char_off + starts[last] + len(lines[last])
        return cs, ce

    while i < n:
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        # Heading
        m = re.match(r"^(#{1,6})\s+(.+)$", line)
        if m:
            level = len(m.group(1))
            cs, ce = char_range(i, i + 1)
            blocks.append(Block(
                kind=f"h{level}", text=m.group(2).strip(),
                char_start=cs, char_end=ce,
                line_start=line_off + i + 1, line_end=line_off + i + 1,
            ))
            i += 1
            continue

        # Table block: caption line followed by | ... | rows
        if stripped.startswith("**Table:") and stripped.endswith("**"):
            cap = stripped[len("**Table:"):-2].strip(" :")
            start = i
            i += 1
            while i < n and lines[i].lstrip().startswith("|"):
                i += 1
            end = i
            cs, ce = char_range(start, end)
            blocks.append(Block(
                kind="table",
                text="\n".join(lines[start:end]),
                table_caption=cap,
                char_start=cs, char_end=ce,
                line_start=line_off + start + 1, line_end=line_off + end,
            ))
            continue

        # "(omitted: ...)" — emit nothing
        if stripped.startswith("(omitted:"):
            i += 1
            continue

        # Paragraph: continue until blank line, heading, or table caption
        start = i
        while (
            i < n and lines[i].strip()
            and not re.match(r"^#{1,6}\s+", lines[i])
            and not lines[i].strip().startswith("**Table:")
        ):
            i += 1
        end = i
        text = " ".join(L.strip() for L in lines[start:end] if L.strip())
        if text:
            cs, ce = char_range(start, end)
            blocks.append(Block(
                kind="paragraph", text=text,
                char_start=cs, char_end=ce,
                line_start=line_off + start + 1, line_end=line_off + end,
            ))
    return blocks


# ─── Child sizing helpers ─────────────────────────────────────────────────


_SENT_END = re.compile(r"(?<=[\.\!\?])\s+")


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Split a long paragraph at sentence boundaries, packing into <= max_chars."""
    if len(text) <= max_chars:
        return [text]
    sents = _SENT_END.split(text)
    out: list[str] = []
    buf = ""
    for s in sents:
        if not buf:
            buf = s
        elif len(buf) + 1 + len(s) <= max_chars:
            buf = buf + " " + s
        else:
            out.append(buf)
            buf = s
    if buf:
        out.append(buf)
    # If a single "sentence" is itself > max_chars, hard-split on whitespace
    final: list[str] = []
    for chunk in out:
        if len(chunk) <= max_chars:
            final.append(chunk)
        else:
            words = chunk.split(" ")
            buf = ""
            for w in words:
                if not buf:
                    buf = w
                elif len(buf) + 1 + len(w) <= max_chars:
                    buf += " " + w
                else:
                    final.append(buf)
                    buf = w
            if buf:
                final.append(buf)
    return final


def _merge_undersized(prose_blocks: list[Block], min_chars: int) -> list[Block]:
    """Merge consecutive prose blocks each shorter than min_chars."""
    out: list[Block] = []
    pending: Block | None = None
    for b in prose_blocks:
        if pending is None:
            pending = Block(**asdict(b))
            continue
        if len(pending.text) < min_chars:
            pending.text = pending.text + " " + b.text
            pending.char_end = b.char_end
            pending.line_end = b.line_end
        else:
            out.append(pending)
            pending = Block(**asdict(b))
    if pending:
        out.append(pending)
    return out


# ─── Embedding-text helpers ───────────────────────────────────────────────


def _embedding_prefix(company: str, ticker: str, fy_label: str, item: str, sub_heading: str | None) -> str:
    """Metadata prefix prepended to embedding_text so the index itself is
    entity-aware. The prefix puts company name, ticker, fiscal year, Item
    (with title), and sub-heading into the embedded vector, so an entity-
    scoped query like 'Microsoft AI risk' biases toward MSFT chunks without
    a query-time `where` filter.

    The prefix is NOT part of the chunk's `text` field — the LLM still sees
    the verbatim chunk content. Only the embedder sees this prefix.
    """
    parts = [f"{company} ({ticker})", fy_label, item]
    if sub_heading:
        parts.append(sub_heading)
    return " | ".join(parts) + "\n\n"


def _table_embedding_text(table_md: str, caption: str, strategy: str) -> str:
    """Build the text we embed for a table chunk. Numbers add noise; labels
    (caption, headers, row labels) carry the matchable signal."""
    lines = table_md.strip().split("\n")
    # Skip caption line and divider
    pipe_lines = [L for L in lines if L.lstrip().startswith("|")]
    if len(pipe_lines) < 2:
        return f"{caption}".strip() or table_md[:300]

    def parse_row(line: str) -> list[str]:
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        return parts

    header = parse_row(pipe_lines[0])
    body_rows = [parse_row(L) for L in pipe_lines[2:]]  # skip header + |---| divider
    row_labels = [r[0] for r in body_rows if r and r[0]]

    if strategy == "caption_only":
        return caption.strip()
    if strategy == "caption_first_row":
        return " | ".join([caption.strip()] + header).strip(" |")
    if strategy == "caption_first_row_first_col":
        parts = [caption.strip()] + header + row_labels
        return " | ".join(p for p in parts if p).strip(" |")
    if strategy == "full_text":
        return f"{caption}\n{table_md}"
    return caption.strip() or table_md[:300]


# ─── Main: build children + parents per filing ────────────────────────────


def _make_parent_id(ticker: str, fy: str, item: str, sub: str | None) -> str:
    parts = [ticker, fy, item.replace(" ", "")]
    if sub:
        slug = re.sub(r"[^a-z0-9]+", "-", sub.lower()).strip("-")[:40]
        parts.append(slug)
    return "-".join(parts)


def _base_meta(meta: dict) -> dict:
    """Common metadata fields applied to every chunk."""
    return {
        "ticker": meta["ticker"],
        "company": meta["company"],
        "fy_end": meta["fy_end"],
        "fy_label": meta["fy_label"],
        "calendar_year": meta["calendar_year"],
        "filed_date": meta["filed_date"],
        "cik": meta["cik"],
        "accession": meta["accession"],
        "edgar_url": meta["edgar_url"],
        "source_md": f"{meta['ticker']}-{meta['fy_label']}.md",
    }


def chunk_filing(
    md_path: Path,
    meta_path: Path,
    cfg: Config,
) -> tuple[list[Parent], list[Child]]:
    """Read one filing's clean Markdown, return (parents, children)."""
    md_path = Path(md_path)
    meta = json.loads(Path(meta_path).read_text())
    md = md_path.read_text()
    body, char_off, line_off = _strip_frontmatter(md)
    blocks = _parse_blocks(body, char_off, line_off)

    items_kept = set(cfg.cleaning.items_kept)
    base = _base_meta(meta)

    parents: list[Parent] = []
    children: list[Child] = []

    # Disambiguation counter for repeated sub-heading slugs within a filing.
    # MD&A often has multiple "Fiscal Year 2024 Compared..." sections, one
    # per segment. The first gets the bare slug; subsequent get -2, -3, etc.
    base_id_counts: dict[str, int] = {}

    # Walk blocks; group prose+table blocks under their containing
    # (Item or sub-heading) section. When we hit a new Item / sub-heading,
    # flush whatever's in the buffer as a parent + children.

    state = {
        "part": None,
        "item": None,         # e.g., "1A"
        "item_title": None,   # e.g., "Risk Factors"
        "sub_heading": None,  # e.g., "Operational Risks"
    }
    buffer: list[Block] = []  # blocks that belong to the current parent

    def _split_buffer_into_parts(
        blocks_buf: list[Block], max_chars: int,
    ) -> list[list[Block]]:
        """Group blocks so each group's joined text fits under max_chars.
        Greedy pack: append blocks while under cap; start new part on overflow."""
        parts: list[list[Block]] = []
        current: list[Block] = []
        current_chars = 0
        for b in blocks_buf:
            blen = len(b.text) + (2 if current else 0)  # +2 for "\n\n" separator
            if current and current_chars + blen > max_chars:
                parts.append(current)
                current = [b]
                current_chars = len(b.text)
            else:
                current.append(b)
                current_chars += blen
        if current:
            parts.append(current)
        return parts

    def flush() -> None:
        if not state["item"] or state["item"] not in items_kept or not buffer:
            buffer.clear()
            return
        item = state["item"]
        sub = state["sub_heading"]
        candidate_id = _make_parent_id(meta["ticker"], meta["fy_label"], item, sub)
        # Disambiguate repeated sub-heading slugs within this filing
        n_seen = base_id_counts.get(candidate_id, 0)
        base_id_counts[candidate_id] = n_seen + 1
        base_parent_id = candidate_id if n_seen == 0 else f"{candidate_id}-{n_seen + 1}"

        full_text = "\n\n".join(b.text for b in buffer)
        if len(full_text) <= cfg.chunking.parent_max_chars:
            parts = [list(buffer)]
        else:
            parts = _split_buffer_into_parts(list(buffer), cfg.chunking.parent_max_chars)

        for part_idx, part_blocks in enumerate(parts):
            parent_id = base_parent_id
            if len(parts) > 1:
                parent_id = f"{base_parent_id}-part{part_idx + 1}"

            parent_text = "\n\n".join(b.text for b in part_blocks)
            oversized = len(parent_text) > cfg.chunking.parent_max_chars
            parent_meta = {
                **base,
                "part": state["part"],
                "item": f"Item {item}",
                "item_title": state["item_title"],
                "sub_heading": sub,
                "split_part": part_idx + 1 if len(parts) > 1 else None,
                "split_total": len(parts) if len(parts) > 1 else None,
                "md_char_start": part_blocks[0].char_start,
                "md_char_end": part_blocks[-1].char_end,
                "md_line_start": part_blocks[0].line_start,
                "md_line_end": part_blocks[-1].line_end,
                "oversized": oversized,
            }
            parents.append(Parent(parent_id=parent_id, text=parent_text, metadata=parent_meta))
            _emit_children_for(parent_id, part_blocks, state, sub, item)
        buffer.clear()

    def _emit_children_for(
        parent_id: str, part_blocks: list[Block],
        state: dict, sub: str | None, item: str,
    ) -> None:
        """Emit prose + table children for one parent (or parent-part)."""
        prose_blocks = [b for b in part_blocks if b.kind == "paragraph"]
        merged_prose = _merge_undersized(prose_blocks, cfg.chunking.child_min_chars)

        # Build a single ordered list of (kind, block) walking part_blocks,
        # using merged_prose for prose entries.
        merged_iter = iter(merged_prose)
        ordered: list[tuple[str, Block]] = []
        prose_consumed = 0
        for b in part_blocks:
            if b.kind == "paragraph":
                if prose_consumed < len(merged_prose):
                    ordered.append(("prose", merged_prose[prose_consumed]))
                    prose_consumed += 1
            elif b.kind == "table":
                ordered.append(("table", b))

        seen: set[int] = set()
        para_idx = 0
        table_idx = 0
        for kind, blk in ordered:
            if id(blk) in seen:
                continue
            seen.add(id(blk))
            if kind == "prose":
                pieces = _split_oversized(blk.text, cfg.chunking.child_max_chars)
                for k, piece in enumerate(pieces):
                    para_idx += 1
                    cid = f"{parent_id}-p{para_idx}"
                    if cfg.chunking.embed_metadata_prefix:
                        prefix = _embedding_prefix(
                            base["company"], base["ticker"], base["fy_label"],
                            f"Item {item}", sub,
                        )
                        emb_text = prefix + piece
                    else:
                        emb_text = piece
                    children.append(Child(
                        chunk_id=cid,
                        parent_id=parent_id,
                        chunk_type="prose",
                        text=piece,
                        embedding_text=emb_text,
                        metadata={
                            **base,
                            "part": state["part"],
                            "item": f"Item {item}",
                            "item_title": state["item_title"],
                            "sub_heading": sub,
                            "paragraph_index": para_idx,
                            "split_part": k if len(pieces) > 1 else None,
                            "split_total": len(pieces) if len(pieces) > 1 else None,
                            "md_char_start": blk.char_start,
                            "md_char_end": blk.char_end,
                            "md_line_start": blk.line_start,
                            "md_line_end": blk.line_end,
                        },
                    ))
            else:
                table_idx += 1
                cid = f"{parent_id}-t{table_idx}"
                emb_text = _table_embedding_text(
                    blk.text, blk.table_caption,
                    cfg.chunking.table_embedding_text,
                )
                if cfg.chunking.embed_metadata_prefix:
                    prefix = _embedding_prefix(
                        base["company"], base["ticker"], base["fy_label"],
                        f"Item {item}", sub,
                    )
                    emb_text = prefix + emb_text
                table_oversized = len(blk.text) > cfg.chunking.table_oversized_threshold
                children.append(Child(
                    chunk_id=cid,
                    parent_id=parent_id,
                    chunk_type="table",
                    text=blk.text,
                    embedding_text=emb_text,
                    metadata={
                        **base,
                        "part": state["part"],
                        "item": f"Item {item}",
                        "item_title": state["item_title"],
                        "sub_heading": sub,
                        "table_caption": blk.table_caption,
                        "table_oversized": table_oversized,
                        "md_char_start": blk.char_start,
                        "md_char_end": blk.char_end,
                        "md_line_start": blk.line_start,
                        "md_line_end": blk.line_end,
                    },
                ))

    for b in blocks:
        if b.kind == "h1":
            # PART X
            flush()
            state["part"] = b.text
            state["item"] = None
            state["item_title"] = None
            state["sub_heading"] = None
        elif b.kind == "h2":
            # Item NN. Title
            flush()
            m = ITEM_REGEX.match(b.text)
            if m:
                state["item"] = m.group(1).upper()
                state["item_title"] = m.group(2).strip() or None
                state["sub_heading"] = None
        elif b.kind in ("h3", "h4", "h5", "h6"):
            # Sub-heading inside an Item
            flush()
            if state["item"] and cfg.chunking.use_subheadings_as_parents:
                state["sub_heading"] = b.text
            elif state["item"]:
                # don't change parent, but include the sub-heading text as
                # a paragraph inside the current parent
                buffer.append(b)
        elif b.kind in ("paragraph", "table"):
            if state["item"] and state["item"] in items_kept:
                buffer.append(b)

    flush()

    return parents, children


# ─── CLI entry point ──────────────────────────────────────────────────────


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Chunk clean Markdown into hierarchical children + parents.")
    ap.add_argument(
        "--cleaned-dir", type=Path,
        default=REPO_ROOT / "data" / "cleaned",
        help="Directory of {TICKER}-FY{YEAR}.md + .meta.json files",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "data" / "chunks",
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--filter", type=str, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(args.cleaned_dir.glob("*.md"))
    if args.filter:
        md_files = [f for f in md_files if args.filter in f.name]

    parents_out = args.output_dir / "parents.jsonl"
    children_out = args.output_dir / "children.jsonl"

    n_parents = 0
    n_children = 0
    n_prose = 0
    n_tables = 0
    parent_size_violations = 0
    children_per_parent = []

    print(f"Chunking {len(md_files)} filings → {args.output_dir}")
    with open(parents_out, "w") as pf, open(children_out, "w") as cf:
        for md_path in md_files:
            meta_path = md_path.with_suffix(".meta.json")
            if not meta_path.exists():
                continue
            parents, children = chunk_filing(md_path, meta_path, cfg)
            for p in parents:
                pf.write(json.dumps(asdict(p)) + "\n")
                if p.metadata.get("oversized"):
                    parent_size_violations += 1
            for c in children:
                cf.write(json.dumps(asdict(c)) + "\n")
                if c.chunk_type == "prose":
                    n_prose += 1
                else:
                    n_tables += 1
            # Children per parent
            from collections import Counter
            counts = Counter(c.parent_id for c in children)
            children_per_parent.extend(counts.values())
            n_parents += len(parents)
            n_children += len(children)
            print(
                f"  ✓ {md_path.stem:18s}  "
                f"{len(parents):3d} parents, {len(children):4d} children "
                f"({sum(1 for c in children if c.chunk_type=='prose')} prose / "
                f"{sum(1 for c in children if c.chunk_type=='table')} tables)"
            )

    print()
    print(f"Wrote {n_parents} parents to {parents_out.name}")
    print(f"Wrote {n_children} children to {children_out.name}")
    print(f"  Prose: {n_prose}   Tables: {n_tables}")
    if children_per_parent:
        avg = sum(children_per_parent) / len(children_per_parent)
        print(f"  Children per parent: min={min(children_per_parent)}, "
              f"avg={avg:.1f}, max={max(children_per_parent)}")
    print(f"  Oversized parents: {parent_size_violations}")


if __name__ == "__main__":
    main()

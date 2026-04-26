"""parse_10k.py — clean an SEC 10-K HTML filing into structured Markdown.

Pipeline stage 1. Reads a single 10-K HTML file (with manifest metadata),
walks the DOM to extract a typed token stream (headings, paragraphs,
tables), detects the content occurrence of each SEC Item, drops the
boilerplate Items, and emits:

  data/cleaned/{TICKER}-FY{YEAR}.md          — clean Markdown (interim format)
  data/cleaned/{TICKER}-FY{YEAR}.meta.json   — frontmatter as JSON
  data/cleaned/{TICKER}-FY{YEAR}.lineage.json — paragraph/table → char offset map

The resulting Markdown is the input format for chunker.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup, NavigableString, Tag

from src.config_loader import Config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent

# ─── Regexes ──────────────────────────────────────────────────────────────

# Match "Item 1A.", "Item 1A:", "Item 1A —", "Item  1A", "ITEM 1A" etc.
# Anchored to start of (stripped) text to avoid in-prose mentions.
ITEM_REGEX = re.compile(
    r"^\s*Item\s+(\d{1,2}[A-C]?)\b\s*[\.\:\—\-]?\s*(.*)$",
    re.IGNORECASE,
)
PART_REGEX = re.compile(r"^\s*PART\s+([IV]+)\b\s*[\.\:]?\s*$", re.IGNORECASE)

# Standard Item titles (for filling in when the heading is just "Item 1A.")
STANDARD_TITLES = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters and Issuer Purchases of Equity Securities",
    "6": "[Reserved]",
    "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director Independence",
    "14": "Principal Accountant Fees and Services",
    "15": "Exhibit and Financial Statement Schedules",
    "16": "Form 10-K Summary",
}


# ─── Token model ──────────────────────────────────────────────────────────


@dataclass
class Token:
    """A typed text element extracted from the DOM in document order."""
    kind: str               # "heading" | "bold_iso" | "paragraph" | "table"
    text: str               # canonical text representation (for prose)
    level: int = 0          # for heading: 1-6
    table_md: str = ""      # for table: markdown serialization
    table_caption: str = ""
    table_rows: int = 0
    table_cols: int = 0

    # populated by the annotation pass
    part: str | None = None         # e.g., "PART I"
    item_number: str | None = None  # e.g., "1A"
    is_part_heading: bool = False
    is_item_heading: bool = False
    is_subheading: bool = False


@dataclass
class LineageEntry:
    kind: str            # "part" | "item" | "subheading" | "paragraph" | "table"
    char_start: int
    char_end: int
    line_start: int
    line_end: int
    item: str | None = None
    sub_heading: str | None = None
    paragraph_index: int | None = None
    table_caption: str | None = None


@dataclass
class ParseResult:
    md_path: Path
    meta_path: Path
    lineage_path: Path
    stats: dict


# ─── Step 1: Load + clean DOM ─────────────────────────────────────────────


def _load_and_clean(html_path: Path, cfg: Config) -> BeautifulSoup:
    raw = html_path.read_bytes().decode("utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")

    # Strip wholesale
    for tag in soup.find_all(["script", "style", "head", "noscript", "meta", "link", "title"]):
        tag.decompose()
    # XBRL inline tags: drop the tag wrapper but keep inner text
    if cfg.cleaning.drop_xbrl_tags:
        for tag in soup.find_all(lambda t: t.name and t.name.startswith("ix:")):
            tag.unwrap()
    # Hidden elements
    if cfg.cleaning.drop_hidden_elements:
        for tag in soup.find_all(attrs={"hidden": True}):
            tag.decompose()
        for tag in soup.find_all(style=re.compile(r"display\s*:\s*none", re.I)):
            tag.decompose()
    # Comments
    from bs4 import Comment
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    return soup


# ─── Step 2: Table classification + serialization ─────────────────────────


def _classify_table(table: Tag, cfg: Config) -> str:
    """Classify a <table> as one of:
      "data"   — real financial/data table; serialize as markdown chunk
      "layout" — used as a layout container for prose/headings; unwrap to flow
      "skip"   — empty / page chrome; drop entirely

    Many SEC filers (AMZN, AAPL, GOOGL, META, NVDA, TSLA) wrap section
    headings AND prose paragraphs in <table> elements for visual layout,
    not for tabular data. We must unwrap those, not decompose them.
    """
    rows = table.find_all("tr")
    n_rows = len(rows)
    cells = table.find_all(["td", "th"])
    n_cells = len(cells)
    text = table.get_text(" ", strip=True)

    # Truly empty
    if n_cells == 0 or not text.strip():
        return "skip"

    # 1×1 table: a wrapper for a heading or single block. Unwrap.
    if n_cells <= 1:
        return "layout"

    # Short total text (e.g., <80 chars): heading-wrapper tables. Unwrap.
    # AMZN, AAPL, GOOGL et al. wrap each Item heading in a small 2×N table.
    if len(text) < cfg.cleaning.table_min_text_chars:
        return "layout"

    # Width threshold: data tables almost always have ≥3 columns
    max_cols = max((len(tr.find_all(["td", "th"])) for tr in rows), default=0)
    if max_cols < cfg.cleaning.table_min_cols:
        return "layout"

    # Item-heading wrapper: tables that BEGIN with "Item N." or "PART N" are
    # heading containers, not data. AMZN/AAPL use 2-cell tables this way.
    if ITEM_REGEX.match(text) or PART_REGEX.match(text):
        return "layout"

    # Numerical density — data tables have many digits (years, $ amounts, %)
    digits = sum(1 for c in text if c.isdigit())
    digit_ratio = digits / max(len(text), 1)

    # Heuristic: layout tables tend to be either small (<100 chars per cell
    # average, no numbers) or huge (full-page text grid with no digits).
    avg_cell_chars = len(text) / max(n_cells, 1)
    if digit_ratio < 0.03 and avg_cell_chars > 100:
        return "layout"
    if digit_ratio < 0.015:
        return "layout"

    # Default: it's a data table
    return "data"


def _unwrap_table(table: Tag, soup: BeautifulSoup) -> None:
    """Replace <table> with a flat sequence of <div>s carrying each cell's
    contents. Preserves DOM order so the surrounding walker sees the cells
    as inline block elements.
    """
    container = soup.new_tag("div")
    container["data-unwrapped-table"] = "1"
    for cell in table.find_all(["td", "th"]):
        cell_div = soup.new_tag("div")
        # copy children of the cell into the new div
        for child in list(cell.children):
            cell_div.append(child.extract() if hasattr(child, "extract") else child)
        if cell_div.get_text(strip=True):
            container.append(cell_div)
    table.replace_with(container)


def _is_content_table(table: Tag, cfg: Config) -> bool:
    """Backwards-compat shim: True iff classify == 'data'."""
    return _classify_table(table, cfg) == "data"


def _md_escape(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ").strip()


def _is_spacer_cell(text: str, empty_marker: str) -> bool:
    """SEC tables pad with empty cells and standalone $/(/)/% as visual spacers."""
    t = text.strip()
    return t in ("", empty_marker, "$", "(", ")", "%", ":", ";")


def _table_to_markdown(table: Tag, cfg: Config) -> tuple[str, int, int]:
    """Convert HTML table to pipe-syntax markdown. Returns (md, rows, cols).

    SEC financial tables typically embed empty <td>s as visual spacers
    (e.g., between a "$" cell and the numeric cell). We prune columns
    that are entirely spacers in a post-pass.
    """
    rows = table.find_all("tr")
    if not rows:
        return "", 0, 0

    header_rows: list[list[str]] = []
    body_rows: list[list[str]] = []
    in_header = True
    for tr in rows:
        cells_tags = tr.find_all(["td", "th"], recursive=False)
        if not cells_tags:
            continue
        cells = [
            re.sub(r"\s+", " ", c.get_text(" ", strip=True))
            or cfg.cleaning.table_empty_cell_marker
            for c in cells_tags
        ]
        is_header_row = all((c.name or "").lower() == "th" for c in cells_tags)
        if in_header and is_header_row:
            header_rows.append(cells)
        else:
            in_header = False
            body_rows.append(cells)

    # Drop rows that are entirely empty/spacer (visual padding rows in SEC
    # tables — they precede the real header in financial statements).
    empty_marker = cfg.cleaning.table_empty_cell_marker
    def _row_is_empty(row):
        return all(_is_spacer_cell(c, empty_marker) for c in row)
    header_rows = [r for r in header_rows if not _row_is_empty(r)]
    body_rows = [r for r in body_rows if not _row_is_empty(r)]

    if not header_rows and body_rows:
        header_rows = [body_rows[0]]
        body_rows = body_rows[1:]

    if not header_rows:
        return "", 0, 0

    # Pad all rows to the same width
    all_rows = header_rows + body_rows
    max_cols = max(len(r) for r in all_rows)
    for r in all_rows:
        if len(r) < max_cols:
            r += [empty_marker] * (max_cols - len(r))

    # Prune columns whose BODY values are entirely spacers. Header presence
    # alone doesn't justify keeping a column — SEC tables routinely have
    # year labels above all-empty spacer columns.
    keep_cols = []
    for c in range(max_cols):
        has_body_content = any(
            not _is_spacer_cell(r[c], empty_marker) for r in body_rows
        )
        if has_body_content:
            keep_cols.append(c)

    # If body is empty, fall back to keeping any header-content columns
    if not keep_cols and header_rows:
        for c in range(max_cols):
            if any(not _is_spacer_cell(r[c], empty_marker) for r in header_rows):
                keep_cols.append(c)

    if not keep_cols:
        return "", 0, 0

    header_rows = [[r[c] for c in keep_cols] for r in header_rows]
    body_rows = [[r[c] for c in keep_cols] for r in body_rows]

    # Detect "unit annotation" first row (e.g. "(In millions)") and treat as
    # part of the caption rather than the header. Heuristic: if row 0 has only
    # one non-empty cell, it's an annotation.
    if header_rows:
        first = header_rows[0]
        non_empty = [c for c in first if not _is_spacer_cell(c, empty_marker)]
        if len(non_empty) == 1 and len(header_rows) > 1:
            # demote this row out of the header — we won't use it (caption
            # extraction happens elsewhere; this row will fall away)
            header_rows = header_rows[1:]

    if len(header_rows) > 1:
        sep = cfg.cleaning.table_multirow_header_separator
        flat = []
        for col in range(len(keep_cols)):
            parts = [
                r[col].strip()
                for r in header_rows
                if not _is_spacer_cell(r[col], empty_marker)
            ]
            flat.append(sep.join(parts) if parts else empty_marker)
        header = flat
    elif header_rows:
        header = header_rows[0]
    else:
        header = [empty_marker] * len(keep_cols)

    lines = ["| " + " | ".join(_md_escape(c) for c in header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")
    for r in body_rows:
        lines.append("| " + " | ".join(_md_escape(c) for c in r) + " |")

    return "\n".join(lines), len(header_rows) + len(body_rows), len(keep_cols)


def _extract_table_caption(table: Tag) -> str:
    """Walk back in DOM to find nearest preceding short heading-style text.

    Cap at 120 chars so prose paragraphs immediately above the table aren't
    mistaken for captions. We prefer text that looks like a heading: short,
    not ending with sentence punctuation.
    """
    MAX = 120
    for ancestor in [table] + list(table.parents)[:2]:
        prev = ancestor.previous_sibling
        steps = 0
        while prev is not None and steps < 8:
            if isinstance(prev, NavigableString):
                t = prev.strip()
                if t and 0 < len(t) <= MAX and not t.endswith((".", "!", "?")):
                    return t
            else:
                t = prev.get_text(" ", strip=True)
                if t and len(t) <= MAX and not t.endswith((".", "!", "?")):
                    if (prev.name or "").lower() in (
                        "h1", "h2", "h3", "h4", "h5", "h6", "b", "strong", "p", "div", "span"
                    ):
                        return t
            prev = prev.previous_sibling
            steps += 1
    return ""


# ─── Step 3: DOM walk → token stream ──────────────────────────────────────


def _has_bold_style(style: str | None) -> bool:
    """True if an inline CSS style declares bold-class font weight."""
    s = (style or "").lower().replace(" ", "")
    return any(f"font-weight:{x}" in s for x in ("bold", "600", "700", "800", "900"))


def _is_bold_isolated(elem: Tag, cfg: Config) -> bool:
    """Element looks like a section heading marked via bold/strong styling.

    Detects three patterns:
      • <p><b>Heading</b></p>                    (semantic bold tag)
      • <p style="font-weight:bold">Heading</p>  (CSS on the parent block)
      • <p><span style="font-weight:bold">Heading</span></p>  (CSS on inner span)
    """
    if not cfg.cleaning.bold_subheading_must_be_sole_content:
        return False

    text = elem.get_text(" ", strip=True)
    if not text or len(text) > cfg.cleaning.bold_subheading_max_chars:
        return False

    # Pattern 1: bold style on the block element itself
    if _has_bold_style(elem.get("style")):
        return True

    # Patterns 2 & 3: must have exactly one meaningful child
    children = [
        c for c in elem.children
        if not (isinstance(c, NavigableString) and not c.strip())
    ]
    if len(children) != 1:
        return False
    only = children[0]
    if isinstance(only, NavigableString):
        return False

    name = (only.name or "").lower()
    if name in ("b", "strong"):
        return True
    # Inline-style bold inside a span/font wrapper (the SEC template default)
    if name in ("span", "font", "em", "i") and _has_bold_style(only.get("style")):
        return True

    return False


def _is_leaf_block(elem: Tag) -> bool:
    """Has text content but no nested block-level elements with their own paragraphs.

    Note: table-marker spans (data-table-idx) count as block-level — we must
    recurse to emit them, even though they have no text themselves.
    """
    BLOCKY = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol", "section", "article"}
    for desc in elem.descendants:
        if isinstance(desc, NavigableString):
            continue
        n = (desc.name or "").lower()
        if n in BLOCKY:
            return False
        if n == "span" and desc.get("data-table-idx") is not None:
            return False
        if n == "div":
            t = desc.get_text(strip=True)
            if t and len(t) > 30:
                return False
    return True


def _walk_tokens(soup: BeautifulSoup, cfg: Config) -> list[Token]:
    """Walk the DOM in document order, emit Token list."""
    body = soup.body or soup

    # Pass 1: classify each <table>:
    #   "data"   → serialize as markdown, replace with marker
    #   "layout" → unwrap (cells become divs flowing in DOM order)
    #   "skip"   → decompose
    # We must process tables OUTERMOST first so unwrapping doesn't disturb
    # nested layout tables we haven't classified yet.
    tables: list[dict] = []
    # snapshot to a list so we can iterate while mutating the tree
    all_tables = list(body.find_all("table"))
    # iterate from outermost to innermost (DOM order is fine for our purposes,
    # since nested data tables are rare in 10-Ks)
    for table in all_tables:
        if not table.parent:
            continue  # already removed/unwrapped
        kind = _classify_table(table, cfg)
        if kind == "skip":
            table.decompose()
            continue
        if kind == "layout":
            _unwrap_table(table, soup)
            continue
        # data table
        caption = _extract_table_caption(table)
        md, rows, cols = _table_to_markdown(table, cfg)
        if not md:
            table.decompose()
            continue
        idx = len(tables)
        tables.append(dict(caption=caption, md=md, rows=rows, cols=cols))
        marker = soup.new_tag("span")
        marker["data-table-idx"] = str(idx)
        table.replace_with(marker)

    # Pass 2: walk what remains
    tokens: list[Token] = []

    def walk(elem):
        if isinstance(elem, NavigableString):
            return
        name = (elem.name or "").lower()

        # Table marker
        if name == "span" and elem.get("data-table-idx") is not None:
            t = tables[int(elem["data-table-idx"])]
            tokens.append(Token(
                kind="table",
                text=t["md"],
                table_md=t["md"],
                table_caption=t["caption"],
                table_rows=t["rows"],
                table_cols=t["cols"],
            ))
            return

        # Real heading tags
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = elem.get_text(" ", strip=True)
            if text:
                tokens.append(Token(kind="heading", text=text, level=int(name[1])))
            return

        # Paragraph
        if name == "p":
            text = elem.get_text(" ", strip=True)
            if text:
                if _is_bold_isolated(elem, cfg):
                    tokens.append(Token(kind="bold_iso", text=text))
                else:
                    tokens.append(Token(kind="paragraph", text=text))
            return

        # Block-level wrappers — emit as paragraph if leaf, else recurse
        if name in ("div", "section", "article", "td"):
            if _is_leaf_block(elem):
                text = elem.get_text(" ", strip=True)
                if text:
                    if _is_bold_isolated(elem, cfg):
                        tokens.append(Token(kind="bold_iso", text=text))
                    else:
                        tokens.append(Token(kind="paragraph", text=text))
                return

        # Recurse
        for child in elem.children:
            walk(child)

    walk(body)
    return tokens


# ─── Step 4: Annotate (Items, Parts, sub-headings) ────────────────────────


def _promote_heading_paragraphs(tokens: list[Token]) -> None:
    """Promote paragraph tokens that look like Item/Part headings.

    SEC filings often use plain styled <p> for section headings; semantic
    <h1>-<h6> tags are rare. We rescue those: any short paragraph whose
    text matches ITEM_REGEX or PART_REGEX is reclassified as a heading.
    """
    for tok in tokens:
        if tok.kind != "paragraph":
            continue
        text = tok.text.strip()
        if len(text) > 200:
            continue
        if ITEM_REGEX.match(text) or PART_REGEX.match(text):
            tok.kind = "heading"


def _detect_content_items(tokens: list[Token]) -> dict[str, int]:
    """Pick the content (not TOC) occurrence of each Item heading.

    Heuristic: among all heading-class tokens whose text matches Item N,
    the content occurrence is the one with the LARGEST gap (in token
    indices) to the next Item-match anywhere. TOC entries cluster tightly.
    """
    candidates: list[tuple[int, str]] = []  # (token_index, item_number)
    for i, tok in enumerate(tokens):
        if tok.kind not in ("heading", "bold_iso"):
            continue
        m = ITEM_REGEX.match(tok.text.strip())
        if m:
            candidates.append((i, m.group(1).upper()))

    by_item: dict[str, tuple[int, int]] = {}  # item_num → (idx, gap)
    for k, (idx, item_num) in enumerate(candidates):
        next_idx = candidates[k + 1][0] if k + 1 < len(candidates) else len(tokens)
        gap = next_idx - idx
        prev = by_item.get(item_num)
        if prev is None or gap > prev[1]:
            by_item[item_num] = (idx, gap)

    return {item_num: idx for item_num, (idx, _) in by_item.items()}


def _detect_content_parts(tokens: list[Token]) -> dict[str, int]:
    """Same idea as Items but for PART I/II/III/IV."""
    candidates: list[tuple[int, str]] = []
    for i, tok in enumerate(tokens):
        if tok.kind not in ("heading", "bold_iso"):
            continue
        m = PART_REGEX.match(tok.text.strip())
        if m:
            candidates.append((i, m.group(1).upper()))

    by_part: dict[str, tuple[int, int]] = {}
    for k, (idx, part_num) in enumerate(candidates):
        next_idx = candidates[k + 1][0] if k + 1 < len(candidates) else len(tokens)
        gap = next_idx - idx
        prev = by_part.get(part_num)
        if prev is None or gap > prev[1]:
            by_part[part_num] = (idx, gap)

    return {part_num: idx for part_num, (idx, _) in by_part.items()}


def _annotate(tokens: list[Token], items: dict[str, int], parts: dict[str, int]) -> None:
    """Tag each token with its containing Part and Item, and mark sub-headings."""
    parts_in_order = sorted(parts.items(), key=lambda x: x[1])
    items_in_order = sorted(items.items(), key=lambda x: x[1])

    # Tag part regions
    for k, (part_num, start) in enumerate(parts_in_order):
        end = parts_in_order[k + 1][1] if k + 1 < len(parts_in_order) else len(tokens)
        for j in range(start, end):
            tokens[j].part = f"PART {part_num}"
        tokens[start].is_part_heading = True

    # Tag item regions
    for k, (item_num, start) in enumerate(items_in_order):
        end = items_in_order[k + 1][1] if k + 1 < len(items_in_order) else len(tokens)
        for j in range(start, end):
            tokens[j].item_number = item_num
        tokens[start].is_item_heading = True
        # mark sub-headings within this Item
        for j in range(start + 1, end):
            t = tokens[j]
            if t.kind in ("heading", "bold_iso"):
                if not ITEM_REGEX.match(t.text.strip()) and not PART_REGEX.match(t.text.strip()):
                    t.is_subheading = True


# ─── Step 5: Emit Markdown + lineage ──────────────────────────────────────


def _build_frontmatter(manifest: dict, stats: dict) -> str:
    fm = {
        "ticker": manifest["ticker"],
        "company": manifest["company"],
        "fy_end": manifest["fy_end"],
        "fy_label": f"FY{manifest['fy_end'][:4]}",
        "calendar_year": int(manifest["fy_end"][:4]),
        "filed_date": manifest["filed"],
        "cik": manifest["cik"],
        "accession": manifest["accession"],
        "edgar_url": _edgar_url(manifest),
        **{f"stats_{k}": v for k, v in stats.items()},
    }
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            lines.append(f"{k}: {json.dumps(v)}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines)


def _edgar_url(manifest: dict) -> str:
    cik_int = str(int(manifest["cik"]))
    acc_nodash = manifest["accession"].replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{manifest['doc']}"


def _emit_markdown(
    tokens: list[Token],
    manifest: dict,
    cfg: Config,
) -> tuple[str, list[LineageEntry], dict]:
    """Emit Markdown text and lineage records. Returns (md_text, lineage, stats)."""
    items_kept = set(cfg.cleaning.items_kept)
    items_dropped = set(cfg.cleaning.items_dropped)

    out: list[str] = []
    lineage: list[LineageEntry] = []
    chars_emitted = 0
    line_no = 1

    def emit(text: str, lineage_entry: LineageEntry | None = None):
        nonlocal chars_emitted, line_no
        start_char = chars_emitted
        start_line = line_no
        out.append(text)
        # update counters
        n_lines = text.count("\n")
        chars_emitted += len(text)
        line_no += n_lines
        if lineage_entry is not None:
            lineage_entry.char_start = start_char
            lineage_entry.char_end = chars_emitted
            lineage_entry.line_start = start_line
            lineage_entry.line_end = line_no
            lineage.append(lineage_entry)

    # Frontmatter (stats updated at the end with a placeholder approach)
    # Simpler: emit a placeholder, fill later. But we actually compute stats AFTER
    # emit. So emit frontmatter at the end and prepend.

    current_part: str | None = None
    current_item: str | None = None
    current_subheading: str | None = None
    paragraph_index_in_subsection = 0
    skip_until_next_item = False
    items_kept_seen: set[str] = set()
    items_dropped_seen: set[str] = set()
    tables_kept = 0
    subheadings_kept = 0

    for tok in tokens:
        # ─ Part heading ─
        if tok.is_part_heading:
            current_part = tok.part or ""
            current_item = None
            current_subheading = None
            paragraph_index_in_subsection = 0
            skip_until_next_item = False
            entry = LineageEntry(kind="part", char_start=0, char_end=0, line_start=0, line_end=0)
            emit(f"\n# {current_part}\n\n", entry)
            continue

        # ─ Item heading ─
        if tok.is_item_heading:
            item_num = tok.item_number or ""
            current_item = item_num
            current_subheading = None
            paragraph_index_in_subsection = 0
            title = STANDARD_TITLES.get(item_num, "")
            heading_line = f"## Item {item_num}. {title}".rstrip() + "\n\n"

            if item_num in items_dropped:
                items_dropped_seen.add(item_num)
                skip_until_next_item = True
                emit(heading_line)
                emit("(omitted: incorporated by reference or boilerplate)\n\n")
                continue

            if item_num in items_kept:
                items_kept_seen.add(item_num)
                skip_until_next_item = False
                entry = LineageEntry(
                    kind="item", item=item_num,
                    char_start=0, char_end=0, line_start=0, line_end=0,
                )
                emit(heading_line, entry)
            else:
                # Unrecognized Item — keep it but flag in stats
                skip_until_next_item = False
                emit(heading_line)
            continue

        if skip_until_next_item:
            continue

        # We require a current_item that's kept; otherwise pre-Items prologue/skip
        if current_item is None or current_item not in items_kept:
            continue

        # ─ Sub-heading ─
        if tok.is_subheading:
            current_subheading = tok.text.strip()
            paragraph_index_in_subsection = 0
            subheadings_kept += 1
            entry = LineageEntry(
                kind="subheading", item=current_item, sub_heading=current_subheading,
                char_start=0, char_end=0, line_start=0, line_end=0,
            )
            emit(f"### {current_subheading}\n\n", entry)
            continue

        # ─ Table ─
        if tok.kind == "table":
            tables_kept += 1
            cap = tok.table_caption.strip() or "(no caption)"
            entry = LineageEntry(
                kind="table", item=current_item, sub_heading=current_subheading,
                table_caption=tok.table_caption,
                char_start=0, char_end=0, line_start=0, line_end=0,
            )
            emit(f"**Table: {cap}**\n{tok.table_md}\n\n", entry)
            continue

        # ─ Paragraph ─
        if tok.kind == "paragraph":
            text = tok.text.strip()
            if not text:
                continue
            paragraph_index_in_subsection += 1
            entry = LineageEntry(
                kind="paragraph", item=current_item, sub_heading=current_subheading,
                paragraph_index=paragraph_index_in_subsection,
                char_start=0, char_end=0, line_start=0, line_end=0,
            )
            emit(f"{text}\n\n", entry)
            continue

    body_md = "".join(out)

    # Build stats
    stats = {
        "items_kept": sorted(items_kept_seen, key=lambda x: (int(re.match(r"\d+", x).group()), x)),
        "items_dropped": sorted(items_dropped_seen, key=lambda x: (int(re.match(r"\d+", x).group()), x)),
        "tables_kept": tables_kept,
        "subheadings_kept": subheadings_kept,
        "chars_clean": len(body_md),
    }

    # Build front-matter and prepend
    fm = _build_frontmatter(manifest, stats)
    full_md = fm + "\n" + body_md

    # Shift lineage entries by frontmatter length
    fm_offset_chars = len(fm) + 1  # +1 for the newline after fm
    fm_offset_lines = fm.count("\n") + 1
    for le in lineage:
        le.char_start += fm_offset_chars
        le.char_end += fm_offset_chars
        le.line_start += fm_offset_lines
        le.line_end += fm_offset_lines

    return full_md, lineage, stats


# ─── Top-level orchestrator ───────────────────────────────────────────────


def parse_10k(
    html_path: Path,
    manifest_entry: dict,
    cfg: Config,
    output_dir: Path,
) -> ParseResult:
    """Parse one 10-K HTML file. Writes md, meta, lineage to output_dir."""
    html_path = Path(html_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    soup = _load_and_clean(html_path, cfg)
    tokens = _walk_tokens(soup, cfg)
    _promote_heading_paragraphs(tokens)
    items = _detect_content_items(tokens)
    parts = _detect_content_parts(tokens)
    _annotate(tokens, items, parts)
    md, lineage, stats = _emit_markdown(tokens, manifest_entry, cfg)

    fy_label = f"FY{manifest_entry['fy_end'][:4]}"
    base = f"{manifest_entry['ticker']}-{fy_label}"
    md_path = output_dir / f"{base}.md"
    meta_path = output_dir / f"{base}.meta.json"
    lineage_path = output_dir / f"{base}.lineage.json"

    md_path.write_text(md, encoding="utf-8")
    meta = {
        "ticker": manifest_entry["ticker"],
        "company": manifest_entry["company"],
        "fy_end": manifest_entry["fy_end"],
        "fy_label": fy_label,
        "calendar_year": int(manifest_entry["fy_end"][:4]),
        "filed_date": manifest_entry["filed"],
        "cik": manifest_entry["cik"],
        "accession": manifest_entry["accession"],
        "edgar_url": _edgar_url(manifest_entry),
        "stats": stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lineage_path.write_text(
        json.dumps([asdict(le) for le in lineage], indent=2),
        encoding="utf-8",
    )

    stats["chars_raw_html"] = html_path.stat().st_size
    return ParseResult(
        md_path=md_path,
        meta_path=meta_path,
        lineage_path=lineage_path,
        stats=stats,
    )


# ─── CLI ──────────────────────────────────────────────────────────────────


def _load_manifest(manifest_path: Path) -> list[dict]:
    return json.loads(manifest_path.read_text())


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Parse 10-K HTML to clean Markdown.")
    ap.add_argument("--manifest", type=Path, required=True, help="Path to manifest.json")
    ap.add_argument(
        "--source-dir", type=Path, required=True,
        help="Directory containing the HTML files referenced by manifest.local_path",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data" / "cleaned",
    )
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument(
        "--filter", type=str, default=None,
        help="Substring filter on manifest local_path (e.g., 'MSFT' for MSFT only)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = _load_manifest(args.manifest)
    if args.filter:
        manifest = [m for m in manifest if args.filter in m["local_path"]]

    print(f"Parsing {len(manifest)} filings → {args.output_dir}")
    print(f"Config: {cfg}")
    for m in manifest:
        html_path = args.source_dir / m["local_path"]
        if not html_path.exists():
            print(f"  ⚠️  {m['ticker']} {m['fy_end']}: missing {html_path}")
            continue
        result = parse_10k(html_path, m, cfg, args.output_dir)
        s = result.stats
        print(
            f"  ✓ {m['ticker']:6s} {m['fy_end']}  "
            f"HTML {s['chars_raw_html']/1024/1024:5.1f}MB → "
            f"MD {s['chars_clean']/1024:6.1f}KB  "
            f"({len(s['items_kept'])} items, {s['tables_kept']} tables, "
            f"{s['subheadings_kept']} subheadings)"
        )


if __name__ == "__main__":
    main()

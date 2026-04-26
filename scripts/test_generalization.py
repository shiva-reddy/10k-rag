"""Parse 3 non-MAG7 10-Ks via src/parse_10k.py and summarize stats.

Adds a separate count of raw `<table>` elements + classification breakdown
(data / layout / skip) by re-running the classifier on the cleaned soup.

Writes data/generalization/results.json. Prints a tight summary only.
"""
from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup

from src.config_loader import load_config
from src.parse_10k import _classify_table, _load_and_clean, parse_10k

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path("~/ai-eng-datasets/non-mag7-10k").expanduser()
OUT_DIR = REPO_ROOT / "data" / "generalization"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_OUT = OUT_DIR / "cleaned"
CHUNK_OUT.mkdir(exist_ok=True)


def classify_breakdown(html_path: Path, cfg) -> dict:
    soup = _load_and_clean(html_path, cfg)
    counts = {"data": 0, "layout": 0, "skip": 0}
    for t in soup.find_all("table"):
        counts[_classify_table(t, cfg)] += 1
    counts["total"] = sum(counts.values())
    return counts


def main():
    cfg = load_config()
    manifest = json.loads((SRC_DIR / "manifest.json").read_text())

    results = []
    for m in manifest:
        html_path = SRC_DIR / m["local_path"]
        # Classification breakdown BEFORE parse (so we count all tables present).
        # parse_10k internally unwraps layout tables, so re-walk fresh soup.
        breakdown = classify_breakdown(html_path, cfg)

        result = parse_10k(html_path, m, cfg, CHUNK_OUT)
        s = result.stats
        row = {
            "ticker": m["ticker"],
            "company": m["company"],
            "fy_end": m["fy_end"],
            "html_mb": round(s["chars_raw_html"] / 1024 / 1024, 2),
            "md_kb": round(s["chars_clean"] / 1024, 1),
            "items_kept": s["items_kept"],
            "items_dropped": s["items_dropped"],
            "subheadings_kept": s["subheadings_kept"],
            "tables_kept": s["tables_kept"],
            "tables_classified": breakdown,
        }
        results.append(row)
        print(
            f"{m['ticker']:6s} {m['fy_end']}  "
            f"items={len(s['items_kept']):>2}  "
            f"subhdg={s['subheadings_kept']:>4}  "
            f"tbl_kept={s['tables_kept']:>4}  "
            f"tbl_total={breakdown['total']:>5}  "
            f"data={breakdown['data']:>4}  layout={breakdown['layout']:>4}  skip={breakdown['skip']:>4}"
        )

    out = OUT_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

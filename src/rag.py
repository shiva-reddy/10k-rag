"""rag.py — query → retrieve → expand parents → Claude → cited answer.

Pipeline stage 4 (and the system's user-facing entry point).

The retrieval flow:
  1. Embed query with the same model used for indexing.
  2. ChromaDB HNSW similarity search over CHILDREN, top-k.
  3. Dedupe matched children to unique parent_ids (preserving ranking).
  4. Look up parent texts from parents.jsonl (loaded once into memory).
  5. Build a prompt with system + parent excerpts + user query.
  6. Call Claude with temperature=0, get a cited answer.

This is the literal "one hop child→parent" step — the hierarchical
chunking moment. Retrieval ranks children; the answer reads parents.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config_loader import Config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHROMA_DIR = REPO_ROOT / "data" / "chroma"
DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "chunks"


# ─── Result dataclasses ───────────────────────────────────────────────────


@dataclass
class RetrievedChild:
    chunk_id: str
    parent_id: str
    text: str
    similarity: float
    metadata: dict


@dataclass
class ParentContext:
    """A parent that one or more retrieved children point to."""
    parent_id: str
    text: str
    metadata: dict
    matched_children: list[RetrievedChild] = field(default_factory=list)
    best_similarity: float = 0.0


@dataclass
class RAGAnswer:
    query: str
    answer: str
    parents: list[ParentContext]   # in the order they were sent to the LLM
    children: list[RetrievedChild] # raw retrieval hits, ranked
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ─── The system (loads heavy artifacts once) ──────────────────────────────


class RAGSystem:
    """Encapsulates the retrieve+generate pipeline. Construct once per process."""

    def __init__(
        self,
        cfg: Config,
        chroma_dir: Path = DEFAULT_CHROMA_DIR,
        chunks_dir: Path = DEFAULT_CHUNKS_DIR,
        llm_backend: str = "auto",
    ):
        """
        llm_backend:
          "auto"        — anthropic SDK if ANTHROPIC_API_KEY set, else claude CLI
          "anthropic"   — Anthropic SDK (requires ANTHROPIC_API_KEY)
          "claude_cli"  — shell out to `claude -p`
          "none"        — retrieval only, no generation
        """
        self.cfg = cfg

        import chromadb
        from sentence_transformers import SentenceTransformer

        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self._client.get_collection(cfg.vector_db.collection_name)

        self._embedder = SentenceTransformer(cfg.embedding.model)

        # Load parents into memory — small (~50 MB worst case)
        self.parents: dict[str, dict] = {}
        parents_path = chunks_dir / "parents.jsonl"
        with open(parents_path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    self.parents[p["parent_id"]] = p

        self.llm_backend = self._resolve_backend(llm_backend)
        self._anthropic = None
        if self.llm_backend == "anthropic":
            import anthropic
            self._anthropic = anthropic.Anthropic()

    @staticmethod
    def _resolve_backend(backend: str) -> str:
        if backend == "auto":
            if os.environ.get("ANTHROPIC_API_KEY"):
                try:
                    import anthropic  # noqa
                    return "anthropic"
                except ImportError:
                    pass
            import shutil
            if shutil.which("claude"):
                return "claude_cli"
            return "none"
        if backend == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY not set")
        if backend == "claude_cli":
            import shutil
            if not shutil.which("claude"):
                raise RuntimeError("`claude` CLI not found in PATH")
        return backend

    # ─── Retrieval ───────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list[float]:
        vec = self._embedder.encode(
            [query],
            normalize_embeddings=self.cfg.embedding.normalize_embeddings,
            show_progress_bar=False,
        )
        return vec[0].tolist()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        where: dict | None = None,
    ) -> list[RetrievedChild]:
        """Return ranked child matches. `where` is a ChromaDB metadata filter
        (e.g., {"ticker": "MSFT"} or {"$and": [...]} ).
        """
        k = top_k or self.cfg.retrieval.top_k_children
        qv = self._embed_query(query)
        kwargs: dict[str, Any] = dict(query_embeddings=[qv], n_results=k)
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)

        children: list[RetrievedChild] = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for cid, doc, md, dist in zip(ids, docs, metas, dists):
            similarity = max(0.0, 1.0 - dist)  # cosine distance → similarity
            children.append(RetrievedChild(
                chunk_id=cid,
                parent_id=md.get("parent_id", ""),
                text=doc,
                similarity=similarity,
                metadata=dict(md),
            ))
        return children

    def expand_parents(
        self,
        children: list[RetrievedChild],
        max_unique: int | None = None,
    ) -> list[ParentContext]:
        """Dedupe matched children to unique parents, preserving rank order.
        Each parent gets the children that pointed to it + best similarity."""
        max_unique = max_unique or self.cfg.retrieval.max_unique_parents

        seen: dict[str, ParentContext] = {}
        order: list[str] = []
        for c in children:
            pid = c.parent_id
            if not pid or pid not in self.parents:
                continue
            if pid not in seen:
                p = self.parents[pid]
                seen[pid] = ParentContext(
                    parent_id=pid,
                    text=p["text"],
                    metadata=p["metadata"],
                    matched_children=[c],
                    best_similarity=c.similarity,
                )
                order.append(pid)
            else:
                pc = seen[pid]
                pc.matched_children.append(c)
                pc.best_similarity = max(pc.best_similarity, c.similarity)
        ordered = [seen[pid] for pid in order[:max_unique]]
        return ordered

    # ─── Prompt construction ─────────────────────────────────────────────

    def _format_citation(self, md: dict) -> str:
        """Inline citation per config.generation.citation_style."""
        ticker = md.get("ticker", "?")
        fy = md.get("fy_label", "?")
        item = md.get("item", "?")
        sub = md.get("sub_heading") or ""
        style = self.cfg.generation.citation_style
        if style == "inline_full":
            company = md.get("company", ticker)
            base = f"{company} {fy} 10-K, {item}"
            return f"({base})" if not sub else f"({base}, \"{sub}\")"
        if style == "footnote_numbered":
            return ""  # numbering done elsewhere if used
        # default inline_compact
        if sub:
            return f"({ticker} {fy} {item}, \"{sub}\")"
        return f"({ticker} {fy} {item})"

    def build_prompt(self, query: str, parents: list[ParentContext]) -> tuple[str, str]:
        """Return (system_prompt, user_prompt) for the LLM."""
        system = (
            "You are a financial analyst answering questions about Microsoft (MSFT), "
            "Apple (AAPL), Alphabet (GOOGL), Amazon (AMZN), Meta (META), Nvidia (NVDA), "
            "and Tesla (TSLA) based ONLY on excerpts from their SEC 10-K filings (FY2021 "
            "through FY2025) provided to you in the user message.\n\n"
            "Rules:\n"
            "1. Use ONLY the provided excerpts. Do NOT draw on prior knowledge of these "
            "   companies, their products, or events not described in the excerpts.\n"
            "2. If the excerpts do not contain enough information to answer, say so "
            "   plainly: \"The provided 10-K excerpts do not address this.\"\n"
            "3. Cite every factual claim inline using the company-fiscal-year-Item form "
            "   shown next to each excerpt below, e.g., (MSFT FY2024 Item 1A).\n"
            "4. When comparing companies or years, quote or paraphrase from each "
            "   relevant excerpt and cite each one separately.\n"
            "5. Be concise. Match length to the question; don't pad."
        )

        ctx_lines: list[str] = []
        for i, p in enumerate(parents, 1):
            cite = self._format_citation(p.metadata)
            heading = f"--- SOURCE {i}: {cite} ---"
            ctx_lines.append(heading)
            ctx_lines.append(p.text.strip())
            ctx_lines.append("")
        context_block = "\n".join(ctx_lines).strip()

        user = (
            f"Below are excerpts from MAG7 10-K filings.\n\n"
            f"{context_block}\n\n"
            f"--- QUESTION ---\n{query.strip()}\n\n"
            f"Answer the question using ONLY these excerpts. Cite inline."
        )
        return system, user

    # ─── Generation ──────────────────────────────────────────────────────

    def generate(self, system: str, user: str) -> tuple[str, int, int]:
        if self.llm_backend == "anthropic":
            return self._generate_anthropic(system, user)
        if self.llm_backend == "claude_cli":
            return self._generate_claude_cli(system, user)
        raise RuntimeError(f"No LLM backend available (llm_backend={self.llm_backend!r})")

    def _generate_anthropic(self, system: str, user: str) -> tuple[str, int, int]:
        msg = self._anthropic.messages.create(
            model=self.cfg.generation.llm_model,
            max_tokens=self.cfg.generation.llm_max_tokens,
            temperature=self.cfg.generation.llm_temperature,
            system=[
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            messages=[{"role": "user", "content": user}],
        )
        text_parts = [b.text for b in msg.content if b.type == "text"]
        return "\n".join(text_parts), msg.usage.input_tokens, msg.usage.output_tokens

    def _generate_claude_cli(self, system: str, user: str) -> tuple[str, int, int]:
        """Shell out to `claude -p` (Claude Code CLI). System and user are
        concatenated; token counts are not available from the CLI."""
        import subprocess
        prompt = f"{system}\n\n{user}"
        proc = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p failed (exit {proc.returncode}): {proc.stderr.strip()}"
            )
        return proc.stdout.strip(), 0, 0

    # ─── End-to-end ──────────────────────────────────────────────────────

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        max_unique: int | None = None,
        where: dict | None = None,
    ) -> RAGAnswer:
        children = self.retrieve(query, top_k=top_k, where=where)
        parents = self.expand_parents(children, max_unique=max_unique)
        if not parents:
            return RAGAnswer(
                query=query,
                answer="(no relevant excerpts retrieved)",
                parents=[], children=children,
            )
        system, user = self.build_prompt(query, parents)
        answer, in_toks, out_toks = self.generate(system, user)
        return RAGAnswer(
            query=query, answer=answer,
            parents=parents, children=children,
            prompt_tokens=in_toks, completion_tokens=out_toks,
        )


# ─── CLI ──────────────────────────────────────────────────────────────────


def _print_answer(r: RAGAnswer, *, verbose: bool = False) -> None:
    print()
    print("=" * 78)
    print(f"Q: {r.query}")
    print("=" * 78)
    print()
    print(r.answer)
    print()
    print("─" * 78)
    print(f"Sources used (parents):")
    for i, p in enumerate(r.parents, 1):
        md = p.metadata
        ticker = md.get("ticker", "?")
        fy = md.get("fy_label", "?")
        item = md.get("item", "?")
        sub = md.get("sub_heading") or "(item-level)"
        url = md.get("edgar_url", "")
        print(f"  [{i}] {ticker} {fy} {item} → {sub}")
        if url:
            print(f"      {url}")
        if verbose:
            print(f"      best_similarity={p.best_similarity:.3f}, "
                  f"matched {len(p.matched_children)} child(ren), "
                  f"parent_text {len(p.text):,} chars")
    if verbose and r.prompt_tokens:
        print(f"\nTokens: {r.prompt_tokens} in / {r.completion_tokens} out")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Ask the 10k-rag system a question.")
    ap.add_argument("query", nargs="*", help="The question. If empty, reads stdin.")
    ap.add_argument("--ticker", help="Filter to one ticker (e.g., MSFT)")
    ap.add_argument("--fy", help="Filter to one fiscal year label (e.g., FY2024)")
    ap.add_argument("--item", help="Filter to one item (e.g., 'Item 1A')")
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--max-parents", type=int)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--retrieve-only", action="store_true",
        help="Skip LLM generation; just print retrieval results.",
    )
    ap.add_argument(
        "--backend", default="auto",
        choices=["auto", "anthropic", "claude_cli", "none"],
        help="LLM backend: auto picks anthropic SDK (if API key set) else claude CLI.",
    )
    ap.add_argument(
        "--print-prompt", action="store_true",
        help="Print the assembled prompt instead of calling the LLM.",
    )
    args = ap.parse_args()

    query = " ".join(args.query) if args.query else sys.stdin.read().strip()
    if not query:
        ap.error("No query provided")

    cfg = load_config(args.config)
    backend = "none" if args.retrieve_only else args.backend
    system = RAGSystem(cfg=cfg, llm_backend=backend)

    where: dict[str, Any] = {}
    filters = []
    if args.ticker:
        filters.append({"ticker": args.ticker})
    if args.fy:
        filters.append({"fy_label": args.fy})
    if args.item:
        filters.append({"item": args.item})
    if len(filters) == 1:
        where = filters[0]
    elif filters:
        where = {"$and": filters}

    if args.retrieve_only or args.print_prompt:
        children = system.retrieve(query, top_k=args.top_k, where=where or None)
        parents = system.expand_parents(children, max_unique=args.max_parents)
        if args.print_prompt:
            sys_prompt, user_prompt = system.build_prompt(query, parents)
            print("=" * 78)
            print("SYSTEM PROMPT")
            print("=" * 78)
            print(sys_prompt)
            print()
            print("=" * 78)
            print("USER PROMPT")
            print("=" * 78)
            print(user_prompt)
            return
        print(f"Q: {query}\n")
        for i, p in enumerate(parents, 1):
            md = p.metadata
            print(f"[{i}] sim={p.best_similarity:.3f}  "
                  f"{md.get('ticker')} {md.get('fy_label')} {md.get('item')} → "
                  f"{md.get('sub_heading') or '(item-level)'}  "
                  f"({len(p.text):,} chars)")
        return

    r = system.ask(query, top_k=args.top_k, max_unique=args.max_parents,
                   where=where or None)
    _print_answer(r, verbose=args.verbose)


if __name__ == "__main__":
    main()

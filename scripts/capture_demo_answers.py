"""Run all 5 eval queries plus the out-of-scope query through the full pipeline,
capture top-3 retrievals + LLM answer, write to data/demo_answers.json."""
import json
from pathlib import Path

from src.config_loader import load_config
from src.rag import RAGSystem

REPO_ROOT = Path(__file__).resolve().parent.parent

QUERIES = [
    ("Q1", "How does Microsoft frame AI risk in its risk factors?"),
    ("Q2", "Apple's iPhone revenue in fiscal 2024"),
    ("Q3", "Tesla's key-person succession risk"),
    ("Q4", "Google Cloud segment performance"),
    ("Q5", "Cybersecurity governance disclosures across MAG7"),
    ("OOS", "What is Berkshire Hathaway's combined ratio?"),
]


def main():
    cfg = load_config()
    rag = RAGSystem(cfg=cfg, llm_backend="claude_cli")

    out = []
    for qid, q in QUERIES:
        print(f"\n[{qid}] {q}")
        children = rag.retrieve(q, top_k=cfg.retrieval.top_k_children)
        parents = rag.expand_parents(children, max_unique=cfg.retrieval.max_unique_parents)
        system_prompt, user_prompt = rag.build_prompt(q, parents)
        answer, prompt_toks, completion_toks = rag.generate(system_prompt, user_prompt)

        row = {
            "qid": qid,
            "query": q,
            "top3": [
                {
                    "ticker": c.metadata.get("ticker"),
                    "fy": c.metadata.get("fy_label"),
                    "item": c.metadata.get("item"),
                    "sub_heading": c.metadata.get("sub_heading") or "",
                    "similarity": round(c.similarity, 4),
                } for c in children[:3]
            ],
            "n_parents": len(parents),
            "answer": answer,
        }
        out.append(row)
        print(f"  top1: {row['top3'][0]['ticker']} {row['top3'][0]['fy']} {row['top3'][0]['item']} (sim={row['top3'][0]['similarity']})")
        print(f"  answer: {answer[:200]}...")

    (REPO_ROOT / "data" / "demo_answers.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote data/demo_answers.json ({len(out)} queries)")


if __name__ == "__main__":
    main()

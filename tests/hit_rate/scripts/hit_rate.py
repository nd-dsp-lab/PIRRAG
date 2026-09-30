#!/usr/bin/env python3
"""
Hit-rate analysis for Tiptoe logs, using the same matching rules as the
repo's post-PIR evaluators (tests/*/evaluate_post_pir_retrieval.py):

  FEVER   : hit if any evidence entity (page title, '_' -> ' ') appears in the doc text
  NQ      : hit if any sentence of the long `answer` passage appears verbatim in the doc text
  HotpotQA: hit if the `answer` string appears in the doc text (pir_input has no supporting_facts)

hit@K = fraction of queries with >=1 hit among the first K retrieved docs; MRR uses the first hit.
"""
import json, re, statistics as st, sys
from pathlib import Path

K_VALUES = [1, 3, 5, 10]

def norm(s): return s.lower().strip()

def fever_targets(q):
    ents = {e[2] for g in q["evidence"] for e in g if len(e) >= 3 and e[2]}
    return [norm(e.replace("_", " ")) for e in ents]

def nq_targets(q):
    t = [norm(a) for a in q.get("short_answer_strings", []) if a and a.strip()]
    ans = q.get("answer", "") or ""
    t += [norm(s) for s in re.split(r"[.!?]\s+", ans) if len(s.strip()) > 3]
    return t

def hotpot_targets(q):
    a = (q.get("answer") or "").strip()
    return [norm(a)] if a else []

def nq_title_targets(q):
    return [norm(q.get("answer", "") or "")]   # the long answer starts with its page title

TARGETS = {"fever": fever_targets, "nq": nq_targets, "nq_title": nq_title_targets, "hotpot": hotpot_targets}
LABEL = {"fever": "evidence entity in doc", "nq": "long-answer sentence in doc",
         "nq_title": "answer-page title among retrieved titles", "hotpot": "answer string in doc"}
TITLE_RULES = {"nq_title"}

def evaluate(tag, parsed, docs):
    queries = {q["query_id"]: q for q in parsed["queries"]}
    rows = []
    for r in parsed["results"]:
        q = queries[r["query_id"]]
        targets = TARGETS[tag](q)
        texts = [norm(docs[str(i)]["text"]) for i in r["retrieved_rows"] if str(i) in docs]
        if tag in TITLE_RULES:
            titles = [norm(str(docs[str(i)]["metadata"].get("title") or "")) for i in r["retrieved_rows"] if str(i) in docs]
            first_hit = next((k + 1 for k, t in enumerate(titles) if len(t) >= 4 and any(x.startswith(t) for x in targets)), None)
        else:
            first_hit = next((k + 1 for k, t in enumerate(texts) if any(x in t for x in targets)), None)
        rows.append({"query_id": r["query_id"], "n_retrieved": len(texts), "first_hit_rank": first_hit,
                     "has_targets": bool(targets), "selected_cluster": r.get("selected_cluster"),
                     "query_wall_s": r.get("query_wall_s")})
    scored = [x for x in rows if x["has_targets"]]
    n = len(scored)
    out = {"dataset": tag, "rule": LABEL[tag], "n_queries": len(rows), "n_scored": n,
           "retrieved_per_query_mean": st.mean(x["n_retrieved"] for x in rows),
           "queries_with_10_docs": sum(x["n_retrieved"] >= 10 for x in rows),
           "hit_at": {k: sum(1 for x in scored if x["first_hit_rank"] and x["first_hit_rank"] <= k) / n for k in K_VALUES},
           "hit_any": sum(1 for x in scored if x["first_hit_rank"]) / n,
           "mrr": sum(1 / x["first_hit_rank"] for x in scored if x["first_hit_rank"]) / n,
           "query_wall_s_mean": st.mean([x["query_wall_s"] for x in rows if x.get("query_wall_s")] or [float("nan")]),
           "per_query": rows}
    return out

def main():
    work = Path(sys.argv[1]); docs = json.load(open(work / "docs.json"))
    results = {tag: evaluate(tag, json.load(open(work / f"{tag.replace('_title', '')}_parsed.json")), docs) for tag in ["fever", "hotpot", "nq", "nq_title"]}
    json.dump(results, open(work / "hit_rate_results.json", "w"), indent=1)
    lines = ["| dataset | rule | queries | docs/query | hit@1 | hit@3 | hit@5 | hit@10 | hit@any | MRR | s/query |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, r in results.items():
        h = r["hit_at"]
        lines.append(f"| {tag} | {r['rule']} | {r['n_scored']} | {r['retrieved_per_query_mean']:.1f} | "
                     f"{h[1]:.2f} | {h[3]:.2f} | {h[5]:.2f} | {h[10]:.2f} | {r['hit_any']:.2f} | {r['mrr']:.3f} | {r['query_wall_s_mean']:.1f} |")
    md = "\n".join(lines); (work / "hit_rate_results.md").write_text(md + "\n"); print(md)

if __name__ == "__main__":
    main()

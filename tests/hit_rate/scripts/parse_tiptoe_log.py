#!/usr/bin/env python3
"""Parse a Tiptoe experiment log into per-query retrieved document rows and timings."""
import ast, json, re, sys
from pathlib import Path

def parse(path):
    lines = Path(path).read_text(errors="replace").read().splitlines() if False else Path(path).read_text(errors="replace").splitlines()
    queries = ast.literal_eval(lines[5])            # the pir_input list echoed at the top
    n_docs = int(re.search(r"with (\d+) documents", "\n".join(lines[:5])).group(1))
    results, cur = [], None
    for i, l in enumerate(lines):
        m = re.match(r"\s*Query (\d+)/(\d+)", l)
        if m:
            cur = {"query_id": int(m.group(1)) - 1}
            results.append(cur); continue
        if cur is None: continue
        m = re.search(r"Computing dot products for (\d+) documents", l)
        if m: cur["cluster_docs_ranked"] = int(m.group(1))
        m = re.search(r"CORRECTED query completed in ([0-9.]+)s", l)
        if m: cur["query_wall_s"] = float(m.group(1))
        if l.startswith("{'total_query_time'"):
            try:
                d = ast.literal_eval(l)
                cur.update({k: d[k] for k in d if k in ("total_query_time","phase1_time","phase2_round1_time","phase2_round2_time","selected_cluster")})
                for k in ("documents_retrieved","num_documents","documents"):
                    if k in d: cur["documents_field"] = d[k] if not isinstance(d[k], list) else len(d[k])
            except Exception:
                cur["selected_cluster"] = int(re.search(r"'selected_cluster': (\d+)", l).group(1))
        if "example.com/doc_" in l and l.lstrip().startswith("["):
            cur["retrieved_rows"] = [int(x) for x in re.findall(r"doc_(\d+)", l)]
    for r in results:
        r.setdefault("retrieved_rows", [])
        q = queries[r["query_id"]]
        r["query_text"] = q.get("claim") or q.get("question") or q.get("question_text")
    return {"n_documents": n_docs, "n_queries": len(results), "queries": queries, "results": results}

if __name__ == "__main__":
    out = parse(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps(out))
    rs = out["results"]; nret = [len(r["retrieved_rows"]) for r in rs]
    import statistics as st
    print(f"{Path(sys.argv[1]).name}: {out['n_queries']} queries over {out['n_documents']:,} docs; "
          f"retrieved/query mean {st.mean(nret):.1f} (min {min(nret)}, max {max(nret)}); "
          f"query wall mean {st.mean(r['query_wall_s'] for r in rs if 'query_wall_s' in r):.1f}s; "
          f"cluster docs ranked mean {st.mean(r['cluster_docs_ranked'] for r in rs if 'cluster_docs_ranked' in r):.0f}; "
          f"sample: q0 rows {rs[0]['retrieved_rows'][:5]} cluster {rs[0].get('selected_cluster')}")

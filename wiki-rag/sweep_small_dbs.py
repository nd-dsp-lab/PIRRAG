#!/usr/bin/env python3
"""
Equal-size clustering for small prefix-truncated databases (1k, 5k, ...).

Database = the first N rows of the index (what prototype/truncate_databases.py
produces); the 1,000 held-out queries are drawn from rows beyond N, so the
database keeps its exact size and every clustering written here is shippable
(id_space = database rows 0..N-1).

Baseline per size: K = round(N/16), p = max(2, round(K/41)) -- the 65k rule scaled.
Equal-size: for each n, the smallest p reaching the baseline's recall.

    python sweep_small_dbs.py --faiss-index-file INDEX --sizes 1000 5000 --work-dir cluster_size_sweep_small
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import ivf_eval, ivf_io, train_ivf
import sweep_cluster_size as sweep


def prefix_query_set(vectors, n_db, n_test, k, seed):
    rng = np.random.default_rng(seed)
    db = np.ascontiguousarray(vectors[:n_db])
    pool = np.arange(n_db, len(vectors))
    qpos = rng.choice(pool, size=min(n_test, len(pool)), replace=False)
    queries = np.ascontiguousarray(vectors[qpos])
    gt = ivf_eval.brute_force_ground_truth(db, queries, k, verbose=False)
    return {"db_vectors": db, "db_ids": np.arange(n_db), "queries": queries, "ground_truth": gt,
            "mode": "prefix-heldout", "metric_name": f"heldout_recall@{k}", "n_test": len(qpos),
            "k": k, "seed": seed, "query_noise": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss-index-file", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000])
    ap.add_argument("--cluster-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--work-dir", default="cluster_size_sweep_small")
    ap.add_argument("--n-test-queries", type=int, default=1000)
    ap.add_argument("--recall-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=train_ivf.DEFAULT_KMEANS_SEED)
    ap.add_argument("--niter", type=int, default=20)
    args = ap.parse_args()

    vectors = sweep.load_vectors(faiss_index_file=args.faiss_index_file)
    base_ns = sweep._base_namespace(argparse.Namespace(
        index_type="ivf-flat", niter=args.niter, seed=args.seed, recall_k=args.recall_k,
        n_test_queries=args.n_test_queries, recall_mode="holdout", query_noise=0.05,
        assignment="balanced", assign_topm=sweep.balanced_ivf.DEFAULT_TOP_M,
        price_iters=sweep.balanced_ivf.DEFAULT_PRICE_ITERS, dual_bound=True,
        lists_json="padded", use_gpu=False))
    work = Path(args.work_dir); work.mkdir(parents=True, exist_ok=True)
    all_records = {}

    for N in args.sizes:
        t0 = time.time()
        qs = prefix_query_set(vectors, N, args.n_test_queries, args.recall_k, args.seed)
        K = max(1, round(N / 16)); p_base = max(2, round(K / 41))
        wd = work / f"N{N}"; wd.mkdir(exist_ok=True)
        print(f"\n{'#'*70}\n# N={N:,}  baseline K={K} p={p_base}  ({qs['n_test']} held-out queries from rows >= {N})\n{'#'*70}")

        train_ivf.run_pipeline(sweep._config(base_ns, k=K, cluster_size=None, output_dir=str(wd / "baseline")),
                               vectors=qs["db_vectors"], dim=int(vectors.shape[1]), query_set=qs)
        base = ivf_io.load_clustering(wd / "baseline", load_centroids=True)
        sizes = np.array([len(base.members(c)) for c in range(base.nlist)])
        base_eval = ivf_eval.evaluate_clustering(qs, base.centroids, base.lists, [p_base])[p_base]
        cands = ivf_eval.candidates_per_query(qs["queries"], base.centroids, sizes, p_base)
        target = base_eval["recall"]; base_total = K + float(cands.mean())
        print(f"  BASELINE recall={target:.4f} candidates mean {cands.mean():,.0f} "
              f"(min {cands.min():,} max {cands.max():,}), sizes {sizes.min()}..{sizes.max()}")

        rows = []
        for n in args.cluster_sizes:
            if n > N // 2:
                print(f"  skip n={n} (fewer than 2 clusters)"); continue
            out = wd / f"n{n}"
            meta = train_ivf.run_pipeline(sweep._config(base_ns, k=None, cluster_size=n, output_dir=str(out)),
                                          vectors=qs["db_vectors"], dim=int(vectors.shape[1]), query_set=qs)
            c = ivf_io.load_clustering(out, load_centroids=True)
            found = sweep.find_min_nprobe(qs, c.centroids, c.slots, target_recall=target,
                                          max_nprobe=c.nlist, verbose=False)
            p = found["nprobe"]; diag = meta["diagnostics"] or {}
            row = {"cluster_size": n, "nlist": c.nlist, "n_padded_slots": meta["n_padded_slots"],
                   "min_nprobe": p, "recall": found["recall"], "candidates": p * n if p else None,
                   "total": c.nlist + p * n if p else None,
                   "vs_baseline": (c.nlist + p * n) / base_total if p else None,
                   "balance_penalty_ratio": diag.get("balance_penalty_ratio"),
                   "duality_gap_pct": diag.get("duality_gap_pct"), "curve": found["curve"], "dir": str(out)}
            rows.append(row)
            vs = "n/a" if p is None else f"{row['vs_baseline']:.2f}x"
            print(f"  n={n:<4d} K={c.nlist:<5d} p={p!s:<5} cands={row['candidates']!s:<7} total={row['total']!s:<7} "
                  f"vs base {vs}  recall={found['recall']:.4f} penalty={row['balance_penalty_ratio']:.4f}")

        feasible = [r for r in rows if r["total"] is not None]
        best = min(feasible, key=lambda r: r["total"]) if feasible else None
        rec = {"N": N, "n_test": qs["n_test"], "metric": qs["metric_name"], "seed": args.seed,
               "baseline": {"K": K, "p": p_base, "recall": target, "candidates_mean": float(cands.mean()),
                            "candidates_min": int(cands.min()), "candidates_max": int(cands.max()),
                            "total": base_total, "size_min": int(sizes.min()), "size_max": int(sizes.max())},
               "rows": rows, "best": best, "seconds": round(time.time() - t0, 1)}
        all_records[str(N)] = rec
        (wd / "sweep.json").write_text(json.dumps(rec, indent=2))
        if best:
            print(f"  BEST for N={N:,}: n={best['cluster_size']} (K={best['nlist']}, p={best['min_nprobe']}) "
                  f"total {best['total']:,} vs baseline {base_total:,.0f} ({best['vs_baseline']:.2f}x)")

    (work / "sweep_small.json").write_text(json.dumps(all_records, indent=2))
    lines = ["# Equal-size clustering on small prefix databases", "",
             "| N | config | K | p | recall@10 | centroid | candidates | total | vs base | penalty |", "|---|---|---|---|---|---|---|---|---|---|"]
    for N, rec in all_records.items():
        b = rec["baseline"]
        lines.append(f"| {int(N):,} | k-means K=N/16 | {b['K']} | {b['p']} | {b['recall']:.4f} | {b['K']} | "
                     f"{b['candidates_mean']:,.0f} ({b['candidates_min']:,}–{b['candidates_max']:,}) | {b['total']:,.0f} | 1.00× | — |")
        for r in rec["rows"]:
            mark = "**" if rec["best"] and r["cluster_size"] == rec["best"]["cluster_size"] else ""
            if r["min_nprobe"] is None:
                lines.append(f"| {int(N):,} | equal-size n={r['cluster_size']} | {r['nlist']} | — | unreachable | | | | | |"); continue
            lines.append(f"| {int(N):,} | {mark}equal-size n={r['cluster_size']}{mark} | {r['nlist']} | {mark}{r['min_nprobe']}{mark} | {r['recall']:.4f} | "
                         f"{r['nlist']} | {r['candidates']:,} | {r['total']:,} | {r['vs_baseline']:.2f}× | {r['balance_penalty_ratio']:.3f} |")
    (work / "sweep_small.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()

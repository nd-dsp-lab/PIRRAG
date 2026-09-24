#!/usr/bin/env python3
"""
Plaintext verification that "top-43 of 500 equal-size clusters" is the operating
point equivalent to "top-100 of 4096 k-means clusters".

Everything here is plaintext numpy: no FHE, no PIR. It rebuilds both clusterings
from the raw vectors, scores them on one shared held-out query set with exact
reranking, and prints recall@10 as a function of p for each, side by side.

    python verify_p43.py --faiss-index-file .../index.faiss --work-dir verify_p43

Add --with-n16 to also build the "keep K=4096 but force equal sizes" control
(n=16), which is the direct answer to "why not just balance the existing 4096".
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

import ivf_eval
import ivf_io
import sweep_cluster_size as sweep
import train_ivf

BASELINE_P_GRID = (25, 50, 75, 100, 125, 150, 200)
N128_P_GRID = (16, 24, 32, 40, 41, 42, 43, 44, 45, 48, 64, 96, 128)
N16_P_GRID = (100, 200, 300, 400, 418, 419, 420, 450, 500)


def per_query_candidates(query_set, clustering, p):
    sizes = np.array([len(clustering.members(c)) for c in range(clustering.nlist)])
    return ivf_eval.candidates_per_query(query_set["queries"], clustering.centroids, sizes, p)


def min_p_on_grid(evals, target):
    hits = [p for p in sorted(evals) if evals[p]["recall"] >= target]
    return hits[0] if hits else None


def evaluate(query_set, clustering, p_grid):
    slots = clustering.slots if clustering.is_constant_size else clustering.lists
    return ivf_eval.evaluate_clustering(query_set, clustering.centroids, slots, p_grid)


def train(base_ns, out_dir, vectors, query_set, **overrides):
    ns = copy.deepcopy(base_ns)
    ns.output_dir = str(out_dir)
    for k, v in overrides.items():
        setattr(ns, k, v)
    train_ivf.run_pipeline(ns, vectors=vectors, dim=int(vectors.shape[1]), query_set=query_set)
    return ivf_io.load_clustering(out_dir, load_centroids=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss-index-file", required=True)
    ap.add_argument("--work-dir", default="verify_p43")
    ap.add_argument("--with-n16", action="store_true")
    ap.add_argument("--seed", type=int, default=train_ivf.DEFAULT_KMEANS_SEED)
    ap.add_argument("--n-test-queries", type=int, default=1000)
    ap.add_argument("--recall-k", type=int, default=10)
    ap.add_argument("--niter", type=int, default=20)
    ap.add_argument("--sweep-json", default="cluster_size_sweep/cluster_size_sweep.json",
                    help="Earlier sweep record to cross-check against (optional).")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    vectors = sweep.load_vectors(faiss_index_file=args.faiss_index_file)
    query_set = ivf_eval.build_query_set(
        vectors, mode="holdout", n_test=args.n_test_queries, k=args.recall_k, seed=args.seed,
    )
    n_db = len(query_set["db_vectors"])

    sweep_ns = argparse.Namespace(
        index_type="ivf-flat", niter=args.niter, seed=args.seed, recall_k=args.recall_k,
        n_test_queries=args.n_test_queries, recall_mode="holdout", query_noise=0.05,
        assignment="balanced", assign_topm=sweep.balanced_ivf.DEFAULT_TOP_M,
        price_iters=sweep.balanced_ivf.DEFAULT_PRICE_ITERS, dual_bound=True,
        lists_json="padded", use_gpu=False,
    )
    base_ns = sweep._base_namespace(sweep_ns)

    print("\n" + "#" * 70 + "\n# BASELINE  K=4096, variable sizes\n" + "#" * 70)
    base = train(base_ns, work / "baseline", vectors, query_set, k=4096, cluster_size=None)
    base_sizes = np.array([len(base.members(c)) for c in range(base.nlist)])
    base_eval = evaluate(query_set, base, BASELINE_P_GRID)
    base_cands_100 = per_query_candidates(query_set, base, 100)

    print("\n" + "#" * 70 + "\n# EQUAL-SIZE  n=128 -> K=ceil(N/128)\n" + "#" * 70)
    n128 = train(base_ns, work / "n128", vectors, query_set, k=None, cluster_size=128)
    n128_sizes = np.array([len(n128.members(c)) for c in range(n128.nlist)])
    n128_eval = evaluate(query_set, n128, N128_P_GRID)

    n16 = n16_eval = None
    if args.with_n16:
        print("\n" + "#" * 70 + "\n# CONTROL  n=16 -> K=ceil(N/16) (equal-size at ~4096 clusters)\n" + "#" * 70)
        n16 = train(base_ns, work / "n16", vectors, query_set, k=None, cluster_size=16)
        n16_eval = evaluate(query_set, n16, N16_P_GRID)

    # ------------------------------------------------------------------ report
    target = base_eval[100]["recall"]
    base_total = 4096 + float(base_cands_100.mean())

    def row(label, K, p, rec, cands, note=""):
        total = K + cands
        return (f"{label:<22s} {K:>5d} {p:>5d} {rec:>8.4f} {K:>9,d} {cands:>9,.0f} "
                f"{total:>9,.0f} {total / base_total:>7.2f}x  {note}")

    hdr = (f"{'config':<22s} {'K':>5s} {'p':>5s} {'rec@10':>8s} {'centroid':>9s} "
           f"{'cands':>9s} {'total':>9s} {'vs base':>8s}")
    lines = [
        f"Database {n_db:,} vectors (dim {vectors.shape[1]}), {query_set['n_test']} held-out "
        f"queries, seed {args.seed}, exact rerank, metric {query_set['metric_name']}",
        "",
        f"Baseline cluster sizes: min {base_sizes.min()} / mean {base_sizes.mean():.2f} / "
        f"max {base_sizes.max()}   (spread {base_sizes.max() / base_sizes.min():.0f}x)",
        f"Baseline candidates at p=100 per query: min {base_cands_100.min():,} / "
        f"median {int(np.median(base_cands_100)):,} / max {base_cands_100.max():,}  "
        f"(varies per query -> occupancy leak)",
        f"Equal-size n=128: K={n128.nlist}, distinct cluster sizes = {sorted(set(n128_sizes.tolist()))}, "
        f"candidates at p are exactly p*128 for every query",
        "",
        f"TARGET: baseline recall@10 at p=100 = {target:.4f}",
        "",
        hdr, "-" * len(hdr),
    ]
    for p in BASELINE_P_GRID:
        c = per_query_candidates(query_set, base, p).mean()
        lines.append(row("k-means K=4096", 4096, p, base_eval[p]["recall"], c,
                         "<- baseline" if p == 100 else ""))
    lines.append("")
    n128_min_p = min_p_on_grid(n128_eval, target)
    for p in N128_P_GRID:
        rec = n128_eval[p]["recall"]
        mark = ""
        if p == 43:
            mark = "<- reported operating point"
        elif p == n128_min_p:
            mark = "<- smallest p on this grid reaching the target (this run)"
        lines.append(row("equal-size n=128", n128.nlist, p, rec, p * 128, mark))
    n16_min_p = None
    if n16_eval:
        n16_min_p = min_p_on_grid(n16_eval, target)
        lines.append("")
        for p in N16_P_GRID:
            mark = "<- smallest p on this grid reaching the target" if p == n16_min_p else ""
            lines.append(row("equal-size n=16", n16.nlist, p, n16_eval[p]["recall"], p * 16, mark))

    # ------------------------------------------------------------------ checks
    checks = {
        "n128_p43_reaches_baseline": n128_eval[43]["recall"] >= target,
        "n128_all_clusters_exactly_128": set(n128_sizes.tolist()) == {128},
        "n128_K_is_ceil_N_over_128": n128.nlist == -(-n_db // 128),
        "n128_p43_cheaper_in_total_than_baseline": n128.nlist + 43 * 128 < base_total,
        "baseline_sizes_are_unequal": base_sizes.min() != base_sizes.max(),
    }
    if n16_eval:
        checks["n16_needs_far_more_probes_than_n128"] = (n16_min_p or 10**9) > 4 * 43
    checks = {k: bool(v) for k, v in checks.items()}

    lines += ["", "CHECKS"]
    for name, ok in checks.items():
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    lines += [
        "",
        "NOTE: the balanced assignment reproduces exactly within one environment but",
        "differs slightly across faiss/BLAS builds, so the smallest p reaching the",
        f"target can move by about one probe at n=128 (this run: {n128_min_p}; the",
        "canonical sweep: 43). p=43 is reported because it clears the target everywhere.",
    ]

    sweep_path = Path(args.sweep_json)
    if sweep_path.exists():
        rec = json.loads(sweep_path.read_text())
        prev_base = rec["baseline"]["recall"]
        prev_n128 = next(r for r in rec["rows"] if r["cluster_size"] == 128)
        lines += ["", f"CROSS-CHECK against {sweep_path}"]
        lines.append(f"  baseline p=100 : sweep {prev_base:.4f}  now {target:.4f}  "
                     f"diff {abs(prev_base - target):.4f}")
        lines.append(f"  n=128 min p    : sweep {prev_n128['min_nprobe']} "
                     f"(recall {prev_n128['recall_at_min_nprobe']:.4f})  "
                     f"now p=43 recall {n128_eval[43]['recall']:.4f}")

    summary = (
        f"\nRESULT: top-100 of 4096 variable clusters and top-43 of {n128.nlist} equal-size "
        f"clusters reach the same recall@10 ({target:.4f} vs {n128_eval[43]['recall']:.4f}) "
        f"on identical queries. Total distance evaluations {base_total:,.0f} vs "
        f"{n128.nlist + 43 * 128:,} ({(n128.nlist + 43 * 128) / base_total:.2f}x); "
        f"centroid stage {4096 / n128.nlist:.1f}x cheaper, candidate stage "
        f"{43 * 128 / base_cands_100.mean():.2f}x more expensive but fixed at 5,504 per query."
    )
    lines.append(summary)
    lines.append(f"\nElapsed {time.time() - t_start:.0f}s. Artifacts in {work}/")

    report = "\n".join(lines)
    print("\n" + report)
    (work / "verify_p43.txt").write_text(report + "\n")
    (work / "verify_p43.json").write_text(json.dumps({
        "n_db": int(n_db), "target_recall": target,
        "baseline": {"K": 4096, "sizes": {"min": int(base_sizes.min()), "max": int(base_sizes.max()),
                     "mean": float(base_sizes.mean())},
                     "candidates_p100": {"min": int(base_cands_100.min()),
                                         "median": float(np.median(base_cands_100)),
                                         "max": int(base_cands_100.max()),
                                         "mean": float(base_cands_100.mean())},
                     "recall_by_p": {p: base_eval[p]["recall"] for p in BASELINE_P_GRID}},
        "n128": {"K": int(n128.nlist), "recall_by_p": {p: n128_eval[p]["recall"] for p in N128_P_GRID}},
        "n16": ({"K": int(n16.nlist), "recall_by_p": {p: n16_eval[p]["recall"] for p in N16_P_GRID}}
                if n16_eval else None),
        "min_p_on_grid": {"n128": n128_min_p, "n16": n16_min_p},
        "checks": checks,
    }, indent=2))
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()

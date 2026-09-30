#!/usr/bin/env python3
"""
Run the FHE centroid stage for the equal-size clustering (508 centroids, top-43)
over a fixed query set and record correctness + timing.

Per query: encrypt -> homomorphic distances to all 508 centroids (64 packed CKKS
ciphertexts) -> decrypt -> top-43 cluster ids. Each result is checked against the
plaintext ranking shipped in inputs/plaintext_top43.json, and optionally turned
into end-to-end recall@10 if the database vectors are available.

Stdlib only unless --db-vectors is given (then numpy). Resume-safe: re-running
skips queries already in results/per_query.jsonl.

    python run_fhe_topk_n128.py --bin-dir ../build/bin --num-threads 20
    python run_fhe_topk_n128.py --bin-dir ../build/bin --mode wrapper   # one-file binary
"""

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def rotation_indices(lanes: int, padded_dim: int) -> str:
    out, step = [], 1
    while step < padded_dim:
        out.append(str(step * lanes))
        step <<= 1
    return ",".join(out)


def run(cmd, log_path):
    t0 = time.perf_counter()
    proc = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    with open(log_path, "a") as f:
        f.write("$ " + " ".join(str(c) for c in cmd) + "\n" + proc.stdout + proc.stderr + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(str(c) for c in cmd)}\n"
                           f"{proc.stderr[-2000:]}")
    return wall, proc.stdout


class Binaries:
    def __init__(self, bin_dir: Path, mode: str):
        self.mode = mode
        if mode == "split":
            self.keygen = [bin_dir / "openfhe_keygen"]
            self.encrypt = [bin_dir / "openfhe_encrypt_query_centroid_batched"]
            self.compute = [bin_dir / "openfhe_compute_distances_centroid_batched"]
            self.decrypt = [bin_dir / "openfhe_decrypt_topk_centroid_batched"]
            needed = [self.keygen[0], self.encrypt[0], self.compute[0], self.decrypt[0]]
        else:
            w = bin_dir / "openfhe_batched_workload"
            self.keygen = [w, "keygen-centroid"]
            self.encrypt = [w, "encrypt-centroid"]
            self.compute = [w, "compute-centroid"]
            self.decrypt = [w, "decrypt-centroid"]
            needed = [w]
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            sys.exit("missing binaries (build openfhe_core first):\n  " + "\n  ".join(missing))


def load_db_side(args):
    """Optional numpy-side data for end-to-end recall@10."""
    if not args.db_vectors:
        return None
    import numpy as np
    slots = np.load(Path(args.clustering_dir) / "cluster_slots.npy")
    db = np.load(args.db_vectors, mmap_mode="r")
    queries = np.load(Path(args.inputs_dir) / "queries.npy")
    gt = json.loads((Path(args.inputs_dir) / "ground_truth_top10.json").read_text())
    return {"np": np, "slots": slots, "db": db, "queries": queries, "gt": gt}


def recall_from_clusters(side, qi, cluster_ids, k=10):
    np = side["np"]
    cand = side["slots"][cluster_ids].reshape(-1)
    cand = cand[cand >= 0]
    sub = np.asarray(side["db"][cand], dtype=np.float32)
    d2 = ((sub - side["queries"][qi]) ** 2).sum(axis=1)
    top = cand[np.argsort(d2, kind="stable")[:k]]
    truth = set(side["gt"][str(qi)])
    return len(truth & set(int(v) for v in top)) / len(truth), int(len(cand))


def summarize(rows, args, meta):
    def stats(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        if not v:
            return None
        v.sort()
        return {"mean": statistics.fmean(v), "median": statistics.median(v),
                "p95": v[min(len(v) - 1, int(0.95 * len(v)))], "min": v[0], "max": v[-1], "n": len(v)}

    n = len(rows)
    overlaps = [r["overlap_top43"] for r in rows]
    summary = {
        "n_queries": n,
        "n_centroids": meta["n_centroids"], "top_k": args.top_k,
        "ciphertexts_per_query": -(-meta["n_centroids"] // args.centroids_per_ciphertext),
        "params": {"poly_modulus_degree": args.poly_modulus_degree, "padded_dim": args.padded_dim,
                   "centroids_per_ciphertext": args.centroids_per_ciphertext,
                   "coeff_mod_bit_sizes": args.coeff_mod_bit_sizes, "num_threads": args.num_threads,
                   "batch_size": args.batch_size, "mode": args.mode},
        "correctness": {
            "overlap_top43_mean": statistics.fmean(overlaps),
            "overlap_top43_min": min(overlaps),
            "queries_with_exact_set_match": sum(o == args.top_k for o in overlaps),
            "queries_with_exact_order_match": sum(r["exact_order_match"] for r in rows),
            "queries_where_fhe_top43_within_plaintext_top45": sum(r["within_plaintext_top45"] for r in rows),
            "max_abs_distance_error": max(r["max_abs_dist_err"] for r in rows),
            "max_rel_distance_error": max(r["max_rel_dist_err"] for r in rows),
        },
        "timing_seconds": {
            "encrypt_wall": stats("encrypt_wall_s"),
            "compute_wall": stats("compute_wall_s"),
            "compute_kernel": stats("compute_kernel_s"),
            "decrypt_wall": stats("decrypt_wall_s"),
            "total_wall": stats("total_wall_s"),
        },
    }
    if rows and rows[0].get("recall_at10_fhe") is not None:
        summary["end_to_end"] = {
            "recall_at10_from_fhe_top43_mean": statistics.fmean(r["recall_at10_fhe"] for r in rows),
            "recall_at10_from_plaintext_top43_mean": statistics.fmean(r["recall_at10_plain"] for r in rows),
            "candidates_per_query": rows[0]["candidates_fhe"],
        }
    return summary


def write_summary_md(path, s):
    t = s["timing_seconds"]
    c = s["correctness"]

    def row(name, st):
        if not st:
            return f"| {name} | – | – | – |"
        return f"| {name} | {st['mean']:.2f} | {st['median']:.2f} | {st['p95']:.2f} |"

    lines = [
        "# FHE centroid stage — equal-size n=128 (508 centroids, top-43)", "",
        f"- queries: {s['n_queries']}  |  centroids: {s['n_centroids']}  |  "
        f"ciphertexts/query: {s['ciphertexts_per_query']}  |  threads: {s['params']['num_threads']}",
        f"- CKKS: N={s['params']['poly_modulus_degree']}, padded_dim={s['params']['padded_dim']}, "
        f"{s['params']['centroids_per_ciphertext']} centroids/ciphertext, "
        f"coeff bits {s['params']['coeff_mod_bit_sizes']}", "",
        "## Correctness (FHE top-43 vs plaintext top-43 on the same centroids)", "",
        f"- mean overlap: **{c['overlap_top43_mean']:.2f} / {s['top_k']}**, min {c['overlap_top43_min']}",
        f"- exact set match: {c['queries_with_exact_set_match']} / {s['n_queries']} queries; "
        f"exact order match: {c['queries_with_exact_order_match']}",
        f"- FHE top-43 contained in plaintext top-45: "
        f"{c['queries_where_fhe_top43_within_plaintext_top45']} / {s['n_queries']}",
        f"- max |FHE − plaintext| squared distance: {c['max_abs_distance_error']:.3e} "
        f"(relative {c['max_rel_distance_error']:.3e})", "",
    ]
    if "end_to_end" in s:
        e = s["end_to_end"]
        lines += ["## End-to-end (fetch the 43 clusters, exact rerank, recall@10 vs brute force)", "",
                  f"- from FHE top-43: **{e['recall_at10_from_fhe_top43_mean']:.4f}**",
                  f"- from plaintext top-43: {e['recall_at10_from_plaintext_top43_mean']:.4f}",
                  f"- candidates fetched per query: {e['candidates_per_query']} (= 43 × 128)", ""]
    lines += ["## Timing per query (seconds)", "", "| stage | mean | median | p95 |", "|---|---|---|---|",
              row("encrypt (wall)", t["encrypt_wall"]), row("compute (wall, incl. key load)", t["compute_wall"]),
              row("compute (kernel only)", t["compute_kernel"]), row("decrypt (wall)", t["decrypt_wall"]),
              row("total (wall)", t["total_wall"]), ""]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", type=Path, default=REPO / "openfhe_core" / "build" / "bin")
    ap.add_argument("--mode", choices=["split", "wrapper"], default="split")
    ap.add_argument("--centroids-file", type=Path,
                    default=REPO / "openfhe_core" / "openfhe_inputs_n128" / "centroids.txt")
    ap.add_argument("--inputs-dir", type=Path, default=HERE / "inputs")
    ap.add_argument("--work-dir", type=Path, default=HERE / "work")
    ap.add_argument("--results-dir", type=Path, default=HERE / "results")
    ap.add_argument("--top-k", type=int, default=43)
    ap.add_argument("--num-threads", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--poly-modulus-degree", type=int, default=16384)
    ap.add_argument("--padded-dim", type=int, default=1024)
    ap.add_argument("--centroids-per-ciphertext", type=int, default=8)
    ap.add_argument("--coeff-mod-bit-sizes", default="60,40,40,40,40,60")
    ap.add_argument("--security-level", default=None, help="passed through to keygen if set")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-ciphertexts", action="store_true",
                    help="keep per-query encrypted distance dirs (~100 MB each)")
    ap.add_argument("--db-vectors", type=Path, default=None,
                    help="prototype/pir_n128/vectors_65k.npy -> also computes end-to-end recall@10")
    ap.add_argument("--clustering-dir", type=Path, default=REPO / "wiki-rag" / "ivf_output_n128")
    args = ap.parse_args()

    bins = Binaries(args.bin_dir.resolve(), args.mode)
    meta = json.loads((args.inputs_dir / "inputs_metadata.json").read_text())
    plain = json.loads((args.inputs_dir / "plaintext_top43.json").read_text())
    queries = [ln for ln in (args.inputs_dir / "queries.txt").read_text().splitlines() if ln.strip()]
    n_centroids = meta["n_centroids"]
    assert sum(1 for ln in args.centroids_file.read_text().splitlines() if ln.strip()) == n_centroids, \
        f"{args.centroids_file} does not have {n_centroids} rows"
    side = load_db_side(args)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    log = args.results_dir / "commands.log"
    per_query = args.results_dir / "per_query.jsonl"
    done = {}
    if per_query.exists():
        for ln in per_query.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done[r["query_id"]] = r

    # ---- keygen once ------------------------------------------------------
    ctx = args.work_dir / "context"
    if not (ctx / "context.bin").exists():
        cmd = bins.keygen + ["--context-dir", ctx, "--poly-modulus-degree", args.poly_modulus_degree,
                             "--coeff-mod-bit-sizes", args.coeff_mod_bit_sizes,
                             "--rotation-indices", rotation_indices(args.centroids_per_ciphertext, args.padded_dim)]
        if args.mode == "wrapper":
            cmd += ["--padded-dim", args.padded_dim, "--centroids-per-ciphertext", args.centroids_per_ciphertext]
        if args.security_level:
            cmd += ["--security-level", args.security_level]
        wall, _ = run(cmd, log)
        print(f"keygen: {wall:.1f}s -> {ctx}")
    else:
        print(f"keygen: reusing {ctx}")

    # ---- per query --------------------------------------------------------
    end = len(queries) if args.limit is None else min(len(queries), args.start + args.limit)
    kernel_re = re.compile(r"in ([0-9.]+) s")
    for qi in range(args.start, end):
        if qi in done:
            continue
        qdir = args.work_dir / f"q{qi:04d}"
        enc, dist = qdir / "enc", qdir / "dist"
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "query.txt").write_text(queries[qi] + "\n")

        t_all = time.perf_counter()
        enc_wall, _ = run(bins.encrypt + ["--context-dir", ctx, "--input-vector", qdir / "query.txt",
                                          "--output-dir", enc, "--centroids-per-ciphertext",
                                          args.centroids_per_ciphertext, "--padded-dim", args.padded_dim], log)
        comp_wall, comp_out = run(bins.compute + [
            "--context-dir", ctx, "--centroids-file", args.centroids_file,
            "--encrypted-query", enc / "encrypted_query_centroid_batched.bin",
            "--encrypted-norm", enc / "encrypted_norm_centroid_batched.bin",
            "--output-dir", dist, "--centroids-per-ciphertext", args.centroids_per_ciphertext,
            "--padded-dim", args.padded_dim, "--num-threads", args.num_threads,
            "--batch-size", args.batch_size], log)
        m = kernel_re.search(comp_out)
        kernel_s = float(m.group(1)) if m else None
        # Decrypt the full ranking once: top-43 is its prefix, and the full list
        # lets us measure CKKS error on every centroid.
        dec_wall, _ = run(bins.decrypt + ["--context-dir", ctx, "--encrypted-distances-dir", dist,
                                          "--top-k", n_centroids, "--output-json", qdir / "top_k_results.json"], log)
        total = time.perf_counter() - t_all

        res = json.loads((qdir / "top_k_results.json").read_text())
        fhe_order = [int(i) for i in res["centroid_indices"]]
        fhe_dist = {int(i): float(d) for i, d in zip(res["centroid_indices"], res["distances"])}
        fhe_top = fhe_order[:args.top_k]
        pl = plain[str(qi)]
        plain_top = pl["centroid_ids"][:args.top_k]
        plain_all = pl["all_sq_dists"]
        errs = [abs(fhe_dist[c] - plain_all[c]) for c in fhe_dist]
        rels = [abs(fhe_dist[c] - plain_all[c]) / max(abs(plain_all[c]), 1e-12) for c in fhe_dist]
        plain_top45 = set(pl["centroid_ids"][:args.top_k + 2])

        row = {
            "query_id": qi,
            "fhe_top43": fhe_top,
            "plaintext_top43": plain_top,
            "overlap_top43": len(set(fhe_top) & set(plain_top)),
            "exact_order_match": fhe_top == plain_top,
            "within_plaintext_top45": set(fhe_top) <= plain_top45,
            "max_abs_dist_err": max(errs), "max_rel_dist_err": max(rels),
            "encrypt_wall_s": enc_wall, "compute_wall_s": comp_wall, "compute_kernel_s": kernel_s,
            "decrypt_wall_s": dec_wall, "total_wall_s": total,
        }
        if side:
            row["recall_at10_fhe"], row["candidates_fhe"] = recall_from_clusters(side, qi, fhe_top)
            row["recall_at10_plain"], _ = recall_from_clusters(side, qi, plain_top)
        with per_query.open("a") as f:
            f.write(json.dumps(row) + "\n")
        done[qi] = row
        print(f"q{qi:04d}  overlap {row['overlap_top43']}/{args.top_k}  "
              f"order={'=' if row['exact_order_match'] else '~'}  "
              f"err {row['max_abs_dist_err']:.1e}  enc {enc_wall:.1f}s  comp {comp_wall:.1f}s "
              f"(kernel {kernel_s if kernel_s is not None else float('nan'):.1f}s)  dec {dec_wall:.1f}s"
              + (f"  recall@10 {row['recall_at10_fhe']:.2f}" if side else ""), flush=True)

        if not args.keep_ciphertexts:
            shutil.rmtree(dist, ignore_errors=True)
            shutil.rmtree(enc, ignore_errors=True)

        rows = [done[k] for k in sorted(done)]
        summary = summarize(rows, args, meta)
        (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        write_summary_md(args.results_dir / "summary.md", summary)
        (args.results_dir / "fhe_top43_cluster_ids.json").write_text(
            json.dumps({str(r["query_id"]): r["fhe_top43"] for r in rows}))

    print(f"\ndone: {len(done)} queries. Results in {args.results_dir}/ "
          f"(summary.md, summary.json, per_query.jsonl, fhe_top43_cluster_ids.json)")


if __name__ == "__main__":
    main()

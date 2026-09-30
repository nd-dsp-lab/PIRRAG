# FHE centroid stage for equal-size clustering — run instructions

Runs the encrypted query → 508 centroid distances → decrypt top-43 pipeline over
100 fixed queries and records correctness and timing. Everything except the
OpenFHE binaries is already in this directory.

## What is here

| path | what |
|---|---|
| `inputs/queries.txt` / `queries.npy` | 100 queries (768-d), seeded; perturbed database vectors |
| `inputs/plaintext_top43.json` | per query: plaintext top-43 centroid ids and exact squared distances to all 508 centroids |
| `inputs/ground_truth_top10.json` | brute-force top-10 vector ids per query (for end-to-end recall) |
| `inputs/oracle_top10_p43.json` | plaintext ANN result at p=43 (what the full pipeline should reproduce) |
| `inputs/inputs_metadata.json` | parameters; oracle recall@10 at p=43 on these queries = 0.9100 |
| `../openfhe_inputs_n128/centroids.txt` | the 508 centroids, FHE text format |
| `../openfhe_inputs_n128/fhe_layout.json` | packing: 8 centroids/ciphertext → 64 ciphertexts, rotation indices 8…4096 |
| `run_fhe_topk_n128.py` | the runner (stdlib only) |

## 1. Build OpenFHE binaries (skip if `openfhe_core/build/bin` exists)

```bash
cmake -S openfhe_core -B openfhe_core/build -DOpenFHE_DIR=/usr/local/lib/OpenFHE
cmake --build openfhe_core/build -j
```

## 2. Run

From the repo root, on the machine with the binaries:

```bash
python3 -u openfhe_core/fhe_run_n128/run_fhe_topk_n128.py \
    --bin-dir openfhe_core/build/bin --num-threads 20
```

- Uses the split binaries (`openfhe_keygen`, `openfhe_encrypt_query_centroid_batched`,
  `openfhe_compute_distances_centroid_batched`, `openfhe_decrypt_topk_centroid_batched`).
  If only the one-file `openfhe_batched_workload` is built, add `--mode wrapper`.
- Keys are generated once (N=16384, coeff bits 60,40,40,40,40,60, rotation
  indices 8…4096) and reused for all queries.
- Expected time: the 4096-centroid run took 49.5 s/query at 20 threads; this is
  64 ciphertexts instead of 512, so expect ~6–8 s/query, ~15 min for 100.
- Resume-safe: re-run the same command after an interruption. `--limit 3` for a
  smoke test first.
- Per-query ciphertexts (~100 MB) are deleted after decryption; `--keep-ciphertexts` to keep.

### Optional: end-to-end recall@10

If `prototype/pir_n128/vectors_65k.npy` (200 MB) is on the machine, add

```bash
    --db-vectors prototype/pir_n128/vectors_65k.npy
```

and the runner also fetches the members of the FHE-selected 43 clusters from
`wiki-rag/ivf_output_n128/cluster_slots.npy`, reranks exactly, and reports
recall@10 against brute force — i.e. the number the full FHE+PIR pipeline should
reproduce, without PIR in the loop. Needs numpy.

## 3. What gets recorded (`results/`)

| file | content |
|---|---|
| `summary.md` / `summary.json` | overlap of FHE top-43 with plaintext top-43 (mean/min, exact-set and exact-order counts), max CKKS distance error, per-stage timing mean/median/p95, end-to-end recall if enabled |
| `per_query.jsonl` | one line per query with all of the above |
| `fhe_top43_cluster_ids.json` | `{query_id: [43 cluster ids]}` — hand this to the PIR side; rows to fetch are `c*128 + i` |
| `commands.log` | every binary invocation with its stdout/stderr |

Send back the whole `results/` directory (a few hundred KB).

## 4. What to expect, and one caveat

The decrypt binary sorts plaintext values after decryption, so the top-43 is exact
up to CKKS noise. Across the 100 queries the smallest gap between the 43rd and
44th plaintext centroid distance is 1.4e-6 (median 4e-4); CKKS at these parameters
typically resolves ~1e-5 or better, so a handful of queries may swap the boundary
centroid. The runner reports this separately (`within_plaintext_top45`) rather than
counting it as an error, and the end-to-end recall check shows whether it matters
(it should not: a boundary swap exchanges the 43rd- and 44th-nearest cluster).

Distances come out as exact squared L2, `|q|^2 + |c|^2 - 2 q·c`; the max absolute
error against `plaintext_top43.json` is the CKKS precision at these parameters and
is worth quoting alongside the timings.

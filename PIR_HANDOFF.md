# PIR Handoff — Equal-Size Clustering (n=128, K=508)

Everything you need is already generated and verified. This file tells you which
files to use, what the addressing logic is, and how to check your PIR path against
a plaintext oracle.

## 1. The operating point

| | old (what you ran before) | **new (use this)** |
|---|---|---|
| clustering | k-means, K=4096, sizes 1–102 | k-means centroids + balanced assignment, **K=508, every cluster exactly 128** |
| clusters probed per query (p) | 100 | **43** |
| records fetched per query | ~2,310 mean, 1,574–3,259 (varies) | **5,504 = 43 × 128, always** |
| PIR address of (cluster c, index i) | `lists.json[c][i]` (needs the 768 KB map on the client) | **`row = c * 128 + i`** (client needs only the integer 128) |
| FHE centroid stage | 4,096 centroids / 512 ciphertexts | 508 centroids / 64 ciphertexts |

Why p=43: it is the smallest p at which top-43-of-508 reaches the same recall@10
as top-100-of-4096 on the same 1,000 held-out queries (0.9529 vs 0.9519). The
plaintext proof is `wiki-rag/verify_p43/verify_p43.txt` (re-run with
`wiki-rag/verify_p43.py`). Full reasoning: `EQUAL_SIZE_PAPER_SECTION.md`.

## 2. Files — exact paths

All produced from one clustering run on the full 65,000-vector corpus
(`--recall-mode perturbed`, so vector ids are the FAISS row ids you already use).
**Do not use anything under `wiki-rag/verify_p43/` or `wiki-rag/cluster_size_sweep/`
for PIR** — those are holdout clusterings whose ids index a 64,000-vector subset.

| what | path | shape / format |
|---|---|---|
| **Read first** — sizing mode, K, n, id space | `wiki-rag/ivf_output_n128/ivf_metadata.json` | `sizing_mode: constant-size`, `nlist: 508`, `cluster_size: 128`, `id_space: database` |
| cluster membership (canonical) | `wiki-rag/ivf_output_n128/cluster_slots.npy` | `(508, 128) int32`; `slots[c][i]` = global vector id, `-1` = padding (24 slots total) |
| cluster membership (compat view) | `wiki-rag/ivf_output_n128/lists.json` | `{"c": [128 ids]}`, `-1` padded — evaluation only, not needed by the client |
| centroids (numpy) | `wiki-rag/ivf_output_n128/centroids.npy` | `(508, 768) float32` |
| centroids (FHE text input) | `openfhe_core/openfhe_inputs_n128/centroids.txt` | 508 rows × 768 whitespace-separated floats — same numbers as the .npy |
| FHE packing layout | `openfhe_core/openfhe_inputs_n128/fhe_layout.json` | 8 centroids/ciphertext, 64 ciphertexts, last one 4 of 8 lanes |
| **PIR vector database, cluster-major** | `prototype/pir_n128/pir_vectors_n128.npy` | `(65024, 768) float32`; row index **is** the PIR address; 24 all-zero padding rows |
| its manifest | `prototype/pir_n128/pir_vectors_n128.npy.manifest.json` | `address_formula: cluster_id * cluster_size + index`, `n_rows: 65024` |
| original vectors in FAISS row order (for your own checks) | `prototype/pir_n128/vectors_65k.npy` | `(65000, 768) float32` |

The `--verify` pass on the PIR export resolved 2,000 random `(cluster, index)` pairs
through `c*128+i` against the clustering, confirmed every vector appears in exactly
one row, and that only the 24 padding rows are blank.

The two 200 MB `.npy` files are not meant to be committed; regenerate with the
commands in §5 if you need them elsewhere.

## 3. The logic, end to end

```
client:  encrypt query q                                         (FHE, unchanged kernel)
server:  d_c = ||q - centroid_c||^2  for c in 0..507             (64 packed ciphertexts)
client:  decrypt, take the 43 smallest d_c  -> cluster ids C     (plaintext sort)
client:  for c in C: for i in 0..127: PIR-fetch row c*128 + i    (5,504 rows, fixed)
client:  drop padding rows (all-zero vector / blank record)      (at most 24 exist in the whole DB)
client:  exact rerank of the fetched vectors, return top-10
```

No `lists.json`, no offsets table, no per-cluster size on the client. The 43
cluster ids come from the FHE stage's `top_k_results.json` exactly as before — only
the `--top-k` argument changes from 100 to 43.

## 4. What changes in your code

`batched_test/run_pir_single_query_loop.py:149-156` currently maps
`(cluster_id, index) -> global id` by indexing into `lists.json`. Replace with
`row = cluster_id * 128 + index` against the cluster-major database, and skip rows
that are padding. Everything else (query JSON format, `top_k_centroid_indices`,
`convert_pir_to_eval_format.py`) keeps working.

If your PIR server serves the **document text** database rather than vectors,
reorder it the same way — same tool, same clustering:

```bash
cd prototype
../.venv-ivf/bin/python pir_export.py \
    --clustering ../wiki-rag/ivf_output_n128 \
    --records ../uniform_index.txt \
    --output pir_n128/pir_db_n128.txt --verify
```

Records must be in FAISS row order (as `pickle_processor.py` writes them). Padding
rows come out blank at the same fixed width.

## 5. Checking your PIR path against the plaintext oracle

The clustering is deterministic once written, so the oracle and your PIR pipeline
must return identical top-10 sets for any query:

```python
import sys; sys.path.insert(0, "wiki-rag")
import numpy as np, ivf_io, ivf_eval

c = ivf_io.load_clustering("wiki-rag/ivf_output_n128", load_centroids=True)
db = np.load("prototype/pir_n128/vectors_65k.npy")          # FAISS row order
q = ...                                                       # (1, 768) float32 query

idx, dist, _ = ivf_eval.ivf_search_from_slots(q, db, c.centroids, c.slots, nprobe=43, k=10)
# idx[0] are global vector ids (FAISS rows). Your PIR path, after reranking the
# 5,504 fetched rows, must produce the same 10 ids.

# Address check for any (cluster, index):
pir = np.load("prototype/pir_n128/pir_vectors_n128.npy")
cid, i = 7, 3
assert np.array_equal(pir[cid * 128 + i], db[c.slots[cid, i]])
```

`prototype/rag_operations/plaintext_rag_pipeline.py` and
`prototype/compute_ground_truth.py` also read this clustering directly via
`ivf_io.load_clustering` and serve as the end-to-end reference.

## 6. Regenerating everything (about 3 minutes)

```bash
INDEX=~/.cache/huggingface/hub/models--royrin--wiki-rag/snapshots/*/wiki_index__top_100000__2025-04-11/index.faiss

cd wiki-rag
../.venv-ivf/bin/python train_ivf.py --faiss-index-file $INDEX --cluster-size 128 \
    --recall-mode perturbed --output-dir ivf_output_n128
../.venv-ivf/bin/python fhe_prepare.py --ivf-dir ivf_output_n128 \
    --out-dir ../openfhe_core/openfhe_inputs_n128

cd ../prototype
../.venv-ivf/bin/python -c "import faiss,numpy as np,glob; i=faiss.read_index(glob.glob('$INDEX')[0]); np.save('pir_n128/vectors_65k.npy', i.reconstruct_n(0,i.ntotal))"
../.venv-ivf/bin/python pir_export.py --clustering ../wiki-rag/ivf_output_n128 \
    --vectors-npy pir_n128/vectors_65k.npy --output pir_n128/pir_vectors_n128.npy --verify
```

## 7. One thing not to misread

`wiki-rag/ivf_output_n128/ivf_metadata.json` reports `perturbed_recall@10 = 0.8992`
at p=43. That is a different metric (queries are noise-perturbed database vectors,
used only so the shipped artifact is measured on the whole corpus). The p=43 ≡
p=100 equivalence was established under **holdout** recall (0.9529 vs 0.9519) in
`wiki-rag/verify_p43/`. Do not compare the two numbers.

## 8. The alternative, if PIR bandwidth is your binding constraint

`padded-naive/README.md` §4: keeping k-means and splitting over-full clusters into
fixed-width chunks fetches 3,168 records/query instead of 5,504 (−42%) at the same
recall, at the cost of 1,249 vs 500 centroid comparisons. Same `pir_export.py`,
same manifest format (with an `offsets` table). Raise it before building if 5,504
rows/query is a problem for your PIR scheme.

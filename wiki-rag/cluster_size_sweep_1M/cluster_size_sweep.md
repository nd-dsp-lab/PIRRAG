# Constant cluster size sweep

- Database: **701,873** vectors, dim 768
- Metric: **holdout_recall@10** over 1000 queries (seed 1234, shared across every row)
- Index: ivf-flat, niter=20

## Baseline (fixed cluster count, variable sizes)

| nlist | nprobe | cluster sizes | centroid dists | candidates | total dists | % DB | holdout_recall@10 |
|---|---|---|---|---|---|---|---|
| 43867 | 1070 | min 1, max 186, mean 16.00 | 43,867 | ~26,046 (varies) | 69,913 | 3.71% | 0.9855 |

## Constant size: smallest p reaching recall 0.9855

Ranked by **total distance evaluations** (centroid stage + candidate stage),
since both are homomorphic comparisons and constant-size clustering trades
one against the other.

| n | x = ceil(N/n) | pad | p | centroid | candidates (p*n) | total | vs base | % DB | holdout_recall@10 | balance penalty | forced | ms (numpy) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 5484 | 79 | 496 | 5,484 | 63,488 | 68,972 | 0.99x | 9.05% | 0.9855 | 1.0528 | 27.2% | 45.48 |
| 256 | 2742 | 79 | 232 | 2,742 | 59,392 | 62,134 | 0.89x | 8.46% | 0.9856 | 1.0371 | 23.2% | 40.37 |

**Equivalent operating point: n = 256, p = 232.** 62,134 total distance evaluations versus 69,913 for the baseline (0.89x), at the same recall, with every cluster exactly 256 wide.

The candidate stage alone costs 2.28x the baseline -- equal sizes give up the baseline's implicit adaptivity, where a query landing in a dense region automatically pulls larger clusters. What pays for it is the centroid stage: 43,867 -> 2,742 centroids, 16.0x fewer encrypted comparisons.

## Notes

- `total dists` counts nlist centroid comparisons plus p*n candidate
  comparisons. Ranking on candidates alone would pick a different (worse) n.
- Latency is numpy, **not** comparable with `recall_analysis.txt`, which was
  measured through FAISS's SIMD search path.
- The baseline recall here is re-measured in `holdout` mode,
  not taken from `recall_analysis.txt` (whose queries came from the database
  itself, making every query its own guaranteed top-1 hit).
- With constant size, candidates scanned is exactly `p*n` for every query.
  The baseline's candidate count varies per query with the sizes of whichever
  clusters were probed.

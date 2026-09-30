# Constant cluster size sweep

- Database: **1,119,486** vectors, dim 768
- Metric: **holdout_recall@10** over 1000 queries (seed 1234, shared across every row)
- Index: ivf-flat, niter=20

## Baseline (fixed cluster count, variable sizes)

| nlist | nprobe | cluster sizes | centroid dists | candidates | total dists | % DB | holdout_recall@10 |
|---|---|---|---|---|---|---|---|
| 69968 | 1707 | min 1, max 183, mean 16.00 | 69,968 | ~42,236 (varies) | 112,204 | 3.77% | 0.9915 |

## Constant size: smallest p reaching recall 0.9915

Ranked by **total distance evaluations** (centroid stage + candidate stage),
since both are homomorphic comparisons and constant-size clustering trades
one against the other.

| n | x = ceil(N/n) | pad | p | centroid | candidates (p*n) | total | vs base | % DB | holdout_recall@10 | balance penalty | forced | ms (numpy) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 8746 | 2 | 1055 | 8,746 | 135,040 | 143,786 | 1.28x | 12.06% | 0.9915 | 1.0524 | 27.0% | 108.67 |
| 256 | 4373 | 2 | 546 | 4,373 | 139,776 | 144,149 | 1.28x | 12.49% | 0.9915 | 1.0360 | 23.4% | 116.10 |

**Equivalent operating point: n = 128, p = 1055.** 143,786 total distance evaluations versus 112,204 for the baseline (1.28x), at the same recall, with every cluster exactly 128 wide.

The candidate stage alone costs 3.20x the baseline -- equal sizes give up the baseline's implicit adaptivity, where a query landing in a dense region automatically pulls larger clusters. What pays for it is the centroid stage: 69,968 -> 8,746 centroids, 8.0x fewer encrypted comparisons.

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

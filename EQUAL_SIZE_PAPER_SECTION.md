# Equal-Size Clustering (paper section draft)

Drop-in text. A LaTeX version of the table is at the bottom, followed by the
command that reproduces every number in plaintext.

> **Note to authors — one correction to the branch READMEs.** The sweep
> (`cluster_size_sweep.json`, `README_EQUAL_SIZE.md` §2, `padded-naive/README.md`
> §3) reports the baseline's candidate count as ~1,562 = 100 × mean cluster size.
> Measured per query it is **2,310 on average (1,574–3,259)**: the clusters nearest
> a query are larger than average, because queries land in dense regions. With the
> correct baseline the equal-size n=128 configuration is **0.94× the baseline's
> total work, not 1.06×**, and its candidate stage is 2.4×, not 3.5×. The
> equal-size rows themselves are exact (candidates = p·n by construction); only the
> baseline row and the "vs. base" ratios change. This section uses the corrected
> figures. The sweep script is fixed; the READMEs and the saved JSON still carry
> the old number until the sweep is re-run.

---

## Balancing the IVF partition

**Motivation.** The retrieval index is an inverted-file (IVF) partition: the
encrypted query is compared against every cluster centroid under FHE, the client
decrypts and selects the top-$p$ clusters, and then fetches the members of those
clusters over PIR before reranking them locally. Plain $k$-means with $K = 4096$
produces a highly unbalanced partition on our 64{,}000-vector database: cluster
sizes range from 1 to 102 (mean 15.6). This has two consequences for a private
pipeline. First, the number of records a query fetches at fixed $p$ is
data-dependent: at $p = 100$ it ranges from 1{,}574 to 3{,}259 records across our
query set (mean 2{,}310), so the size of the PIR access reveals how densely
populated the probed region of the embedding space is. Second, the client-side
rerank has no fixed shape, which complicates any future homomorphic per-cluster
computation. Both are removed if every cluster holds exactly $n$ vectors: every
query then touches exactly $p \cdot n$ records, and a record's address is simply
$\text{cluster} \cdot n + \text{index}$.

**Method.** We keep $k$-means centroids and replace only the assignment step.
Given a target cluster size $n$, the number of clusters is fixed to
$K = \lceil N / n \rceil$ and $k$-means is trained for that $K$. Vectors are then
assigned by solving the capacity-constrained problem

$$\min_{A}\ \sum_{x} \lVert x - c_{A(x)} \rVert^2 \quad \text{s.t.}\quad |A^{-1}(k)| = n\ \ \forall k,$$

a transportation problem whose LP dual we solve by subgradient (price) ascent.
Each cluster carries a price $\lambda_k$; every vector chooses
$\arg\min_k\, \lVert x - c_k\rVert^2 + \lambda_k$; over-subscribed clusters raise
their price and the process repeats until capacities clear. Because every vector
re-chooses at the current prices on every round, the result has no
first-come-first-served bias, unlike a greedy sort-and-fill. For tractability each
vector is restricted to its $m = 32$ nearest centroids ($8.3$ MB instead of a
$1.06$ GB dense cost matrix); the few vectors whose entire candidate set fills are
placed by an exact scan over clusters with remaining capacity, so the assignment
always terminates feasible. A Lagrangian lower bound certifies the solution is
within 2.1% of the optimal equal-size assignment at $n = 128$. The whole procedure
takes seconds on $64\text{k} \times 768$.

The cost of the constraint is small: the mean squared distance from a vector to
its assigned centroid is $1.054\times$ that to its unconstrained nearest centroid
at $n = 128$ (25% of vectors are displaced from their nearest centroid, most by a
negligible margin).

**Choosing $n$.** Balancing is not free at query time: uniform sizes discard the
implicit adaptivity of $k$-means, in which dense regions receive large clusters
and a query landing there automatically draws a larger candidate pool. To choose
$n$ we fix a recall target and measure, for each $n$, the smallest $p$ that reaches
it. The target is the baseline's own recall — top-100 of 4096 variable clusters —
measured on 1{,}000 queries held out of both the database and the clustering
(self-retrieval recall, in which every query is its own top-1 hit, would penalise
exactly the vector displacement that balancing introduces). All configurations use
the same seeded query split and exact reranking, so recall is monotone in $p$ and
the minimal $p$ is found by binary search. We count the cost of a query as the
number of distance evaluations in both stages: $K$ homomorphic centroid
comparisons plus the number of fetched candidates, which is measured per query for
the baseline and is exactly $p \cdot n$ for the equal-size partitions.

| configuration | $K$ | $p$ | recall@10 | centroid | candidates | total | vs. base | penalty |
|---|---|---|---|---|---|---|---|---|
| $k$-means, variable | 4096 | 100 | 0.9519 | 4{,}096 | 2{,}310 (1{,}574–3{,}259) | 6{,}406 | 1.00× | — |
| equal-size $n=16$ | 4000 | 419 | 0.9519 | 4{,}000 | 6{,}704 | 10{,}704 | 1.67× | 1.149 |
| equal-size $n=32$ | 2000 | 163 | 0.9519 | 2{,}000 | 5{,}216 | 7{,}216 | 1.13× | 1.103 |
| equal-size $n=64$ | 1000 | 85 | 0.9521 | 1{,}000 | 5{,}440 | 6{,}440 | 1.01× | 1.081 |
| **equal-size $n=128$** | **500** | **43** | **0.9529** | **500** | **5{,}504** | **6{,}004** | **0.94×** | **1.054** |

Three observations determine the choice. (i) The candidate stage costs
$2.3$–$2.9\times$ the baseline's mean fetch at *every* $n$ — 5{,}216 to 6{,}704
records across an $8\times$ range of $n$. This is the intrinsic price of removing
adaptivity, not a property of any particular $n$. (ii) The centroid stage scales
as $1/n$, from 4{,}000 comparisons to 500. Since the candidate penalty is paid
regardless, the total is minimised by the largest $n$ tested: $n = 128$ gives
$K = 500$ and a total 6% *below* the baseline at slightly higher recall (0.9529 vs
0.9519). (iii) Keeping $K = 4096$ and merely balancing it — i.e. $n = 16$ — is the
worst option: it needs $p = 419$ probes for the same recall ($1.67\times$ the
baseline's work) and has the largest balance penalty (1.149, with 34% of vectors
displaced), because small capacities force many more vectors away from their
nearest centroid. Balancing the partition therefore changes the right operating
point, and the equivalent of "top-100 of 4096" is "top-43 of 500". We did not
sweep beyond $n = 128$; since the candidate stage is flat, the remaining 500
centroid comparisons bound any further gain to under 9% of the total.

**Consequences for the private pipeline.** At $n = 128$, $p = 43$ every query
fetches exactly $43 \times 128 = 5{,}504$ records, independent of where it lands
($2.4\times$ the baseline's mean fetch and $1.7\times$ its worst case, in exchange
for a fetch size that carries no information about the query); the record address
is $128 \cdot \text{cluster} + \text{index}$, so the client needs no cluster map.
The FHE centroid stage compares against 500 centroids instead of 4{,}096 — 63
packed CKKS ciphertexts instead of 512 at eight centroids per ciphertext — an
$8.1\times$ reduction in homomorphic work for that stage. The existing
centroid-distance kernel is unchanged; only the centroid file is regenerated.

*(Optional, if space permits.)* Where PIR bandwidth rather than homomorphic
compute is the binding constraint, an alternative that keeps the $k$-means
partition and splits over-full clusters into fixed-width chunks fetches 42% fewer
records at the same recall (3{,}168 vs 5{,}504) at the cost of $2.5\times$ more
centroid comparisons; the two tie on total work (5{,}985 vs 6{,}004). We report the
equal-size variant as the primary configuration for its simpler fixed shape and
cheaper FHE stage.

**Reproducibility note (footnote or appendix).** The baseline reproduces exactly
(recall 0.9519 in every run). The price-ascent assignment reproduces exactly within
one environment but differs slightly across FAISS/BLAS builds, moving the smallest
$p$ that reaches the target by about one probe at $n = 128$ (42 or 43; recall at
$p = 43$ is 0.9529–0.9534). We report $p = 43$ because it clears the target in
every environment observed.

---

## LaTeX table

```latex
\begin{table}[t]
\centering
\small
\caption{Operating points matched at the baseline's recall@10 (top-100 of 4096
$k$-means clusters) on 1{,}000 held-out queries over a 64{,}000-vector database.
\emph{Candidates} is the number of records fetched per query (for the baseline:
mean, with the observed range); \emph{total} is centroid comparisons plus
candidates; \emph{penalty} is the mean squared distance to the assigned centroid
relative to the unconstrained nearest centroid.}
\label{tab:equal-size}
\begin{tabular}{lrrrrrrrr}
\toprule
configuration & $K$ & $p$ & recall@10 & centroid & candidates & total & vs.\ base & penalty \\
\midrule
$k$-means, variable sizes & 4096 & 100 & 0.9519 & 4{,}096 & 2{,}310 \scriptsize(1{,}574--3{,}259) & 6{,}406 & 1.00$\times$ & --- \\
equal-size, $n=16$        & 4000 & 419 & 0.9519 & 4{,}000 & 6{,}704 & 10{,}704 & 1.67$\times$ & 1.149 \\
equal-size, $n=32$        & 2000 & 163 & 0.9519 & 2{,}000 & 5{,}216 & 7{,}216  & 1.13$\times$ & 1.103 \\
equal-size, $n=64$        & 1000 &  85 & 0.9521 & 1{,}000 & 5{,}440 & 6{,}440  & 1.01$\times$ & 1.081 \\
\textbf{equal-size, $n=128$} & \textbf{500} & \textbf{43} & \textbf{0.9529} & \textbf{500} & \textbf{5{,}504} & \textbf{6{,}004} & \textbf{0.94$\times$} & \textbf{1.054} \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Reproducing the numbers in plaintext

```bash
cd wiki-rag
../.venv-ivf/bin/python verify_p43.py \
    --faiss-index-file ~/.cache/huggingface/hub/models--royrin--wiki-rag/snapshots/*/wiki_index__top_100000__2025-04-11/index.faiss \
    --work-dir verify_p43 --with-n16
```

About two minutes. Rebuilds the baseline, the $n=128$ clustering, and the $n=16$
control from the raw vectors; scores all three on the same held-out queries with
exact reranking; prints recall@10 against $p$ for each, the baseline's per-query
fetch distribution, and pass/fail checks. Output: `wiki-rag/verify_p43/verify_p43.txt`.
The equal-size rows of the table above (p, candidates, penalty) come from the
canonical sweep `wiki-rag/cluster_size_sweep/cluster_size_sweep.json`; the
baseline's candidate figures come from the verification run.

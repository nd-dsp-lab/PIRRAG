# Overleaf edits: equal-size clustering and probe selection

Upload `figures/probe_selection.pdf` to Overleaf (same folder as the other figures).
Each block below says **where** it goes and **what it replaces**.

---

## 1. §II-B, the `IVF-Flat.Setup` definition — replace the bullet

Old: *"IVF-Flat.Setup(DB', c) → C: Given an embedding database DB' and target number of centroids c …"*

```latex
\item \textsf{IVF-Flat.Setup}$(\mathit{DB}', n) \rightarrow \mathcal{C}$: Given an embedding database $\mathit{DB}'$ of $N$ entries and a cluster width $n$, produce $K=\lceil N/n\rceil$ centroids $\mathcal{C}$ and an assignment of every entry to exactly one centroid such that each cluster holds exactly $n$ entries (the last $Kn-N<n$ slots are padding).
```

---

## 2. §III-C1, server setup — replace the "map of the clusters" sentences

Old: *"However, there is a per-database storage requirement on the client side, as the client must download a map of the clusters and their indices. We assume the map is publicly available …"*

```latex
Because every cluster holds exactly $n$ entries, the server stores the embedding and document databases in cluster-major order: the $i$-th member of cluster $c$ sits at position $c\cdot n+i$. The client therefore needs no cluster-to-index map; the single public integer $n$ suffices to address any record, and the per-query PIR fetch size $p\cdot n$ is identical for every query, so it reveals nothing about how densely populated the selected region of the embedding space is.
```

---

## 3. Fig. 2 (protocol box), Server Pre-Processing steps 2–3 — replace

```latex
2) Then, to generate the cluster database $\mathit{DB}_c$ the server runs \textsf{IVF-Flat.Setup} with $\mathit{DB}'$ and cluster width $n$, obtaining $K=\lceil N/n\rceil$ equal-size clusters.
3) The server stores the squared norms of the centroids in $\mathit{DB}_c$ and lays out $\mathit{DB}'$ and $\mathit{DB}$ in cluster-major order (entry $i$ of cluster $c$ at position $c\cdot n+i$, padding slots blank). Only $n$ is sent to $\mathcal{C}$.
```

In Stage One step 3, replace *"using the stored map of cluster to database indices, produces the set of indices"* with:

```latex
computes the positions $\{c\cdot n+i : c\in C',\, 0\le i<n\}$ of the members of those clusters, discarding padding slots after retrieval.
```

---

## 4. §III-E "IVF-Flat Considerations" — replace the whole subsection

```latex
\subsection{Equal-Size IVF Clustering}
\label{sec:equal-size}
To securely compute the documents most relevant to the query we use IVF-Flat, since na\"ive FHE-kNN supports only $N<15$~\cite{...} while our databases reach $1.1$M entries. Standard IVF-Flat clusters with $k$-means and assigns each entry to its nearest centroid, which leaves cluster sizes highly unbalanced (1 to 102 entries at $K=4096$ on our 65k database). In a private pipeline this is undesirable twice over: the number of records a query fetches through PIR depends on where the query lands, leaking the density of its neighborhood, and the client needs the full cluster map to address records.

We therefore fix the cluster \emph{width} $n$ instead of the cluster count: $K=\lceil N/n\rceil$ centroids are trained with $k$-means, and entries are then assigned under the constraint that every cluster receives exactly $n$ of them, as close to nearest-centroid as possible. We solve this capacity-constrained assignment by price (dual) ascent on the transportation problem: each cluster carries a price that rises while it is over-subscribed, and every entry repeatedly re-chooses the cluster minimizing distance plus price, restricted to its 32 nearest centroids for tractability; any entry left unplaced is assigned exactly to the nearest cluster with free capacity. A Lagrangian lower bound certifies the result within 2.1\% of the optimal equal-size assignment, and the mean distance to the assigned centroid is only 3.6--6.7\% above that of unconstrained nearest-centroid assignment across all database sizes. The whole step takes seconds at 65k and minutes at 1M.

The server compares the encrypted query against all $K$ centroids (Section~\ref{sec:...}), the client decrypts the $K$ distances and keeps the top-$p$ clusters, and PIR then fetches exactly $p\cdot n$ records. Table~\ref{tab:operating_points} lists the configuration at each database size and Figure~\ref{fig:probe_selection} shows how $p$ was chosen.
```

---

## 5. Replace Fig. 3 with the new figure, and add the operating-point table

Delete the old Fig. 3 (*"Accuracy and Latency vs. Number of Probes …"*) and insert:

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/probe_selection.pdf}
  \caption{Choice of the number of probed clusters $p$ at each database size. (a) Held-out recall@10 of equal-size clustering as $p$ grows; dotted lines are the recall of an unbalanced $k$-means IVF baseline at the same scale ($K=N/16$, $p=K/41$, i.e.\ $K=4096$, $p=100$ at 65k). Stars mark the smallest $p$ that matches the baseline. (b) Hit@1 on NQ, FEVER and HotpotQA queries (plaintext top-1 among the returned results) against the fraction of the database fetched through PIR; stars mark the deployed $p$. At 1M, $p=192$ gives the same Hit@1 as the baseline-matched $p=232$ with 17\% fewer records.}
  \label{fig:probe_selection}
\end{figure*}

\begin{table}[t]
\caption{Equal-size clustering configuration at each database size: cluster width $n$, clusters $K=\lceil N/n\rceil$, probed clusters $p$, records fetched per query $p\cdot n$, and plaintext Hit@1 on NQ/FEVER/HotpotQA queries.}
\label{tab:operating_points}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Database} & $N$ & $n$ & $K$ & $p$ & $p\cdot n$ & \textbf{Hit@1} \\
\midrule
1k  & 1{,}000     & 64  & 16      & 10  & 640      & 94.7\% \\
65k & 65{,}000    & 128 & 508     & 43  & 5{,}504   & 91.0\% \\
1M  & 702{,}873   & 256 & 2{,}746 & 192 & 49{,}152  & 94.3\% \\
10M & 1{,}120{,}486 & 256 & 4{,}377 & 546 & 139{,}776 & 99.0\% \\
\bottomrule
\end{tabular}
\end{table}
```

Text to go with it (right after the table reference, or at the end of §III-E):

```latex
We choose $p$ as the smallest number of probes whose held-out recall@10 matches an unbalanced IVF-Flat baseline of the same scale (Figure~\ref{fig:probe_selection}a), so balancing never costs accuracy relative to standard IVF. The cluster width grows with the database ($n=128$ at 65k, $n=256$ at 1M and 10M): the candidate cost of balancing is roughly constant in $n$, while the centroid stage shrinks as $1/n$, so the widest width that still leaves enough clusters minimizes total cost. The 1k database is the controlled baseline-comparison set of Section~\ref{sec:...}; there we use $p=10$ of $16$ clusters, the smallest $p$ reaching about $95\%$ Hit@1.
```

---

## 6. §IV, remove/replace the 4096 sentence

Old: *"When using 4096 clusters for the 65,000-entry database, which would improve accuracy, the PIR-RAG total RAG processing time is approximately 1.5 hours."* — this is about PIR-RAG's own clustering, so it can stay. Anywhere else the text says **"4096 clusters"** or **"top-100 centroids"** for *our* system (abstract, intro, §III-E, Fig. 4 caption), change to the per-size configuration in Table `tab:operating_points` (65k: $K=508$, $p=43$).

The abstract/intro line *"sent to the server for k-means centroid scoring"* can stay; add "equal-size" where it describes the clustering:

```latex
... sent to the server for scoring against equal-size $k$-means centroids ...
```

---

## 7. Appendix B1 (IVF-Flat background) — append one paragraph

```latex
\emph{Equal-size variant.} Standard IVF-Flat assigns each vector to its nearest centroid, so Voronoi cells can hold very different numbers of vectors. Our variant keeps the $k$-means centroids but assigns vectors under an exact per-cluster capacity $n$, minimizing total squared distance to the assigned centroids. This is a transportation problem; we solve its Lagrangian dual by subgradient ascent on per-cluster prices, which lets every vector re-choose at the current prices each round (avoiding the first-come-first-served bias of greedy fill), and certify the result with the dual lower bound. Uniform cluster sizes make the PIR access pattern independent of the query and let records be addressed arithmetically as $c\cdot n+i$.
```

---

## Numbers used above (sources in the repo)

- Balance penalty 3.6–6.7%, duality gap ≤ 2.1%: `wiki-rag/cluster_size_sweep*/`, `cluster_size_sweep_small*/`.
- Held-out recall curves and baseline targets (panel a): `wiki-rag/cluster_size_sweep/cluster_size_sweep.json` (65k), `cluster_size_sweep_1M/`, `cluster_size_sweep_10M/`.
- Real-query Hit@1 curves (panel b, table): `tests/hit_rate/probe_curves.json` (plaintext, same 300 queries).
- The 10M Hit@1 in the table (99.0%) is plaintext at p=546. The secure 10M run scored lower (93/88/88); resolve that before quoting secure 10M numbers next to this table.

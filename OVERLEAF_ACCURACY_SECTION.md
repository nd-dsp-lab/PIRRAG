# Overleaf: probe-selection figure + rewritten accuracy section

Upload `figures/probe_selection.pdf`. Everything below is paste-ready.

---

## A. Probe-selection figure (replaces old Fig. 3 and the earlier two-panel draft)

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/probe_selection.pdf}
  \caption{Choosing the number of probed clusters $p$. For each database size, the blue curve is the held-out recall@10 of equal-size clustering as $p$ grows, and the dashed line is the recall of a standard unbalanced $k$-means IVF-Flat index at the same scale ($K=N/16$ clusters, top $K/41$ probed; $K=4096$, top-100 at 65k). We use the smallest $p$ at which equal-size clustering reaches the baseline (circle), so balancing never costs recall relative to standard IVF-Flat. At 1M we deploy $p=192$, which gives the same real-query Hit@1 as $p=232$ (94.3\% vs.\ 94.7\%) with 17\% fewer PIR records.}
  \label{fig:probe_selection}
\end{figure*}
```

Short text for §III-E (after the operating-point table):

```latex
Figure~\ref{fig:probe_selection} shows how $p$ is chosen. Equal-size clusters give up the natural adaptivity of $k$-means---dense regions no longer receive larger clusters---so matching a standard IVF-Flat index requires probing more clusters, but each probe now fetches a fixed $n$ records and the centroid stage shrinks by $n/16$. We take the smallest $p$ whose held-out recall@10 reaches that of the unbalanced index at the same scale: $p=43$ of $508$ at 65k, $p=232$ of $2{,}746$ at 1M (deployed at $p=192$, see caption) and $p=546$ of $4{,}377$ at 10M. The 1k comparison set is not an IVF operating point but a controlled diagnostic (Section~\ref{sec:accuracy}); there we use $p=10$ of $16$ clusters.
```

---

## B. Accuracy section (replaces the text around old Tables II and III)

Old Table II (100k, 4096 clusters, centroid overlap 75.6) describes the old clustering. Drop it, or keep it only
if you re-measure with the equal-size configuration.

```latex
\subsection{Retrieval Accuracy}
\label{sec:accuracy}
We measure how faithfully each private system reproduces plaintext retrieval. For every query we compute the exact (brute-force) top-10 over the full database with the same embedding model, and compare each system's returned top-10 against it. Throughout we report \emph{Top-10 Accuracy} (Acc.), the fraction of the plaintext top-10 that the system returns; this is the strict fidelity measure and is identical in both tables. We use 100 queries each from NQ, FEVER and HotpotQA.

\paragraph{Controlled comparison (1k).} All baselines are first compared on a 1{,}000-document database built from the 100k corpus so that it contains the plaintext top-3 of every test query; the small size lets every reproduced protocol run to completion on the same candidate set. Because each query's most relevant document is guaranteed to be present, we report \emph{Hit Rate} as whether that document appears in the returned top-10 (Table~\ref{tab:acc_1k}). \ours probes 10 of 16 clusters.

\paragraph{At scale (1M and 10M).} We then evaluate on the full 1M (703k passages) and 10M (1.12M passages) Wikipedia indices, against Graph-PIR and Tiptoe (Table~\ref{tab:acc_scale}). PIR-RAG is omitted here: its released implementation does not scale to these sizes without changes to its parameters and pipeline beyond the original design, and we prefer to omit it rather than report a modified system. At this scale no single document is guaranteed to be \emph{the} relevant one, and the plaintext top-10 is roughly twice as tightly packed as at 1k (median distance spread between the 1st and 10th neighbour 0.066 vs.\ 0.125), so its members are close to interchangeable. We therefore report \emph{Any-Hit@10}---whether at least one plaintext top-10 document is returned---which asks whether a system retrieves from the correct neighbourhood, alongside the strict Hit@10 of the top-1 document and Top-10 Accuracy.
```

### Table 1 — controlled 1k comparison

```latex
\begin{table}[t]
\caption{Controlled comparison on a 1{,}000-document database containing each query's plaintext top-3. Hit: the plaintext top-1 document is in the returned top-10. Acc.: overlap with the plaintext top-10. All values in \%.}
\label{tab:acc_1k}
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccc}
\toprule
 & \multicolumn{2}{c}{\textbf{NQ}} & \multicolumn{2}{c}{\textbf{FEVER}} & \multicolumn{2}{c}{\textbf{HotpotQA}} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
\textbf{Method} & \textbf{Hit} & \textbf{Acc.} & \textbf{Hit} & \textbf{Acc.} & \textbf{Hit} & \textbf{Acc.} \\
\midrule
\textbf{\ours (Ours)} & \textbf{94} & \textbf{93.9} & \textbf{94} & \textbf{95.2} & \textbf{96} & \textbf{93.7} \\
Graph-PIR \cite{wang2025pirragprivateinformationretrieval} & 83 & 79.4 & 85 & 79.5 & 84 & 75.4 \\
Tiptoe \cite{10.1145/3600006.3613134} & 69 & 51.5 & 71 & 42.4 & 72 & 51.0 \\
PIR-RAG \cite{wang2025pirragprivateinformationretrieval} & 32 & 22.6 & 48 & 24.1 & 46 & 27.0 \\
\bottomrule
\end{tabular}
\end{table}
```

### Table 2 — 1M and 10M (no PIR-RAG)

```latex
\begin{table*}[t]
\caption{Retrieval accuracy on the full 1M (703k passages) and 10M (1.12M passages) indices. Any: at least one plaintext top-10 document is returned. Hit: the plaintext top-1 document is in the returned top-10. Acc.: overlap with the plaintext top-10. All values in \%. \ours probes 192 of 2{,}746 (1M) and 546 of 4{,}377 (10M) equal-size clusters; Graph-PIR uses an enlarged search budget; Tiptoe uses $\sqrt{N}$ clusters as in~\cite{10.1145/3600006.3613134}.}
\label{tab:acc_scale}
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccccccccc}
\toprule
 & & \multicolumn{3}{c}{\textbf{NQ}} & \multicolumn{3}{c}{\textbf{FEVER}} & \multicolumn{3}{c}{\textbf{HotpotQA}} \\
\cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}
\textbf{DB} & \textbf{Method} & \textbf{Any} & \textbf{Hit} & \textbf{Acc.} & \textbf{Any} & \textbf{Hit} & \textbf{Acc.} & \textbf{Any} & \textbf{Hit} & \textbf{Acc.} \\
\midrule
\multirow{3}{*}{1M}  & \textbf{\ours (Ours)} & \textbf{100} & \textbf{98} & \textbf{96.4} & \textbf{100} & \textbf{94} & \textbf{87.8} & \textbf{100} & \textbf{92} & \textbf{89.7} \\
 & Graph-PIR & 94 & 70 & 63.7 & 83 & 59 & 54.0 & 90 & 48 & 49.4 \\
 & Tiptoe    & 45 & 17 & 9.0  & 54 & 23 & 12.0 & 37 & 14 & 6.2 \\
\midrule
\multirow{3}{*}{10M} & \textbf{\ours (Ours)} & \textbf{100} & \textbf{93} & \textbf{89.3} & \textbf{100} & \textbf{89} & \textbf{81.1} & \textbf{100} & \textbf{88} & \textbf{84.4} \\
 & Graph-PIR & 89 & 68 & 61.3 & 83 & 48 & 48.4 & 88 & 51 & 47.3 \\
 & Tiptoe    & 46 & 16 & 8.5  & 50 & 23 & 13.0 & 40 & 13 & 6.3 \\
\bottomrule
\end{tabular}
\end{table*}
```

Needs `\usepackage{multirow}`. If you want only Any-Hit and Acc. as originally planned, delete the three
`Hit` columns — but keeping them costs nothing (we lead on all three) and pre-empts the obvious reviewer
question of why the metric changed between tables.

### Discussion paragraph (after both tables)

```latex
On the controlled 1k set all baselines usually recover a relevant document, but their agreement with plaintext retrieval falls to 23--80\%, against 94--95\% for \ours. At scale the gap widens: Tiptoe searches a single one of $\sqrt{N}$ clusters, which contains the query's nearest neighbour for only 29--46\% of queries, and returns under a fifth of the plaintext top-10; Graph-PIR, even with an enlarged search budget, keeps roughly half of it. \ours returns a plaintext top-10 document for every query and preserves 81--96\% of the neighbourhood at 1M and 10M, because the FHE stage ranks every centroid exactly and the client reranks all $p\cdot n$ retrieved embeddings in plaintext.
```

---

## Numbers and caveats

- 1k: secure runs on `db_1k_gt`, p=10. Baselines rerun on the same database (Graph-PIR `results_graphpir`, Tiptoe 2026-09-28 run, PIR-RAG `better_tiptoe1k`).
  PIR-RAG's 1k numbers count ~half of its returned ids as misses because they are corrupted (≥2^24); a footnote is advisable.
- 1M: secure run at p=192 (99 NQ / 100 FEVER / 98 HotpotQA queries).
- 10M: secure run at p=546 (NQ on 95 verified queries). **Plaintext at the same p gives Hit 99/99/99 and Acc 98.0/93.4/96.5**, so the secure 10M row is probably understated (likely a different `--top-k` or cluster file in that run); confirm with your collaborator before submission.
- Tiptoe: ranking simulated in plaintext (Pyfhel unavailable) — mention in the caption or a footnote.
- Graph-PIR at 1M/10M: `results_graphpir_m128`; get the exact parameters for the caption.
- Tie/spread statistic (0.066 vs 0.125): measured on the same 300 queries, `tests/hit_rate/`.

# Retrieval fidelity of RAG-PIANO with equal-size clusters (paper metrics)

**Metrics (as in Table III of the draft).** For each query the reference is the
exact, plaintext top-10 over the full database (brute force, same BGE-base
embedding and query prompt the index was built with).
*Hit Rate* = the most-relevant document (plaintext top-1) appears in the returned
top-10. *Top-10 Accuracy* = overlap between the returned top-10 and the plaintext
top-10. 100 queries per dataset (FEVER claims, HotpotQA questions, NQ questions),
the same query set for every system.

## Results (Hit Rate % / Top-10 Accuracy %)

| system | database | probes p | NQ | FEVER | HotpotQA |
|---|---|---|---|---|---|
| **RAG-PIANO (ours)** | 65k, 508 clusters × 128 | 43 | **93 / 90.6** | **92 / 86.3** | **88 / 87.7** |
| RAG-PIANO (ours) | 65k | 100 | 97 / 95.4 | 97 / 93.9 | 95 / 94.7 |
| RAG-PIANO (ours) | 65k | 10 | 77 / 74.2 | 74 / 66.6 | 71 / 70.3 |
| **RAG-PIANO (ours)** | 1M (703k docs), 2,746 clusters × 256 | 192 | **98 / 96.4** | **94 / 87.8** | **92 / 89.7** |
| **RAG-PIANO (ours)** | 10M (1.12M docs), 4,377 clusters × 256 | 546 | **93 / 89.3** ‡‡ | **89 / 81.1** | **88 / 84.4** |
| Tiptoe | 1M (703k docs), 409 clusters | 1 | 1 / 0.4 | 2 / 1.5 | 3 / 0.9 |
| Tiptoe | 10M (1.12M docs), 56,024 clusters, simulated ranking ‡ | 1 | 3 / 2.2 | 7 / 2.6 | 4 / 1.8 |

All NQ rows are from the corrected reruns (question vectors verified). ‡‡ 10M NQ:
95 of 100 queries from the 10M rerun (question vectors verified 95/95 on 10M).
The 5 remaining queries (ids 2, 4, 25, 54, 62) were supplied separately, but their
results never contain a document beyond row 702,873 even though the 10M exact
top-10 has 2–7 such documents for each of them. They were therefore run on the 1M
index (the first 702,873 rows of the 10M index are identical) and are excluded.
Counting them as-is would give 91 / 87.4. Hit@1 equals Hit@10 for RAG-PIANO on
every dataset (the final exact rerank ranks the true nearest document first
whenever it was fetched), so "Hit Rate" here is also the top-1 hit rate.

Validation of the reference: with the repo's embedding
(`BAAI/bge-base-en`, prompt `"Represent this query for retrieval: …"`) the squared
distances reported by the secure pipeline are reproduced to 1e-6 for all 199
FEVER/HotpotQA queries, confirming the embedding, the prompt and the row→document
mapping.


1M (p=192, secure run): 99 NQ / 100 FEVER / 98 HotpotQA queries (NQ 87 and HotpotQA 46, 59
missing); query vectors verified for all 297, NQ on question text. Per query: 10.8 s FHE +
49.4 s PIR = 60.6 s, 67.7 MB up / 151.0 MB down.

## The NQ bug (fixed)

The NQ query vectors used in the original RAG-PIANO runs were produced from the **long
answer passage**, not from `question_text`: the reported distances match
`"Represent this query for retrieval: {answer}"` for 100/100 queries and the
question for 0/100. FEVER and HotpotQA used the correct text. Because the answer
passage is itself text from the corpus, retrieving it is easier than retrieving
from a question, so the NQ column overstates NQ performance — it should be read
as "the secure pipeline reproduces plaintext retrieval for the vectors it was
given" (97 % / 92.7 % at 65k, 98 % / 94.3 % at 10M). Fix: regenerate
`queries/natural_questions/query_*.txt` from `question_text` with the same prompt
and rerun NQ only; nothing else changes. Done for 65k (p=10/43/100; new vectors
verified to match the question embedding 100/100): NQ drops from the inflated
97 / 92.7 to a genuine **93 / 90.6** at p=43 (77 / 74.2 at p=10, 97 / 95.4 at p=100),
in line with FEVER and HotpotQA. 10M NQ (p=546) rerun: 93 / 89.3 on 95 verified queries.

## Why Tiptoe scores near zero

The Tiptoe runs available are on the 1M index (702,873 documents). Ranking its
returned documents in the exact plaintext order shows the first document it
returns has a median exact rank of ≈ 9,000 (≈ 4,500 under its own, unprompted
query embedding); only 30 of 300 queries have any returned document inside the
exact top-100. This is not a boundary effect but a consequence of its design as
run:

1. **Ranking inside one cluster on 192-d PCA + 5-bit quantised vectors with
   integer BFV scores.** The decrypted scores take values like 30–163 over
   ~1,500 documents, so the "top-10" is chosen among hundreds of ties — the
   dominant effect.
2. **A single probe into 409 very uneven clusters** (sizes 1–8,618). Even a
   perfect in-cluster ranking misses the nearest document whenever it lives in
   another cluster.
3. **No query prompt** (minor: the corpus was embedded with a prompt; Tiptoe's
   query encoder omits it).
4. It returned 10 documents for only 9–30 of 100 queries (2.5–4.2 on average),
   so its Hit@10 is effectively Hit@(what it returned).

‡ **Tiptoe on the 10M index (same database as our 10M row).** This run reports
"Pyfhel not available, using simulated ranking", i.e. the in-cluster ranking was
computed in plaintext rather than under BFV — the most favourable case for
Tiptoe's accuracy — and it used 56,024 clusters (~20 documents each). Its returned
documents are much closer to the exact ranking than in the 1M run (first returned
document median exact rank 192; 126/300 queries have a returned document in the
exact top-100), but with one ~20-document cluster probed the exact nearest
document is rarely inside it: Hit@1 is 3–4 % (NQ 3, FEVER 4, HotpotQA 4), and it
returned only 1.4–1.75 documents per query (72–83 % of queries a single one).
So even with the cryptographic noise removed, single-cluster probing caps Tiptoe
at a few percent on a million-document corpus, against 88–98 % for RAG-PIANO at
p=546 on the same index.

The draft's Tiptoe numbers (87 / 54.4 on NQ, 82 / 57.1 on FEVER) were measured on
a 1,000-document subset, where one cluster is a large slice of the database. At
700k documents the same design collapses; the comparison is between systems at
scale, not a change in Tiptoe's implementation.


## Hit@1 head-to-head (fair comparison independent of how many documents are returned)

Hit@1 = the first document returned is the plaintext exact nearest document.
Tiptoe returned only 2.5–4.2 documents per query (a single one for 45–65 % of
queries), so its Hit@10 is really Hit@(what it returned); Hit@1 removes that
asymmetry. For RAG-PIANO Hit@1 = Hit@10 (exact rerank), so its column is the same
as the Hit Rate table.

| system | database | p | NQ | FEVER | HotpotQA |
|---|---|---|---|---|---|
| **RAG-PIANO** | 65k, 508 × 128 | 43 | **93** | **92** | **88** |
| RAG-PIANO | 65k | 100 | 97 | 97 | 95 |
| RAG-PIANO | 65k | 10 | 77 | 74 | 71 |
| **RAG-PIANO** | 1M, 2,746 × 256 | 192 | **98** | **93** | **92** |
| **RAG-PIANO** | 10M, 4,377 × 256 | 546 | **93** ‡‡ | **88** | **88** |
| Tiptoe | 1M, 409 clusters | 1 | 0 | 2 | 1 |
| Tiptoe | 10M, 56,024 clusters, simulated ranking ‡ | 1 | 3 | 4 | 4 |

(Tiptoe under its own unprompted query embedding: NQ 0, FEVER 3, HotpotQA 2 —
within a point.) † 10M NQ still on the old answer-passage vectors; rerun pending.


## 1k ground-truth database (`wiki-rag/db_1k_gt`, 16 clusters × 64)

Hit Rate % / Top-10 Accuracy % against the exact top-10 inside the 1k set, all 300 queries, secure runs:

| p | NQ | FEVER | HotpotQA | query s | PIR down |
|---|---|---|---|---|---|
| 2 | 80 / 71.1 | 72 / 70.3 | 72 / 71.4 | 0.19 | 0.40 MB |
| **10** | **94 / 93.9** | **94 / 95.2** | **96 / 93.7** | 0.20 | 1.98 MB |

p=10 is the reported operating point: p=2 loses 14–24 points of Hit Rate to save 0.01 s
and 1.6 MB per query. Hit@1 = Hit Rate for both.

Baselines on the same 1k database (rerun on real data, `accuracy_faiss.npy`), same
queries and reference. Hit@1 % / Hit Rate % / Top-10 Accuracy %:

| system | NQ | FEVER | HotpotQA |
|---|---|---|---|
| **RAG-PIANO, p=10** | **94 / 94 / 93.9** | **94 / 94 / 95.2** | **96 / 96 / 93.7** |
| GraphPIR (k=16, 4 steps) | 55 / 67 / 61.0 | 60 / 76 / 61.3 | 58 / 71 / 66.8 |
| Tiptoe (10 clusters, simulated ranking) | 34 / 68 / 47.2 | 37 / 69 / 38.0 | 37 / 72 / 50.9 |
| PIR-RAG (32 clusters, top-3) ¶ | 17 / 32 / 22.6 | 29 / 48 / 24.1 | 26 / 46 / 27.0 |

¶ About half of PIR-RAG's returned ids (487–537 of 1,000 per dataset) are ≥ 2^24
(e.g. 16,777,515) and cannot be rows of the 1k database; subtracting 2^24 does not
recover them. Its numbers count those as misses — an implementation fault in its
id/URL handling, worth reporting to the collaborator.

(Superseded note:) The earlier PIR-RAG / GraphPIR / Tiptoe runs supplied for this database cannot be scored: their
logs report "Data files not found, falling back to synthetic data ... Generating synthetic
test data: 1000 documents", so they searched random vectors (some returned ids exceed
16,777,000), and the Tiptoe run reports real encryption = False. They must be rerun on
`db_1k_gt` (`modified_faiss_1000.npy` = its vectors, row order = `docs_1k.jsonl`).


## Tiptoe "better params" runs (380 clusters on 1M, 405 on 10M; simulated ranking)

Hit@1 % (NQ / FEVER / HotpotQA): **1M 0 / 2 / 0**, **10M 0 / 1 / 1**. Hit@10 ≤ 4 %, Top-10 Acc ≤ 1.8 %.
2.4–4.5 documents returned per query.

These are still far below what the configuration can deliver. A plaintext replica of
exactly this pipeline on 1M (PCA 768→192, 5-bit quantisation, 380 k-means clusters, one
probe, in-cluster ranking) gives Hit@1 = **16–26 %** (nearest document inside the chosen
cluster for 36 % of queries), versus 0–2 % measured. The first document these runs return
has a median exact rank of ≈5,000–7,600; only 16–18 of 300 queries have any returned
document in the exact top-10. The remaining gap is an implementation fault in the in-cluster
ranking or in the score→document / URL mapping, not a property of Tiptoe. For the paper,
report Tiptoe as the plaintext replica (≈21–26 % Hit@1 on 1M, ≤44 % with the paper's 20 %
boundary duplication at full precision), not these runs.

## Notes for the caption

- RAG-PIANO 65k p=43 is the operating point matched to "top-100 of 4096" on
  held-out corpus vectors (recall@10 0.953 vs 0.952). On real questions p=43 →
  p=100 buys ~5 points of Hit Rate, because real queries lie farther from the
  corpus than held-out corpus vectors do; the matched p is slightly optimistic
  for real queries.
- Databases differ across rows (65k / 10M for ours, 1M for Tiptoe); the metric is
  relative to each row's own plaintext reference, so rows are comparable as
  fidelity-to-plaintext, not as absolute recall on one corpus.
- Per-query cost from the same runs: 65k p=43 → 3.3 s (1.7 s FHE + 1.6 s PIR),
  16.9 MB downloaded; 10M p=546 → 47.9 s (15.9 s FHE + 31.7 s PIR), 101 MB.

Sources: `plaintext_ref/metrics_65k_asrun.json`, `plaintext_ref/metrics_10M_asrun.json`,
`plaintext_ref/metrics_1M_tiptoe.json`; scripts in `plaintext_ref/`.

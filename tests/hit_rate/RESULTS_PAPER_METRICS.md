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
| **RAG-PIANO (ours)** | 65k, 508 clusters × 128 | 43 | 97 / 92.7 † | **92 / 86.3** | **88 / 87.7** |
| RAG-PIANO (ours) | 65k | 100 | 100 / 97.1 † | 97 / 93.9 | 95 / 94.7 |
| RAG-PIANO (ours) | 65k | 10 | 92 / 79.1 † | 74 / 66.6 | 71 / 70.3 |
| **RAG-PIANO (ours)** | 10M (1.12M docs), 4,377 clusters × 256 | 546 | 98 / 94.3 † | **89 / 81.1** | **88 / 84.4** |
| Tiptoe | 1M (703k docs), 409 clusters | 1 | 1 / 0.4 | 2 / 1.5 | 3 / 0.9 |

† NQ ran on the wrong query text (see below); the numbers are a valid fidelity
check of the pipeline but not an NQ result. Hit@1 equals Hit@10 for RAG-PIANO on
every dataset (the final exact rerank ranks the true nearest document first
whenever it was fetched), so "Hit Rate" here is also the top-1 hit rate.

Validation of the reference: with the repo's embedding
(`BAAI/bge-base-en`, prompt `"Represent this query for retrieval: …"`) the squared
distances reported by the secure pipeline are reproduced to 1e-6 for all 199
FEVER/HotpotQA queries, confirming the embedding, the prompt and the row→document
mapping.

## The NQ bug

The NQ query vectors used in every RAG-PIANO run were produced from the **long
answer passage**, not from `question_text`: the reported distances match
`"Represent this query for retrieval: {answer}"` for 100/100 queries and the
question for 0/100. FEVER and HotpotQA used the correct text. Because the answer
passage is itself text from the corpus, retrieving it is easier than retrieving
from a question, so the NQ column overstates NQ performance — it should be read
as "the secure pipeline reproduces plaintext retrieval for the vectors it was
given" (97 % / 92.7 % at 65k, 98 % / 94.3 % at 10M). Fix: regenerate
`queries/natural_questions/query_*.txt` from `question_text` with the same prompt
and rerun NQ only (65k at p=43, 10M at p=546); nothing else changes.

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

The draft's Tiptoe numbers (87 / 54.4 on NQ, 82 / 57.1 on FEVER) were measured on
a 1,000-document subset, where one cluster is a large slice of the database. At
700k documents the same design collapses; the comparison is between systems at
scale, not a change in Tiptoe's implementation.

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

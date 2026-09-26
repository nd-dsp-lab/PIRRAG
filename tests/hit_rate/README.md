# Hit-rate evaluation of retrieved documents (equal-size clustering)

Scores the final top-10 documents returned by the FHE + PIR pipeline against the
FEVER / HotpotQA / Natural Questions ground truth shipped in
`batched_test/*/{fever_pir_input,hotpot_pir_input,pir_input}.json` (100 queries
each). Same matching rules as `tests/*/evaluate_post_pir_retrieval.py`, applied to
document texts looked up in the HF docstore (`index.pkl`) by row index.

| dataset | rule ("hit") |
|---|---|
| FEVER | an evidence entity (page title, `_`→space) occurs in the document text; the 25 NOT-ENOUGH-INFO claims have no entities and are excluded |
| HotpotQA | the answer string occurs in the document text (the repo has no `supporting_facts`) |
| NQ (strict) | a sentence of the long `answer` passage occurs verbatim in the document text |
| NQ (title) | the answer page's title (the long answer starts with it) is the title of a retrieved document |

hit@K = fraction of scored queries with at least one hit among the first K
retrieved documents. Retrieval on a top-viewed-articles corpus is bounded by
whether the answer page exists at all ("coverage" below).

## Results — equal-size clustering (RAG-PIANO), 100 queries per dataset

| database | clusters | p | FEVER hit@1 | FEVER hit@10 | HotpotQA hit@1 | HotpotQA hit@10 | NQ-strict hit@10 | NQ-title hit@1 | NQ-title hit@10 |
|---|---|---|---|---|---|---|---|---|---|
| 65k | 508 × 128 | 10 | 0.56 | 0.67 | 0.24 | 0.51 | 0.02 | 0.14 | 0.15 |
| 65k | 508 × 128 | 43 | 0.56 | 0.68 | 0.27 | 0.54 | 0.03 | 0.14 | 0.15 |
| 65k | 508 × 128 | 100 | 0.59 | 0.70 | 0.28 | 0.55 | 0.03 | 0.15 | 0.17 |
| 10M (1.12M docs) | 4,377 × 256 | 546 | 0.61 | 0.73 | 0.39 | 0.65 | 0.17 | 0.40 | 0.50 |

Corpus coverage (ceiling for page-level hits): FEVER evidence page present for
19/75 claims (65k) and 22/75 (10M); NQ answer page present for 20/100 (65k) and
57/100 (10M). Against that ceiling, NQ-title hit@10 is 15/20 on 65k and 50/57 on
10M. FEVER's entity rule matches mentions, so it is not bounded by page presence.

Per-query cost from the same runs (means over completed queries):

| run | FHE centroid s | PIR1 s | query s | PIR up / down MB | refreshes/query |
|---|---|---|---|---|---|
| 65k p=10 | 1.65 | 0.39 | 2.04 | 0.66 / 3.94 | 0.45 |
| 65k p=43 | 1.67 | 1.63 | 3.31 | 2.82 / 16.92 | 1.95 |
| 65k p=100 | 1.56 | 3.69 | 5.27 | 6.56 / 39.33 | 4.53 |
| 10M p=546 | 15.86 | 31.72 | 47.86 | 36.19 / 100.67 | 2.81 |

## Tiptoe baseline (collaborator's Tiptoe runs on the 1M database, 702,873 docs)

Tiptoe clusters with its own MiniBatch k-means (409 clusters on PCA-192 + 5-bit
quantised embeddings), ranks one cluster under BFV and PIR-retrieves URLs. It
returned on average only 2.5–4.2 documents per query (10 for 9–30 of 100 queries).

| dataset | hit@1 | hit@10 | MRR |
|---|---|---|---|
| FEVER | 0.08 | 0.12 | 0.098 |
| HotpotQA | 0.06 | 0.09 | 0.070 |
| NQ (strict) | 0.00 | 0.00 | 0.000 |

## Files

- `scripts/hit_rate.py <dir>` — needs `<dir>/{fever,hotpot,nq}_parsed.json` (query ids + retrieved rows + query text) and `<dir>/docs.json` (row → text/metadata). Writes `hit_rate_results.{json,md}`.
- `scripts/extract_docs.py index.pkl rows.json docs.json` — pulls texts for a set of rows out of a LangChain FAISS docstore pickle without LangChain installed.
- `scripts/parse_tiptoe_log.py log out.json` — parses a Tiptoe experiment log (rows come from its `https://example.com/doc_<row>` URLs).
- `results/<run>/` — parsed inputs and results for each run; `results/coverage*.json`.

Inputs: `topk_summary*.json` from the batched PIR runs (`{dataset, id, top_k:[{index, distance}]}`, rows are FAISS row ids of the named index). A sanity check that must pass before trusting a file: the 300 top-10 lists should all differ — an earlier 65k file had one identical list (identical distances) for every query, i.e. the same query vector was used throughout.

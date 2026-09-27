# 1k ground-truth database

1,000 documents taken from the 65k Wikipedia index (`wiki_index__top_100000__2025-04-11`):
the union of every test query's plaintext top-3 (882 docs), filled to 1,000 by
each query's next-ranked document in round-robin order. 300 test queries
(FEVER / HotpotQA / NQ, 100 each; BGE-base with the query prompt, NQ = question_text).
Every query's top-3 is in the set, and its 1k nearest document equals its 65k nearest document.

| file | what |
|---|---|
| `docs_1k.jsonl` | the database: `{row, row_65k, title, url, text}` per document, row = 0..999 |
| `records_1k_1024.txt` | same documents as fixed-width 1,024-char records, `chr(31)`-separated, row order (PIR record DB) |
| `vectors_1k.npy`, `index_1k.faiss` | 1,000 × 768 embeddings (float32), row order |
| `row_to_65k.npy` | 1k row → 65k FAISS row |
| `plaintext_top10_1k.json` | exact top-10 within the 1k set for every query, key `"<dataset>:<id>"` → 1k rows |
| `ivf_n64/` | equal-size clustering: 16 clusters × 64 (24 padding slots) |
| `pir_records_1k_n64.txt`, `pir_vectors_1k_n64.npy` (+ `.manifest.json`) | cluster-major PIR databases: row = `cluster_id * 64 + index`, padding rows blank/zero |
| `real_query_curve.json` | Hit@1 / Top-10 accuracy vs p on the 300 queries (p=10: 95% / 94%) |

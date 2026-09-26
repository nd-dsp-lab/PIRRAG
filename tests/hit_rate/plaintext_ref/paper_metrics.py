"""Paper metrics vs plaintext exact retrieval: Hit@K = plaintext top-1 in secure top-K; Acc@10 = |secure top-10 ∩ plaintext top-10| / 10."""
import json, sys, glob, os, numpy as np, faiss
S, dbname, faiss_index, out_json = sys.argv[1:5]; runs = sys.argv[5:]
keys = [tuple(k) for k in json.load(open(f"{S}/plaintext_ref/query_keys.json"))]
import os
Q = np.load(f"{S}/plaintext_ref/{os.environ.get('QFILE', 'queries_prompted.npy')}")
idx = faiss.read_index(faiss_index); N = idx.ntotal
db = idx.reconstruct_n(0, N); del idx
db_sq = (db.astype(np.float32) ** 2).sum(1)
with np.errstate(all="ignore"):
    D = db_sq[None, :] - 2.0 * (Q @ db.T) + (Q ** 2).sum(1)[:, None]
assert np.isfinite(D).all(), "non-finite distances"
part = np.argpartition(D, 9, axis=1)[:, :10]
plain = {k: [int(v) for v in part[i][np.argsort(D[i, part[i]])]] for i, k in enumerate(keys)}
print(f"{dbname}: {N:,} vectors; plaintext exact top-10 computed for {len(plain)} queries", flush=True)
name_map = {"hotpotqa": "hotpot", "natural_questions": "nq", "fever": "fever", "nq": "nq", "hotpot": "hotpot"}
results = {"database": dbname, "n_vectors": int(N), "plaintext_top10": {f"{k[0]}:{k[1]}": v for k, v in plain.items()}, "runs": {}}
for run in runs:
    label, path = run.rsplit("=", 1)
    sec = {}
    if path.endswith(".json"):
        for e in json.load(open(path)): sec[(name_map[e["dataset"]], e["id"])] = [x["index"] for x in e["top_k"]]
    else:  # parsed dir (tiptoe)
        for tag in ["fever", "hotpot", "nq"]:
            for r in json.load(open(f"{path}/{tag}_parsed.json"))["results"]: sec[(tag, r["query_id"])] = r["retrieved_rows"]
    per = {}
    for tag in ["fever", "hotpot", "nq"]:
        ks = [k for k in keys if k[0] == tag and k in sec]
        hit1 = np.mean([sec[k][:1] == plain[k][:1] for k in ks]); hit10 = np.mean([plain[k][0] in sec[k][:10] for k in ks])
        acc = np.mean([len(set(sec[k][:10]) & set(plain[k])) / 10 for k in ks]); nret = np.mean([len(sec[k]) for k in ks])
        per[tag] = {"n": len(ks), "hit@1": float(hit1), "hit@10": float(hit10), "acc@10": float(acc), "returned_mean": float(nret)}
    results["runs"][label] = per
    print(f"  {label:28s} " + "  ".join(f"{t}: n={per[t]['n']} hit@1={per[t]['hit@1']:.2f} hit@10={per[t]['hit@10']:.2f} acc@10={per[t]['acc@10']:.3f}" for t in per), flush=True)
json.dump(results, open(out_json, "w"), indent=1)

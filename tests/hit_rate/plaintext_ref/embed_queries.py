import json, sys, numpy as np, time
from sentence_transformers import SentenceTransformer
R, OUT = sys.argv[1:3]
inputs = {"fever": f"{R}/batched_test/fever_output/fever_pir_input.json",
          "hotpot": f"{R}/batched_test/hotpot_output/hotpot_pir_input.json",
          "nq": f"{R}/batched_test/nq_output/pir_input.json"}
texts, keys = [], []
for tag, p in inputs.items():
    for q in json.load(open(p)):
        keys.append((tag, q["query_id"])); texts.append(q.get("claim") or q.get("question") or q.get("question_text"))
t0 = time.time(); model = SentenceTransformer("BAAI/bge-base-en", device="cpu"); print(f"model loaded in {time.time()-t0:.0f}s", flush=True)
variants = {"prompted": [f"Represent this query for retrieval: {t}" for t in texts], "raw": texts}
for name, tx in variants.items():
    e = model.encode(tx, batch_size=32, normalize_embeddings=False, convert_to_numpy=True).astype(np.float32)
    np.save(f"{OUT}/queries_{name}.npy", e); print(f"{name}: {e.shape}, mean norm {np.linalg.norm(e, axis=1).mean():.3f}", flush=True)
json.dump(keys, open(f"{OUT}/query_keys.json", "w"))

#!/usr/bin/env python3
"""
Pull page_content + metadata for a set of FAISS row indices out of a LangChain
FAISS docstore pickle (InMemoryDocstore, index_to_docstore_id) without LangChain
installed: unknown classes are replaced by stubs that just keep their state.

    python extract_docs.py index.pkl rows.json out.json
"""
import json, pickle, sys

class _Stub:
    def __init__(self, *a, **k): self.__dict__["_args"] = a; self.__dict__.update(k)
    def __setstate__(self, state):
        if isinstance(state, dict): self.__dict__.update(state)
        elif isinstance(state, tuple):
            for s in state:
                if isinstance(s, dict): self.__dict__.update(s)
        else: self.__dict__["_state"] = state
    def __reduce__(self): return (_Stub, ())

class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(("langchain", "pydantic")):
            return _Stub
        return super().find_class(module, name)

def doc_fields(doc):
    d = getattr(doc, "__dict__", {})
    # pydantic v1/v2 pickles nest fields under __dict__ / __fields_set__ or directly
    for key in ("__dict__", "_state"):
        if key in d and isinstance(d[key], dict): d = {**d, **d[key]}
    text = d.get("page_content", "")
    meta = d.get("metadata", {}) or {}
    return text, meta

def main():
    pkl, rows_path, out_path = sys.argv[1:4]
    rows = sorted(set(int(r) for r in json.load(open(rows_path))))
    with open(pkl, "rb") as f:
        obj = _Unpickler(f).load()
    docstore, index_to_id = obj[0], obj[1]
    store = getattr(docstore, "_dict", None) or docstore.__dict__.get("_dict")
    print(f"docstore entries: {len(store):,}; index_to_docstore_id: {len(index_to_id):,}; requested rows: {len(rows)}")
    out = {}
    for r in rows:
        did = index_to_id.get(r) if isinstance(index_to_id, dict) else index_to_id[r]
        text, meta = doc_fields(store[did])
        out[str(r)] = {"text": text, "metadata": {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in meta.items()}}
    json.dump(out, open(out_path, "w"))
    r0 = rows[0]; print(f"sample row {r0}: meta={out[str(r0)]['metadata']} text={out[str(r0)]['text'][:200]!r}")

if __name__ == "__main__":
    main()

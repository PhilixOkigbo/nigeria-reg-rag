"""Retrieval stack: dense, BM25, and hybrid RRF over the regulatory corpus."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def searchable_text(r):
    """Prepend document metadata so titles and reference numbers are findable."""
    bits = [str(r.get("regulator") or ""), str(r.get("ref_no") or ""),
            str(r.get("title") or ""), str(r.get("document_date") or "")]
    header = " | ".join(b for b in bits if b and b.lower() != "nan")
    return f"{header}\n{r['text']}"


def tokenize(text):
    """Keep slash-separated references whole; split only ordinary words."""
    out = []
    for t in re.findall(r"[a-z0-9][a-z0-9/\.\-]*", text.lower()):
        t = t.strip(".-")
        if not t:
            continue
        if "/" in t:
            out.append(t)
            out.append(t.replace("/", ""))
        else:
            out.append(t)
            if "-" in t or "." in t:
                out.extend(p for p in re.split(r"[\.\-]", t) if p)
    return out


chunks_df = pd.read_json(ROOT / "data" / "chunks.jsonl", lines=True)
chunks_df["search_text"] = chunks_df.apply(searchable_text, axis=1)

embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
collection = client.get_collection("nigeria_reg")

_records = chunks_df.to_dict("records")
_bm25 = BM25Okapi([tokenize(r["search_text"]) for r in _records])


def search_dense(question, k=5):
    qv = embedder.encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=[qv], n_results=k)
    return [{**m, "text": d, "score": round(1 - dist, 3)}
            for d, m, dist in zip(res["documents"][0],
                                  res["metadatas"][0],
                                  res["distances"][0])]


def search_bm25(question, k=5):
    scores = _bm25.get_scores(tokenize(question))
    top = np.argsort(scores)[::-1][:k]
    return [{**_records[i], "score": round(float(scores[i]), 3)} for i in top]


def search_hybrid(question, k=5, pool=20, c=60):
    """Reciprocal Rank Fusion of dense and BM25. This is the chosen retriever."""
    fused = {}
    for lst in (search_dense(question, k=pool), search_bm25(question, k=pool)):
        for rank, h in enumerate(lst, 1):
            entry = fused.setdefault(h["chunk_id"], {"hit": h, "score": 0.0})
            entry["score"] += 1.0 / (c + rank)
    best = sorted(fused.values(), key=lambda e: -e["score"])[:k]
    return [{**e["hit"], "score": round(e["score"], 5)} for e in best]
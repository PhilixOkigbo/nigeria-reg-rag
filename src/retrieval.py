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


# --- Regulator routing ---------------------------------------------------
# The corpus is 61% SEC text (2,336 chunks) against 1,173 from the CBN, so a
# question about banks competes with a much larger body of capital-market
# rules. Phase 6 saw bank questions answered out of SEC instruments. These
# hints nudge ranking towards the regulator a question is about, without ever
# excluding a document outright - a wrong guess should cost rank, not recall.

BANKING_TERMS = (
    "bank", "banks", "banking", "deposit", "depositor", "depositors",
    "dmb", "psb", "cbn", "central bank", "account", "dormant", "customer",
    "bvn", "teller", "branch", "liquidity", "capital adequacy", "ndic",
    "insured institution", "licence", "license",
)
MARKET_TERMS = (
    "capital market", "securities", "sec ", "broker", "dealer", "issuer",
    "digital asset", "virtual asset", "token", "custodian", "custody",
    "collective investment", "fund manager", "registrar", "cmo", "sro",
    "investor", "listing", "prospectus",
)
BANKING_REGULATORS = ("CBN", "NDIC", "FGN")
MARKET_REGULATORS = ("SEC",)


def infer_regulators(question):
    """Which regulators a question is probably about, or None when unclear.

    Returning None on a mixed or unmarked question is deliberate: no signal is
    better than a confident wrong one.
    """
    q = f" {question.lower()} "
    banking = any(t in q for t in BANKING_TERMS)
    market = any(t in q for t in MARKET_TERMS)
    if banking and not market:
        return BANKING_REGULATORS
    if market and not banking:
        return MARKET_REGULATORS
    return None


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


def search_hybrid(question, k=5, pool=20, c=60, route=True, boost=0.25):
    """Reciprocal Rank Fusion of dense and BM25. This is the chosen retriever.

    With `route`, chunks from the regulator the question appears to be about
    get their fused score lifted by `boost`. It reorders candidates already
    retrieved; nothing is filtered out, so a misread question can cost a
    document rank but never remove it from the pool.
    """
    fused = {}
    for lst in (search_dense(question, k=pool), search_bm25(question, k=pool)):
        for rank, h in enumerate(lst, 1):
            entry = fused.setdefault(h["chunk_id"], {"hit": h, "score": 0.0})
            entry["score"] += 1.0 / (c + rank)

    preferred = infer_regulators(question) if route else None
    if preferred:
        for e in fused.values():
            if e["hit"].get("regulator") in preferred:
                e["score"] *= 1.0 + boost

    best = sorted(fused.values(), key=lambda e: -e["score"])[:k]
    return [{**e["hit"], "score": round(e["score"], 5), "routed_to": preferred}
            for e in best]
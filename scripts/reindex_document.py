"""Re-embed one document's chunks in place, after its text has been corrected.

Chroma ids are positional (`c000000` upwards, assigned from the row order of
data/chunks.jsonl), so a document can be repaired without rebuilding the whole
index - provided chunks.jsonl keeps its original row order and chunk count.
Both are asserted below, because getting either wrong silently misaligns every
later chunk from its vector.

    python scripts/reindex_document.py "Complaints Management"
"""
import sys
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def searchable_text(r):
    bits = [str(r.get("regulator") or ""), str(r.get("ref_no") or ""),
            str(r.get("title") or ""), str(r.get("document_date") or "")]
    header = " | ".join(b for b in bits if b and b.lower() != "nan")
    return f"{header}\n{r['text']}"


def main(title_substring):
    chunks = pd.read_json(ROOT / "data" / "chunks.jsonl", lines=True)
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    collection = client.get_collection("nigeria_reg")

    if collection.count() != len(chunks):
        raise SystemExit(f"index holds {collection.count()} vectors but chunks.jsonl has "
                         f"{len(chunks)} rows - positional ids cannot be trusted; rebuild instead")

    rows = chunks[chunks["title"].str.contains(title_substring, case=False, na=False)]
    if rows.empty:
        raise SystemExit(f"no chunks match {title_substring!r}")

    positions = rows.index.tolist()
    ids = [f"c{i:06d}" for i in positions]
    print(f"{len(rows)} chunks: {rows['title'].iloc[0][:60]}")
    print(f"positions {positions[0]}-{positions[-1]} -> {ids[0]}..{ids[-1]}")

    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode([searchable_text(r) for r in rows.to_dict("records")],
                              batch_size=64, normalize_embeddings=True).tolist()

    meta_cols = ["chunk_id", "regulator", "ref_no", "title", "url",
                 "document_date", "page", "extraction", "source_file"]
    metadatas = rows[meta_cols].fillna("").to_dict("records")
    for m in metadatas:
        m["page"] = int(m["page"])

    collection.upsert(ids=ids, documents=rows["text"].tolist(),
                      embeddings=vectors, metadatas=metadatas)
    print("re-indexed:", collection.count(), "vectors total")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Complaints Management")

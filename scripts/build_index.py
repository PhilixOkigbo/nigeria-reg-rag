"""Build the Chroma vector index from data/chunks.jsonl.

chroma_db/ is not committed, so a fresh clone needs this before the app will
run. It mirrors the final index build in notebooks/02_ingest.ipynb: each chunk
is embedded with its metadata header (regulator, reference number, title, date)
so titles and circular numbers are findable, while the stored document is the
plain chunk text. Ids are positional - c000000 upwards in file order - which
scripts/reindex_document.py relies on to repair one document in place.

    python scripts/build_index.py
"""
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION = "nigeria_reg"
BATCH = 500


def searchable_text(r):
    bits = [str(r.get("regulator") or ""), str(r.get("ref_no") or ""),
            str(r.get("title") or ""), str(r.get("document_date") or "")]
    header = " | ".join(b for b in bits if b and b.lower() != "nan")
    return f"{header}\n{r['text']}"


def main():
    chunks = pd.read_json(ROOT / "data" / "chunks.jsonl", lines=True)
    print(f"{len(chunks)} chunks from {chunks['title'].nunique()} documents")

    embedder = SentenceTransformer(EMBED_MODEL)
    vectors = embedder.encode([searchable_text(r) for r in chunks.to_dict("records")],
                              batch_size=64, show_progress_bar=True,
                              normalize_embeddings=True).tolist()

    meta_cols = ["chunk_id", "regulator", "ref_no", "title", "url",
                 "document_date", "page", "extraction", "source_file"]
    metadatas = chunks[meta_cols].fillna("").to_dict("records")
    for m in metadatas:
        m["page"] = int(m["page"])
    ids = [f"c{i:06d}" for i in range(len(chunks))]
    texts = chunks["text"].tolist()

    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(ids), BATCH):
        collection.add(ids=ids[i:i + BATCH], documents=texts[i:i + BATCH],
                       embeddings=vectors[i:i + BATCH], metadatas=metadatas[i:i + BATCH])
    print("indexed:", collection.count())


if __name__ == "__main__":
    main()

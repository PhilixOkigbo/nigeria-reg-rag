"""Answer generation: prompt, context assembly, caching, and cited sources.

Retrieval lives in `retrieval.py`; this module turns retrieved chunks into a
grounded answer that refuses when the corpus does not cover the question.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import errors as genai_errors

from .retrieval import chunks_df, search_hybrid

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "answer_cache.json"
MODEL = "gemini-3.8-flash"
# Tried when MODEL is busy (503) or out of free-tier quota (429). The free tier
# does both often enough that a live demo needs somewhere to fall back to.
FALLBACK_MODEL = "gemini-3.5-flash-lite"
REFUSAL = "i could not find this in the available documents"

load_dotenv(find_dotenv())
gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT = """You answer questions about Nigerian banking regulation and compliance.

Rules you must follow:
1. Answer ONLY from the passages below. Do not use outside knowledge.
2. If the passages do not contain the answer, say exactly:
   "I could not find this in the available documents."
   Do not guess, and do not fill gaps from general knowledge.
3. Quote the specific rule or figure where there is one.
4. Refer to sources by their number, like [2].
5. A passage marked OCR was machine-read from a scanned document. If your
   answer relies on one, end with this line exactly as written, with no source
   number after it: "Note: this passage was read from a scan and may contain
   transcription errors - please verify figures and paragraph references
   against the source document."

PASSAGES
--------
{context}

QUESTION
--------
{question}

ANSWER"""

# Chunk order within a document, so a hit can be widened to its neighbours.
_ROWS = chunks_df.to_dict("records")
_POS = {r["chunk_id"]: i for i, r in enumerate(_ROWS)}

_cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}


def _key(model, question):
    return hashlib.sha1(f"{model}||{question}".encode()).hexdigest()


def with_neighbours(hits, window=1):
    """Widen each hit to include the chunks either side, within the same document.

    Chunk boundaries fall mid-sentence in scanned statutes, which strands a rule
    from the clause that qualifies it. The neighbours go to the model as context
    but are not themselves indexed or ranked.
    """
    out = []
    for h in hits:
        i = _POS[h["chunk_id"]]
        lo, hi = max(0, i - window), min(len(_ROWS) - 1, i + window)
        parts = [_ROWS[j]["text"] for j in range(lo, hi + 1)
                 if _ROWS[j]["source_file"] == h["source_file"]]
        out.append({**h, "text": "\n".join(parts)})
    return out


def build_context(hits):
    """Number every passage sent to the model; citations refer to these numbers."""
    blocks = []
    for i, h in enumerate(hits, 1):
        tag = " [OCR]" if h.get("extraction") == "ocr" else ""
        ref = f" {h['ref_no']}" if h.get("ref_no") else ""
        blocks.append(
            f"[{i}] {h['regulator']}{ref} - {h['title']} (page {h['page']}){tag}\n{h['text']}"
        )
    return "\n\n".join(blocks)


def cited_indices(answer):
    """Passage numbers the answer refers to, e.g. '... [1, 4]' -> [1, 4]."""
    idx = set()
    for group in re.findall(r"\[([\d,\s]+)\]", answer):
        idx.update(int(n) for n in re.findall(r"\d+", group))
    return sorted(idx)


def _src(h, i):
    # `text` is the widened passage the model actually saw, neighbours included,
    # so a reader checking a citation sees the same words the answer came from.
    return {"n": i, "regulator": h["regulator"], "ref_no": h.get("ref_no", ""),
            "title": h["title"], "page": h["page"], "url": h["url"],
            "ocr": h.get("extraction") == "ocr",
            "chunk_id": h.get("chunk_id", ""), "text": h.get("text", "")}


def sources_for(text, hits, n_sources=3):
    """Report the passages the answer actually cited.

    Showing the top n retrieved instead left citations like [4] pointing at
    passages the reader could not see - the provenance chain has to resolve.
    """
    if REFUSAL in text.lower():
        return []
    cited = [i for i in cited_indices(text) if 1 <= i <= len(hits)]
    return [_src(hits[i - 1], i) for i in cited] or [
        _src(h, i) for i, h in enumerate(hits[:n_sources], 1)]


def _transient(exc):
    """Busy (503) or out of quota (429): worth trying another model. Anything
    else - a bad key, a malformed request - is a real error and should raise."""
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        return "quota"
    if "UNAVAILABLE" in msg or "503" in msg:
        return "busy"
    return None


def ask(question, k=5, n_sources=3, model=None, use_cache=True, window=1):
    """Retrieve, answer, and return the answer with its cited sources.

    With no `model` given, tries MODEL and falls back to FALLBACK_MODEL when the
    first is busy or out of quota. Passing `model` pins it - evaluation runs do
    that so their results come from one known model. Answers are cached on disk
    by (model, window, question); the result's `model` says which one answered.
    """
    candidates = [model] if model else [MODEL, FALLBACK_MODEL]
    hits = with_neighbours(search_hybrid(question, k=k), window=window)

    if use_cache:
        for m in candidates:
            ck = _key(f"{m}|w{window}", question)
            if ck in _cache:
                text = _cache[ck]
                return {"question": question, "answer": text, "cached": True, "model": m,
                        "sources": sources_for(text, hits, n_sources)}

    prompt = PROMPT.format(context=build_context(hits), question=question)
    reason = None
    for m in candidates:
        try:
            response = gemini.models.generate_content(model=m, contents=prompt)
        except (genai_errors.ClientError, genai_errors.ServerError) as exc:
            reason = _transient(exc)
            if reason:
                continue
            raise
        text = response.text.strip()
        _cache[_key(f"{m}|w{window}", question)] = text
        CACHE_PATH.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"question": question, "answer": text, "cached": False, "model": m,
                "sources": sources_for(text, hits, n_sources)}

    # Every candidate failed transiently. `quota_hit` keeps its old meaning for
    # callers - "not answered because of the API" - and `reason` says which.
    label = "QUOTA EXCEEDED" if reason == "quota" else "MODEL BUSY"
    return {"question": question, "answer": f"[{label} - question not answered]",
            "cached": False, "quota_hit": True, "reason": reason, "model": None,
            "sources": [_src(h, i) for i, h in enumerate(hits[:n_sources], 1)]}


def print_answer(result):
    """Print an answer with its sources, suppressing sources on a refusal."""
    print(result["answer"])
    if result.get("quota_hit"):
        print("\n(retrieval ran; these are the passages that would have been used)")
    elif REFUSAL in result["answer"].lower():
        print("\n(no sources - the question was not answered from the corpus)")
        return
    print("\nSources:")
    for s in result["sources"]:
        flag = " [OCR]" if s["ocr"] else ""
        ref = f" {s['ref_no']}" if s["ref_no"] else ""
        print(f"  [{s['n']}] {s['regulator']}{ref} - {s['title'][:60]}, p{s['page']}{flag}")
        print(f"      {s['url']}")

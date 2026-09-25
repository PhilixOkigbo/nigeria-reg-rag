"""Extraction-quality checks for the ingested corpus.

A PDF with an embedded text layer is not necessarily a readable one. The SEC
complaints framework was ingested as `extraction: "text"` because the file had
a text layer, but that layer used a non-standard font encoding and extracted as
a letter-for-letter substitution cipher - 16 chunks of noise that no retriever
could match. Nothing downstream noticed until a test question failed.

English stopword share separates the two cases cheaply: real regulatory prose
runs 0.20-0.46, the ciphered document scored 0.15, and the worst OCR scan 0.14.
Run this after any ingest and look at whatever sits at the bottom.
"""
import re

# Deliberately common words. Garbled text scores low because substitution and
# OCR noise rarely produce these exact short tokens at natural frequency.
STOPWORDS = frozenset("""the of and to in a for or by is are be shall that with
as not any on such person other this which from all may under it at an been his
her its""".split())

UNREADABLE = 0.05
DEGRADED = 0.18


def stopword_ratio(text):
    """Share of words that are common English. Higher is more readable."""
    words = re.findall(r"[A-Za-z]+", text.lower())
    return sum(w in STOPWORDS for w in words) / len(words) if words else 0.0


def readability_report(chunks_df):
    """One row per document, worst first, with a verdict on each."""
    df = chunks_df.copy()
    df["stopword_ratio"] = df["text"].map(stopword_ratio)
    report = (df.groupby(["regulator", "title", "extraction"])
                .agg(chunks=("chunk_id", "size"), stopword_ratio=("stopword_ratio", "mean"))
                .reset_index()
                .sort_values("stopword_ratio"))
    report["verdict"] = report["stopword_ratio"].map(
        lambda r: "unreadable" if r < UNREADABLE else ("degraded" if r < DEGRADED else "ok"))
    return report


def assert_readable(chunks_df):
    """Raise if any document looks unreadable. Use as an ingest gate."""
    bad = readability_report(chunks_df).query("verdict == 'unreadable'")
    if not bad.empty:
        raise ValueError("unreadable documents in corpus:\n" +
                         bad[["title", "extraction", "stopword_ratio"]].to_string(index=False))

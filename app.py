"""Streamlit front end for the Nigerian banking regulatory assistant.

Every claim on screen is traceable: each answer shows the passages it cited,
with the text the model actually saw, so a reader can check it without leaving
the page.
"""
import streamlit as st

st.set_page_config(page_title="Nigeria Regulatory Assistant", page_icon="§", layout="centered")


@st.cache_resource(show_spinner="Loading the index and embedding model...")
def load_rag():
    """Imported lazily and cached: this pulls in the model and opens Chroma."""
    from src.rag import ask, REFUSAL
    return ask, REFUSAL


ask, REFUSAL = load_rag()

st.title("Nigerian banking regulation")
st.caption("Answers come only from the indexed CBN, SEC, NDIC and federal documents. "
           "If the corpus does not cover a question, the assistant says so rather than guessing.")

question = st.text_input(
    "Ask a question",
    placeholder="What is the minimum capital for a commercial bank with international authorisation?",
)

if question:
    with st.spinner("Searching the regulations..."):
        result = ask(question)

    if result.get("quota_hit"):
        why = ("the daily free-tier quota is spent"
               if result.get("reason") == "quota"
               else "the language model is overloaded right now - try again in a minute")
        st.error(f"No answer: {why}. Both models were tried. "
                 "The passages below are what would have been used.")
    elif REFUSAL in result["answer"].lower():
        st.warning(result["answer"])
        st.caption("No sources shown: the question was not answered from the corpus.")
    else:
        st.markdown(result["answer"])
        st.caption(f"Answered by {result['model']}"
                   + (" · from cache" if result.get("cached") else ""))

    if result["sources"]:
        st.divider()
        st.subheader("Sources")
        for s in result["sources"]:
            ref = f" {s['ref_no']}" if s["ref_no"] else ""
            flag = "  ·  scanned document" if s["ocr"] else ""
            label = f"[{s['n']}]  {s['regulator']}{ref} — {s['title']}  ·  page {s['page']}{flag}"
            with st.expander(label):
                if s["ocr"]:
                    st.caption("Read by OCR from a scan; verify figures against the original.")
                st.text(s["text"])
                st.markdown(f"[Open the source document]({s['url']})")

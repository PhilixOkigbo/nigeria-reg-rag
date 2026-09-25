"""Streamlit front end for the Nigerian banking regulatory assistant.

Every claim on screen is traceable: each answer shows the passages it cited,
with the text the model actually saw, so a reader can check it without leaving
the page. The conversation stays on screen, each answer with its own sources.
"""
import streamlit as st

st.set_page_config(page_title="Nigeria Regulatory Assistant", page_icon="§", layout="centered")


@st.cache_resource(show_spinner="Loading the index and embedding model...")
def load_rag():
    """Imported lazily and cached: this pulls in the model and opens Chroma."""
    from src.rag import ask, REFUSAL
    return ask, REFUSAL


ask, REFUSAL = load_rag()

# One list per browser session. Streamlit keeps session_state separate for each
# tab, so two people using the app never see each other's conversation.
if "messages" not in st.session_state:
    st.session_state.messages = []


def render_result(result):
    """Draw one answer: the text (or why there isn't one), then its sources."""
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

    for s in result["sources"]:
        ref = f" {s['ref_no']}" if s["ref_no"] else ""
        flag = "  ·  scanned document" if s["ocr"] else ""
        label = f"[{s['n']}]  {s['regulator']}{ref} — {s['title']}  ·  page {s['page']}{flag}"
        with st.expander(label):
            if s["ocr"]:
                st.caption("Read by OCR from a scan; verify figures against the original.")
            st.text(s["text"])
            st.markdown(f"[Open the source document]({s['url']})")


with st.sidebar:
    st.subheader("About")
    st.caption("Answers come only from the indexed CBN, SEC, NDIC and federal documents. "
               "If the corpus does not cover a question, the assistant says so rather "
               "than guessing.")
    st.caption("Each question is answered on its own for now - follow-ups like "
               "\"what about merchant banks?\" need the full question.")
    if st.button("New conversation", use_container_width=True,
                 disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()

st.title("Nigerian banking regulation")

if not st.session_state.messages:
    st.caption("Try: What is the minimum capital for a commercial bank with "
               "international authorisation?")

# Replay the conversation so far. Streamlit reruns this script top to bottom on
# every interaction, so history has to be redrawn from session_state each time.
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_result(m["result"])

question = st.chat_input("Ask about Nigerian banking regulation")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the regulations..."):
            result = ask(question)
        render_result(result)

    st.session_state.messages.append({"role": "assistant", "result": result})

"""
Demo UI for the  PaperPilot AI
-----------------------------------------------
This is NOT part of the MCP protocol — MCP servers have no frontend of
their own. This is a small standalone Streamlit app that imports and
calls the exact same functions your MCP tools use, so you can visually
demo the project (for your resume/portfolio) without needing an MCP
client like Claude Desktop open.

Run with:  streamlit run demo_app.py
"""
import streamlit as st
import server  # reuses your real search/fetch/ask logic — no duplication

st.set_page_config(page_title="PaperPilot AI", layout="wide")

with st.sidebar:
    st.header(" PaperPilot AI")
    st.caption("An MCP server + RAG pipeline for querying academic papers.")

    st.subheader("How it works")
    st.markdown(
        "1. **Search** — query arXiv's API\n"
        "2. **Fetch** — download the PDF, extract text, split into chunks\n"
        "3. **Embed** — each chunk → a 384-dim vector (sentence-transformers, local, free)\n"
        "4. **Retrieve** — question → vector → cosine similarity → top matching chunks\n"
        "5. **Generate** — chunks + question → LLM (Groq/Llama) → grounded answer"
    )

    st.subheader("Also an MCP server")
    st.markdown(
        "This same logic is exposed as an **MCP server** — usable directly "
        "from Claude Desktop or any MCP client, with 4 tools, 2 resources, "
        "and 2 prompts. This Streamlit app is just a visual demo layer on top."
    )

    n_papers = len(server.list_all_papers())
    st.metric("Papers in library", n_papers)

    st.divider()
    st.caption("🔗 [View source on GitHub](#)")  # replace with your repo link

st.title("PaperPilot AI")
st.caption(
    "MCP-Powered Semantic Research Platform using RAG, Groq, and arXiv."
)

# Session state: remember what's been fetched. Seed it from the database
# on first load of each session, so papers fetched in a PREVIOUS session
# (or via the MCP server directly) still show up here — not just ones
# fetched in this exact browser tab.
if "fetched_papers" not in st.session_state:
    st.session_state.fetched_papers = dict(server.list_all_papers())  # {paper_id: title}
if "search_results" not in st.session_state:
    st.session_state.search_results = []

col_search, col_ask, col_compare = st.columns([1, 1, 1])

# --- Left column: search + fetch -------------------------------------------
with col_search:
    st.subheader("1. Search & fetch a paper")
    query = st.text_input("Search arXiv", placeholder="e.g. retrieval augmented generation")

    if st.button("Search", type="primary") and query:
        with st.spinner("Searching arXiv..."):
            try:
                st.session_state.search_results = server._search_arxiv_raw(query, max_results=5)
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.search_results = []

    for paper in st.session_state.search_results:
        with st.container(border=True):
            st.markdown(f"**{paper['title']}**")
            st.caption(f"{', '.join(paper['authors'])} · {paper['published']} · `{paper['id']}`")
            st.write(paper["summary"][:250] + "...")

            already_fetched = paper["id"] in st.session_state.fetched_papers
            if already_fetched:
                st.success("Fetched ✓")
            else:
                if st.button("Fetch full text", key=f"fetch_{paper['id']}"):
                    with st.spinner("Downloading + embedding paper (can take ~10-20s)..."):
                        result = server.fetch_paper(paper["id"])
                    if result.startswith("Fetched"):
                        st.session_state.fetched_papers[paper["id"]] = paper["title"]
                        st.success(result)
                        st.rerun()
                    else:
                        st.error(result)

# --- Right column: ask questions --------------------------------------------
with col_ask:
    st.subheader("2. Ask a fetched paper")

    if not st.session_state.fetched_papers:
        st.info("Fetch a paper on the left first.")
    else:
        selected_id = st.selectbox(
            "Which paper?",
            options=list(st.session_state.fetched_papers.keys()),
            format_func=lambda pid: f"{st.session_state.fetched_papers[pid]} ({pid})",
        )
        question = st.text_area("Your question", placeholder="What method does this paper propose?")

        if st.button("Ask", type="primary") and question:
            with st.spinner("Retrieving relevant excerpts and generating an answer..."):
                result = server._ask_paper_raw(selected_id, question)

            if result.get("error"):
                st.error(result["error"])
            else:
                st.markdown("**Answer:**")
                st.write(result["answer"])
                with st.expander("🔍 Show retrieved excerpts (this is the RAG part)"):
                    st.text(result["context"])

# --- Third column: compare two fetched papers -------------------------------
with col_compare:
    st.subheader("3. Compare two papers")

    if len(st.session_state.fetched_papers) < 2:
        st.info("Fetch at least 2 papers to compare them.")
    else:
        ids = list(st.session_state.fetched_papers.keys())
        paper_a = st.selectbox(
            "Paper A", options=ids,
            format_func=lambda pid: f"{st.session_state.fetched_papers[pid]} ({pid})",
            key="compare_a",
        )
        paper_b = st.selectbox(
            "Paper B", options=[pid for pid in ids if pid != paper_a],
            format_func=lambda pid: f"{st.session_state.fetched_papers[pid]} ({pid})",
            key="compare_b",
        )

        if st.button("Compare", type="primary"):
            with st.spinner("Retrieving context from both papers and comparing..."):
                result = server._compare_papers_raw(paper_a, paper_b)

            if result.get("error"):
                st.error(result["error"])
            else:
                st.markdown("**Comparison:**")
                st.write(result["comparison"])
                with st.expander("🔍 Show retrieved excerpts (this is the RAG part)"):
                    st.markdown(f"**Paper A ({paper_a}):**")
                    st.text(result["context_1"])
                    st.markdown(f"**Paper B ({paper_b}):**")
                    st.text(result["context_2"])

st.divider()
st.caption(
    "This is a demo wrapper for showcasing the project — the actual deliverable "
    "is the MCP server (server.py), usable from any MCP client such as Claude Desktop."
)
"""
Demo UI for the Research Assistant MCP Server
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

st.set_page_config(page_title="PaperPilot AI", page_icon="📄", layout="wide")

with st.sidebar:
    st.header("📄 PaperPilot AI")
    st.caption("MCP-powered semantic research platform using RAG, Groq, and arXiv.")

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

st.title("📄 PaperPilot AI")
st.caption("Search arXiv, fetch a paper, and ask questions grounded in its actual content.")

# Session state: remember what's been fetched. Seed it from the database
# on first load of each session, so papers fetched in a PREVIOUS session
# (or via the MCP server directly) still show up here — not just ones
# fetched in this exact browser tab.
if "fetched_papers" not in st.session_state:
    st.session_state.fetched_papers = dict(server.list_all_papers())  # {paper_id: title}
if "search_results" not in st.session_state:
    st.session_state.search_results = []

tab_search, tab_ask, tab_compare = st.tabs(["🔍  Search & Fetch", "💬  Ask a Paper", "⚖️  Compare Papers"])

# --- Tab 1: search + fetch, full width, grid layout -------------------------
with tab_search:
    query = st.text_input("Search arXiv", placeholder="e.g. retrieval augmented generation")
    search_clicked = st.button("Search", type="primary")

    if search_clicked and query:
        with st.spinner("Searching arXiv..."):
            try:
                st.session_state.search_results = server._search_arxiv_raw(query, max_results=6)
            except Exception as e:
                st.error(f"Search failed: {e}")
                st.session_state.search_results = []

    results = st.session_state.search_results
    if results:
        st.caption(f"{len(results)} results")
        # Grid: 3 cards per row instead of one long vertical stack, so all
        # results are visible with far less scrolling.
        cards_per_row = 3
        for row_start in range(0, len(results), cards_per_row):
            row_papers = results[row_start:row_start + cards_per_row]
            cols = st.columns(cards_per_row)
            for col, paper in zip(cols, row_papers):
                with col:
                    with st.container(border=True, height=320):
                        st.markdown(f"**{paper['title']}**")
                        st.caption(f"{', '.join(paper['authors'])} · {paper['published']}")
                        st.code(paper["id"], language=None)
                        with st.expander("Abstract"):
                            st.write(paper["summary"])

                        already_fetched = paper["id"] in st.session_state.fetched_papers
                        if already_fetched:
                            st.success("Fetched ✓")
                        else:
                            if st.button("Fetch full text", key=f"fetch_{paper['id']}", use_container_width=True):
                                with st.spinner("Downloading + embedding (~10-20s)..."):
                                    result = server.fetch_paper(paper["id"])
                                if result.startswith("Fetched"):
                                    st.session_state.fetched_papers[paper["id"]] = paper["title"]
                                    st.success(result)
                                    st.rerun()
                                else:
                                    st.error(result)

# --- Tab 2: ask questions, full width ---------------------------------------
with tab_ask:
    if not st.session_state.fetched_papers:
        st.info("Fetch a paper in the Search tab first.")
    else:
        col_left, col_right = st.columns([1, 1.4])
        with col_left:
            selected_id = st.selectbox(
                "Which paper?",
                options=list(st.session_state.fetched_papers.keys()),
                format_func=lambda pid: f"{st.session_state.fetched_papers[pid]} ({pid})",
            )
            question = st.text_area("Your question", placeholder="What method does this paper propose?", height=120)
            ask_clicked = st.button("Ask", type="primary")

        with col_right:
            if ask_clicked and question:
                with st.spinner("Retrieving relevant excerpts and generating an answer..."):
                    result = server._ask_paper_raw(selected_id, question)

                if result.get("error"):
                    st.error(result["error"])
                else:
                    st.markdown("**Answer**")
                    st.write(result["answer"])
                    with st.expander("🔍 Show retrieved excerpts (this is the RAG part)"):
                        st.text(result["context"])
            else:
                st.caption("Your answer will appear here.")

# --- Tab 3: compare two fetched papers ---------------------------------------
with tab_compare:
    if len(st.session_state.fetched_papers) < 2:
        st.info("Fetch at least 2 papers in the Search tab to compare them.")
    else:
        ids = list(st.session_state.fetched_papers.keys())
        col_a, col_b = st.columns(2)
        with col_a:
            paper_a = st.selectbox(
                "Paper A", options=ids,
                format_func=lambda pid: f"{st.session_state.fetched_papers[pid]} ({pid})",
                key="compare_a",
            )
        with col_b:
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
                st.markdown("**Comparison**")
                st.write(result["comparison"])
                with st.expander("🔍 Show retrieved excerpts (this is the RAG part)"):
                    ctx_a, ctx_b = st.columns(2)
                    with ctx_a:
                        st.markdown(f"**Paper A ({paper_a})**")
                        st.text(result["context_1"])
                    with ctx_b:
                        st.markdown(f"**Paper B ({paper_b})**")
                        st.text(result["context_2"])

st.divider()
st.caption(
    "This is a demo wrapper for showcasing the project — the actual deliverable "
    "is the MCP server (server.py), usable from any MCP client such as Claude Desktop."
)
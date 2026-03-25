import streamlit as st
from agents.supervisor import build_research_graph

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Multi-Agent Research Assistant")
st.caption("Powered by LangGraph + Groq (LLaMA 3) + ChromaDB — 100% Free")

topic = st.text_input(
    "Enter a research topic:",
    placeholder="e.g. Quantum computing advances 2024",
    key="research_topic"
)

if st.button("🚀 Start Research", type="primary") and topic:
    with st.spinner("Agents are working..."):
        col1, col2, col3 = st.columns(3)
        col1.info("🔍 Searching the web...")
        col2.info("📄 Scraping pages...")
        col3.info("✍️ Writing report...")

        graph = build_research_graph()
        result = graph.invoke({
            "topic": topic,
            "search_results": [],
            "scraped_content": [],
            "final_report": "",
            "errors": []
        })

        col1.success("✅ Search done")
        col2.success("✅ Scraping done")
        col3.success("✅ Report ready")

    if result.get("errors"):
        with st.expander("⚠️ Warnings during research"):
            for err in result["errors"]:
                st.warning(err)

    st.divider()
    st.subheader("📋 Research Report")
    st.markdown(result["final_report"])

    with st.expander("📦 Raw Search Findings"):
        for i, r in enumerate(result["search_results"]):
            st.write(f"**Result {i+1}:** {r}")

    with st.expander("🌐 Scraped Page Summaries"):
        for i, s in enumerate(result["scraped_content"]):
            st.write(f"**Source {i+1}:** {s}")
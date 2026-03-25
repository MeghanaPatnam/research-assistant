# agents/writer_agent.py
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from memory.vector_store import get_vector_store
from config import WRITER_MODEL
from logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

def write_final_report(topic: str, search_results: list, scraped_content: list) -> str:
    try:
        logger.info("Retrieving from vector store...")
        vectorstore = get_vector_store()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(topic)
        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

        all_content = "\n\n".join([
            "=== SEARCH FINDINGS ===",
            "\n".join(search_results),
            "=== SCRAPED CONTENT ===",
            "\n".join(scraped_content),
            "=== RETRIEVED FROM MEMORY ===",
            retrieved_text
        ])

        logger.info("Writing final report...")
        llm = ChatGroq(model=WRITER_MODEL, temperature=0.3)
        report = llm.invoke([
            HumanMessage(content=f"""
            You are an expert research analyst and writer.
            Write a comprehensive report on: "{topic}"

            Structure:
            1. Executive Summary
            2. Background & Context
            3. Key Findings
            4. Analysis & Insights
            5. Conclusions & Recommendations

            Research Findings:
            {all_content}
            """)
        ])
        logger.info("Report complete")
        return report.content

    except Exception as e:
        logger.error(f"Report writing failed: {e}")
        return f"Report generation failed: {str(e)}"
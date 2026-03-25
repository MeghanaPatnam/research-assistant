# agents/supervisor.py
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from typing import TypedDict, List
from agents.search_agent import create_search_agent
from agents.scraper_agent import scrape_and_summarize
from memory.vector_store import get_vector_store
from config import WRITER_MODEL, MAX_SEARCH_RESULTS
from logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

class ResearchState(TypedDict):
    topic: str
    search_results: List[str]
    scraped_content: List[str]
    final_report: str
    errors: List[str]

def search_node(state: ResearchState) -> ResearchState:
    try:
        search_fn = create_search_agent()
        result = search_fn(state['topic'])
        state["search_results"] = [result]
    except Exception as e:
        logger.error(f"Search node failed: {e}")
        state["errors"] = state.get("errors", []) + [str(e)]
        state["search_results"] = []
    return state

def scrape_node(state: ResearchState) -> ResearchState:
    try:
        from duckduckgo_search import DDGS
        scraped = []
        with DDGS() as ddgs:
            results = list(ddgs.text(
                state["topic"],
                max_results=MAX_SEARCH_RESULTS
            ))
            for r in results:
                summary = scrape_and_summarize(r["href"], state["topic"])
                scraped.append(summary)

        state["scraped_content"] = scraped

        vectorstore = get_vector_store()
        docs = [
            Document(page_content=s, metadata={"topic": state["topic"]})
            for s in scraped if s and s.strip()
        ]
        if docs:
            vectorstore.add_documents(docs)
            logger.info(f"Stored {len(docs)} docs in ChromaDB")

    except Exception as e:
        logger.error(f"Scrape node failed: {e}")
        state["errors"] = state.get("errors", []) + [str(e)]
        state["scraped_content"] = []
    return state

def write_report_node(state: ResearchState) -> ResearchState:
    try:
        llm = ChatGroq(model=WRITER_MODEL, temperature=0.3)
        all_content = "\n\n".join(
            state["search_results"] + state["scraped_content"]
        )
        report = llm.invoke([
            HumanMessage(content=f"""
            You are an expert research writer.
            Write a comprehensive report on: {state['topic']}

            Include:
            - Executive summary
            - Key findings (with sections)
            - Conclusions

            Research Findings:
            {all_content}
            """)
        ])
        state["final_report"] = report.content
    except Exception as e:
        logger.error(f"Write node failed: {e}")
        state["errors"] = state.get("errors", []) + [str(e)]
        state["final_report"] = "Report generation failed."
    return state

def build_research_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("scrape", scrape_node)
    graph.add_node("write",  write_report_node)

    graph.set_entry_point("search")
    graph.add_edge("search", "scrape")
    graph.add_edge("scrape", "write")
    graph.add_edge("write",  END)

    return graph.compile()
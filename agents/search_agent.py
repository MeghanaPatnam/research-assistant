# agents/search_agent.py
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import SEARCH_MODEL
from logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

def create_search_agent():
    llm = ChatGroq(model=SEARCH_MODEL, temperature=0)
    search_tool = DuckDuckGoSearchRun()

    def run_search(topic: str) -> str:
        try:
            logger.info(f"Searching for: {topic}")
            results = search_tool.run(topic)
            summary = llm.invoke([
                HumanMessage(content=f"""
                You are a research agent. Based on the search results below,
                extract and summarize the key facts, statistics, and insights
                about: {topic}

                Search Results:
                {results}

                Provide a clear, structured summary.
                """)
            ])
            logger.info("Search complete")
            return summary.content
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {str(e)}"

    return run_search
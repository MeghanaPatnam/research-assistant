# agents/scraper_agent.py
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import SEARCH_MODEL
from logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

def scrape_and_summarize(url: str, topic: str) -> str:
    try:
        logger.info(f"Scraping: {url}")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)[:4000]

        if not text.strip():
            logger.warning(f"Empty content from {url}")
            return ""

        llm = ChatGroq(model=SEARCH_MODEL, temperature=0)
        summary = llm.invoke([
            HumanMessage(content=f"""
            Summarize the following content related to: {topic}
            Focus only on relevant facts and insights.

            Content: {text}
            """)
        ])
        logger.info(f"Scraping done for: {url}")
        return summary.content

    except Exception as e:
        logger.error(f"Scraping failed for {url}: {e}")
        return ""
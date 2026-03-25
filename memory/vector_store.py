# memory/vector_store.py
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from config import CHROMA_DIR, EMBEDDING_MODEL, DEVICE
from logger import get_logger

logger = get_logger(__name__)

def get_vector_store():
    logger.info("Initializing vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True}
    )
    vectorstore = Chroma(
        collection_name="research_findings",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    logger.info("Vector store ready")
    return vectorstore
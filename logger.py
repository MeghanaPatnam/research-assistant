# logger.py
import logging
from config import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def get_logger(name: str):
    return logging.getLogger(name)
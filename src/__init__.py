import logging
import os

os.makedirs("../logs", exist_ok=True)

str_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    format=str_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="../logs/logging.log"),
    ],
)

logger = logging.getLogger(__name__)

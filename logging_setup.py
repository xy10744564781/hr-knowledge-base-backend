import logging
from config import LOG_FILE_NAME

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
    filename=LOG_FILE_NAME
)
logger = logging.getLogger("factory_kb")

import logging
import os

from microscopy_proc.constants import CACHE_DIR

LOG_FILE_FORMAT = "%Y-%m-$d_%H-%M-%S"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def init_logger(name: str = __name__) -> logging.Logger:
    """
    Setup logging configuration
    """
    # Making cache directory if it does not exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    # Initialising/getting logger and its configuration
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # If logger does not have handlers, add them
    if not logger.hasHandlers():
        # Formatter
        formatter = logging.Formatter(LOG_FORMAT)
        # File handler
        # NOTE: May not work for multiprocessing - multiple files will be created
        # curr_time = datetime.datetime.now().strftime(LOG_FILE_FORMAT)
        log_fp = os.path.join(CACHE_DIR, "debug.log")
        file_handler = logging.FileHandler(log_fp, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

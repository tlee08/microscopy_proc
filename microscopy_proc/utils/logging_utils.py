import logging
import os
import traceback

from microscopy_proc.constants import TEMP_DIR


def init_logger() -> logging.Logger:
    """
    Setup logging configuration
    """
    # For total logging
    total_log_fp = os.path.join(TEMP_DIR, "log.log")
    os.makedirs(TEMP_DIR, exist_ok=True)
    # Setting up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(total_log_fp),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging configuration set up")
    return logger


def log_func_decorator(logger: logging.Logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                logger.info(f"STARTED {func.__name__}")
                func(*args, **kwargs)
                logger.info(f"FINISHED {func.__name__}")
            except Exception as e:
                logger.error(f"ERROR {func.__name__}")
                logger.error(traceback.format_exc())
                raise e

        return wrapper

    return decorator

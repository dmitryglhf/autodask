import os
import logging
import logging.handlers
from datetime import datetime


def get_logger(
        logger_name: str,
        log_dir='adsk_logs',
        level=logging.INFO,
        log_format: str = '%(asctime)s - %(name)s - %(message)s',
        rotation_bytes: int = 5_242_880,  # 5MB
        backup_count: int = 3
) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(log_format)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    if logger.hasHandlers():
        logger.handlers.clear()

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"{log_dir}/autodask_launch_{current_time}.log"

    file_handler = logging.handlers.RotatingFileHandler(
        filename=filename,
        maxBytes=rotation_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

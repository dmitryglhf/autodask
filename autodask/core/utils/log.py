import logging
import os
from datetime import datetime


def get_logger(name: str):
    os.makedirs("adsk_logs", exist_ok=True)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    log_filename = f"logs/autodask_launch_{current_time}.log"

    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    return logger

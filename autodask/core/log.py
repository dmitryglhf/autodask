import logging


def get_logger(name: str):
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    formatter = logging.Formatter(
        "{asctime} - {levelname} - {message}",
        style = "{",
        datefmt = "%Y-%m-%d %H:%M",
    )
    console_handler.setFormatter(formatter)
    return logger
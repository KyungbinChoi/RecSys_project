import logging

def get_logger(name: str = "recsys"):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger
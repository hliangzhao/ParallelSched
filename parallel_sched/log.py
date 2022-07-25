"""
The logging system.
"""
import logging


def get_logger(name="logger", level="INFO", mode="w", file_handler=True, stream_handler=True, prefix=""):
    logger = logging.getLogger(name)

    fh = logging.FileHandler(name + ".log", mode)
    sh = logging.StreamHandler()

    if level == "INFO":
        logger.setLevel(logging.INFO)
        fh.setLevel(logging.INFO)
        sh.setLevel(logging.INFO)
    elif level == "DEBUG":
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        sh.setLevel(logging.DEBUG)
    elif level == "ERROR":
        logger.setLevel(logging.ERROR)
        fh.setLevel(logging.ERROR)
        sh.setLevel(logging.ERROR)

    formatter = logging.Formatter(prefix + ' ' + '%(filename)s:%(lineno)s %(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    if file_handler:
        logger.addHandler(fh)
    if stream_handler:
        logger.addHandler(sh)

    return logger

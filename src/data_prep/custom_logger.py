from loguru import logger


def loguru_setup(logfile):
    fmt = "{time} - {name} - {level} - {message}"
    logger.add(logfile, level="DEBUG", format=fmt, colorize=True)
    return logger

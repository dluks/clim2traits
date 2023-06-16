import logging
from bisect import bisect
from logging import Formatter, LogRecord, StreamHandler, getLogger
from typing import Dict


# Taken from
# https://stackoverflow.com/questions/14844970/modifying-logging-message-format-based-on-message-logging-level-in-python3/68154386#68154386
class LevelFormatter(Formatter):
    def __init__(self, formats: Dict[int, str], **kwargs):
        super().__init__()

        if "fmt" in kwargs:
            raise ValueError(
                "Format string must be passed to level-surrogate formatters, "
                "not this one"
            )

        self.formats = sorted(
            (level, Formatter(fmt, **kwargs)) for level, fmt in formats.items()
        )

    def format(self, record: LogRecord) -> str:
        idx = bisect(self.formats, (record.levelno,), hi=len(self.formats) - 1)
        level, formatter = self.formats[idx]
        return formatter.format(record)


def get_logger(name: str = None) -> logging.Logger:
    logger = getLogger(name)
    logger.setLevel(logging.INFO)
    
    logger.handlers.clear()

    handler = StreamHandler()
    handler.setFormatter(
        LevelFormatter(
            {
                logging.INFO: "%(message)s",
            }
        )
    )
    handler.setLevel(logging.INFO)

    logger.addHandler(handler)

    return logger

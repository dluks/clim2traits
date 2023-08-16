from __future__ import annotations

import time
from enum import Enum


class Unit(Enum):
    """Unit of measurement for the dataset."""

    def __init__(self, value: str, abbr: str):
        self._value_ = value
        self.abbr = abbr

    def __new__(cls, value: str, abbr: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.abbr = abbr
        return obj

    DEGREE = "degree", "deg"
    SECOND = "second", "s"
    METER = "meter", "m"
    KILOMETER = "kilometer", "km"
    MINUTE = "minute", "min"

    @classmethod
    def from_abbr(cls, abbr: str) -> Unit:
        for unit in cls:
            if unit.abbr == abbr:
                return unit
        raise ValueError(f"Unknown unit abbreviation: {abbr}")


class FileExt(Enum):
    """File extension for the raw data in the dataset."""

    TIF = "tif"
    NETCDF4 = "nc"
    GRID = "grd"
    ANY = "*"


def timer(func):
    """Decorator to time a function."""

    def inner(*args, **kwargs):
        t1 = time.time()
        f = func(*args, **kwargs)
        t2 = time.time()
        print(f"Runtime took {t2-t1:.2f} seconds")
        return f

    return inner

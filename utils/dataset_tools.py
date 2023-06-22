from enum import Enum


class Unit(Enum):
    """Unit of measurement for the dataset."""

    def __new__(cls, value: str, abbr: str):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.abbr = abbr
        return obj

    DEGREE = "degree", "deg"
    METER = "meter", "m"
    KILOMETER = "kilometer", "km"
    MINUTE = "minute", "min"


class FileExt(Enum):
    """File extension for the raw data in the dataset."""

    TIF = "tif"
    NETCDF4 = "nc"

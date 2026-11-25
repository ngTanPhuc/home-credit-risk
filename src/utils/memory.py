"""Memory helpers for tabular pipelines.

The project has a 16 GB RAM ceiling. These helpers keep dataframes in explicit,
small dtypes where possible and log memory use so heavy steps are observable.
They do not change values (apart from dtype representation), and they never use
the target column.
"""

from __future__ import annotations

import logging
import resource
import sys
from collections.abc import Iterable
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)


def frame_memory_mb(frame: pd.DataFrame) -> float:
    """Return the deep memory footprint of a dataframe in megabytes."""
    return float(frame.memory_usage(deep=True).sum()) / (1024.0**2)


def log_frame_memory(
    frame: pd.DataFrame, name: str, *, logger: logging.Logger | None = None
) -> float:
    """Log a dataframe's shape and memory footprint; return the size in MB."""
    size_mb = frame_memory_mb(frame)
    (logger or LOGGER).info(
        "%s: shape=%s, memory=%.1f MB",
        name,
        frame.shape,
        size_mb,
    )
    return size_mb


def downcast_frame(
    frame: pd.DataFrame,
    *,
    exclude: Iterable[str] = (),
    downcast_float: bool = True,
    downcast_int: bool = True,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with numerical columns in smaller dtypes.

    Parameters
    ----------
    frame:
        Input dataframe.
    exclude:
        Column names that must not be touched (for example IDs or the target).
    downcast_float:
        Convert float columns to ``float32`` when values fit.
    downcast_int:
        Convert integer columns without missing values to the smallest signed
        integer dtype that fits.

    Returns
    -------
    pandas.DataFrame
        A new dataframe with downcast numerical columns. Object, string,
        category, boolean, and datetime columns are left unchanged.

    Notes
    -----
    The function is label-free and safe to run before cross-validation. It does
    not impute, scale, or otherwise learn from data.
    """
    excluded = set(exclude)
    result = frame.copy()

    for column in result.columns:
        if column in excluded:
            continue

        series = result[column]

        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        if pd.api.types.is_float_dtype(series):
            if downcast_float:
                try:
                    result[column] = pd.to_numeric(series, downcast="float")
                except (TypeError, ValueError):
                    LOGGER.debug("Could not downcast float column %s", column)
            continue

        if pd.api.types.is_integer_dtype(series):
            if not downcast_int or series.isna().any():
                continue
            try:
                result[column] = pd.to_numeric(series, downcast="integer")
            except (TypeError, ValueError):
                LOGGER.debug("Could not downcast integer column %s", column)

    return result


def peak_rss_mb() -> float:
    """Return the process peak resident set size in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports kilobytes; macOS reports bytes.
    if sys.platform == "darwin":
        return float(usage) / (1024.0**2)
    return float(usage) / 1024.0


class MemoryTracker:
    """Context manager that logs elapsed peak memory for a code block.

    Example
    -------
    >>> with MemoryTracker("load application_train"):
    ...     frame = pd.read_csv(path)
    """

    def __init__(self, label: str, *, logger: logging.Logger | None = None) -> None:
        self.label = label
        self.logger = logger or LOGGER
        self.start_peak_mb: float | None = None
        self.end_peak_mb: float | None = None

    def __enter__(self) -> MemoryTracker:
        self.start_peak_mb = peak_rss_mb()
        self.logger.info("%s: start peak RSS=%.1f MB", self.label, self.start_peak_mb)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.end_peak_mb = peak_rss_mb()
        self.logger.info("%s: end peak RSS=%.1f MB", self.label, self.end_peak_mb)

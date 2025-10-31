"""Utility functions: metrics and small helpers.

This module provides:
- `classification_metrics` (existing)
- `setup_logger` for file + console logging
- `log_step` decorator to instrument pipeline steps
- `log_dataframe_info` to record DataFrame metadata
"""
from typing import Dict, Any, Callable, Optional
import logging
import logging.handlers
import datetime
import pathlib
import functools

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score


def classification_metrics(y_true, y_pred) -> Dict[str, float]:
    """Return common classification metrics.

    Kept unchanged from the original implementation.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def setup_logger(name: str = "smoking_cessation", log_dir: str = "logs", level: int = logging.INFO, force_reconfigure: bool = False) -> logging.Logger:
    """Configure and return a logger with both console and file handlers.

    - Console handler at `level` (INFO by default)
    - File handler at DEBUG for persistent logs stored in `log_dir`
    """
    log_path = pathlib.Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Timestamped filename for traceability
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = log_path / f"data_cleaning_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # capture everything; handlers will filter
    logger.propagate = False

    # Decide whether to (re)configure handlers.
    need_configure = force_reconfigure or not logger.handlers

    if not need_configure:
        # inspect existing handlers to see if they match requested config
        console_ok = False
        file_ok = False
        for h in logger.handlers:
            if isinstance(h, logging.StreamHandler):
                if h.level == level:
                    console_ok = True
            if isinstance(h, logging.handlers.RotatingFileHandler) or isinstance(h, logging.FileHandler):
                try:
                    existing_path = pathlib.Path(getattr(h, "baseFilename", ""))
                    if existing_path.parent.resolve() == log_path.resolve():
                        file_ok = True
                except Exception:
                    file_ok = False

        if not (console_ok and file_ok):
            need_configure = True

    if need_configure:
        # remove and close existing handlers
        for h in list(logger.handlers):
            try:
                logger.removeHandler(h)
                h.close()
            except Exception:
                pass

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # Rotating file handler to avoid unbounded growth (10 MB, 5 backups)
        try:
            fh = logging.handlers.RotatingFileHandler(filename, maxBytes=10 * 1024 * 1024, backupCount=5)
        except Exception:
            fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger


def _safe_repr_arg(arg: Any, max_str_len: int = 200) -> Any:
    """Return a safe representation for logging arguments (avoid printing whole DataFrames/arrays).

    Handles DataFrame, Series, ndarray, common collections and truncates long strings.
    """
    try:
        if isinstance(arg, pd.DataFrame):
            return f"<DataFrame shape={arg.shape}>"
        if isinstance(arg, pd.Series):
            name = getattr(arg, "name", None)
            return f"<Series name={name!r} length={len(arg)} dtype={arg.dtype}>"
        if isinstance(arg, np.ndarray):
            return f"<ndarray shape={arg.shape} dtype={arg.dtype}>"
        if isinstance(arg, (list, tuple, set)):
            length = len(arg)
            preview = list(arg)[:3]
            return f"<{type(arg).__name__} length={length} preview={preview}>"
        if isinstance(arg, dict):
            keys = list(arg.keys())[:5]
            return f"<dict length={len(arg)} keys_preview={keys}>"
        if isinstance(arg, str):
            if len(arg) > max_str_len:
                return arg[:max_str_len] + "..."
            return arg
    except Exception:
        try:
            return repr(arg)[:max_str_len]
        except Exception:
            return f"<{type(arg).__name__}>"

    return arg


def log_step(_func: Optional[Callable] = None, *, logger: Optional[logging.Logger] = None):
    """Decorator to log entry/exit, params (safely) and duration of pipeline steps.

    Can be used with or without arguments:

    @log_step
    def step(...):
        ...

    or

    @log_step(logger=my_logger)
    def step(...):
        ...
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or logging.getLogger("smoking_cessation")

            # Ensure there's at least a console handler so START/END messages appear
            if not log.handlers:
                sh = logging.StreamHandler()
                sh.setLevel(logging.INFO)
                sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
                log.addHandler(sh)

            start = datetime.datetime.now()

            safe_args = [_safe_repr_arg(a) for a in args]
            safe_kwargs = {k: _safe_repr_arg(v) for k, v in kwargs.items()}

            log.info(f"START {func.__name__} args={safe_args} kwargs={safe_kwargs}")
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.datetime.now() - start).total_seconds()
                log.info(f"END {func.__name__} (took {elapsed:.3f}s)")
                return result
            except Exception:
                log.exception(f"Exception in {func.__name__}")
                raise

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)


def log_dataframe_info(df: pd.DataFrame, name: str = "df", logger: Optional[logging.Logger] = None) -> None:
    """Log useful DataFrame metadata for data quality reporting.

    Logs shape, memory usage, dtypes and missing value counts.
    """
    log = logger or logging.getLogger("smoking_cessation")
    try:
        rows, cols = df.shape
        mem_bytes = int(df.memory_usage(deep=True).sum())

        # human readable memory
        def _human(n: int) -> str:
            for unit in ["B", "KB", "MB", "GB"]:
                if n < 1024.0:
                    return f"{n:.1f}{unit}"
                n /= 1024.0
            return f"{n:.1f}TB"

        log.debug(f"DataFrame '{name}': shape={rows}x{cols}, memory={_human(mem_bytes)}")
        # log columns and dtypes at DEBUG level to avoid noisy console output
        dtypes = df.dtypes.apply(lambda x: str(x)).to_dict()
        log.debug(f"{name} dtypes: {dtypes}")

        missing = df.isna().sum()
        # only include columns with missing values to keep logs compact
        missing_summary = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}
        log.debug(f"{name} missing values (per column): {missing_summary}")
    except Exception:
        log.exception("Failed to log dataframe info")

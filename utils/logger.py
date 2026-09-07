"""
SentinelCold — Structured Logger
Per-tier logging to console + file with timestamps.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def setup_logger(
    name: str = "sentinelcold",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure and return a logger that writes to both stderr and an
    optional log file.  Safe to call multiple times — subsequent calls
    return the same logger.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)

    if _CONFIGURED:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-7s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "sentinelcold") -> logging.Logger:
    """Return an existing logger (or a default one if setup_logger was never called)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger


def tier_logger(tier: int) -> logging.Logger:
    """Convenience: return a child logger named ``sentinelcold.tierN``."""
    return get_logger(f"sentinelcold.tier{tier}")

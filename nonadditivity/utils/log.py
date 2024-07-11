"""Logging submodule for the nonadditivity analysis code."""

import logging
import sys
from collections.abc import Callable
from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import Any

_ = logging.getLogger(__name__)


def setup_logging(
    level: int = logging.INFO,
    out_file: Path | None = None,
) -> None:
    """Set up module wide logging.

    If out_file is given the log is printed to
    stdout as well as into the out_file.

    Args:
        level (int, optional): logging level. Defaults to logging.INFO.
        out_file (Path | None, optional): file to write logs to. Defaults
        to None.
    """
    # set correct log level
    logging.root.setLevel(level)
    logging.root.handlers.clear()

    # create console handler and set level to info
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(module)s %(levelname)-7s %(message)s",
        datefmt="%Y/%b/%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)

    if out_file:
        # create file log
        out_file = out_file.absolute()

        console = logging.FileHandler(out_file, mode="a")
        console.setFormatter(formatter)
        logging.root.addHandler(console)


def get_logger() -> logging.Logger:
    """Get logger for nonadditivity module.

    Returns:
        logging.Logger: setup logger.
    """
    logger = logging.getLogger()
    return logger


def log_versions(
    logger: logging.Logger,
    packages: list[str],
    workflow: str,
) -> Callable:
    """Log package names and versions for decorated workflow.

    Decorator for functions that takes the logger as well as a list of
    strings containing valid package names and prints logs the versions
    of the package names given in packages.

    Args:
        logger (logging.Logger): logger to use
        packages (list[str]): packages to log versions for
        workflow: (str, optional): name of the workflow. Defaults to Free-
        Wilson Analysis.

    Returns:
        Callable: wrapped function
    """

    def inner(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Callable[..., Any]:
            versions = ", ".join(
                [f"{p} v{import_module(p).__version__}" for p in packages],
            )
            logger.info("Running %s using %s", workflow, versions)
            return func(*args, **kwargs)

        return wrapper

    return inner

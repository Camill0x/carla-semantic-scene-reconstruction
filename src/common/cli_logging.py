import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _logger_name(component: str) -> str:
    """Return a normalized logger name for the given component."""
    return component.strip().replace(" ", "_").lower()


def configure_logging(
    component: str,
    *,
    verbose: bool = False,
) -> logging.Logger:
    """Configure and return a console logger for a CLI component."""
    logger = logging.getLogger(_logger_name(component))
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)

    return logger

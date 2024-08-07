import logging
from pathlib import Path

from src.io.utils import make_path_safe

_default_log_dir = "logs/"
_default_log_name = "log.log"
_default_log_path = _default_log_dir + _default_log_name
_log_path = make_path_safe(_default_log_path)


def set_log_path(path: str):
    global _log_path
    global _default_log_path
    if str(_log_path) != _default_log_path:
        logger.warning(f"custom log path already set {_log_path} new path will be respected nonetheless")
    _log_path = make_path_safe(path, ".log")


def set_log_name(name: str):
    set_log_path(_default_log_dir + name)


def get_logger(name: str, script=False) -> logging.Logger:
    """
    This logger can be used in sub modules with script=False,
    scripts should use the method with script=True
    ```
    logger = logging.get_logger(__name__)               # for modules
    logger = logging.get_logger(__file__, script=True)  # for scripts
    ```
    """
    if script:
        name = Path(name).name  # only keep the name of the script file
        set_log_name(name)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s %(name)-12s [%(levelname).1s]: %(message)s",
        datefmt="%y.%m.%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(_log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


logger = get_logger(__name__)

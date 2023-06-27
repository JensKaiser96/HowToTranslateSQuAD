import logging

from src.tools.io import str_to_safe_path

_default_log_dir = "logs/"
_default_log_name = "log.log"
_default_log_path = _default_log_dir + _default_log_name
_log_path = str_to_safe_path(_default_log_path)


def set_log_path(path: str):
    global _log_path
    global _default_log_path
    if str(_log_path) != _default_log_path:
        logger.warn(
                f"custom log path already set {_log_path} "
                f"new path will be respected nonetheless"
                )
    _log_path = str_to_safe_path(path, ".log")


def set_log_name(name: str):
    set_log_path(_default_log_dir + name)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
            '%(asctime)s %(name)-12s [%(levelname).1s]: %(message)s',
            datefmt='%y.%m.%d %H:%M:%S'
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

import logging
import sys
from typing import Optional


def setup_logging(
        level: str = "INFO",
        log_file: Optional[str] = None
):
    """
    Настройка логирования.
    :param level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
    :param log_file: Если указан, логи пишутся и в файл
    :return:
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(level.upper())

    logger.handlers.clear()

    logger.addHandler(console_handler)

    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Получение logger с именем модуля.
    :param name: Имя модуля.
    :return:
    """
    return logging.getLogger(name)

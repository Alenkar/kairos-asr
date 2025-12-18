import logging
import sys
from typing import Optional
from pathlib import Path

def setup_logging(
    level: int = logging.NOTSET,
    log_file: Optional[str] = None,
    logger_name: Optional[str] = None
):
    """
    Настройка логирования для указанного логгера.
    :param level: Уровень логирования (например, logging.DEBUG, logging.INFO; по умолчанию NOTSET)
    :param log_file: Если указан, логи пишутся и в файл
    :param logger_name: Имя логгера (по умолчанию корневой)
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

from .logger import setup_logging, get_logger
from . import audio_utils
from .audio_utils import *
from .device_utils import check_device
from .text_processing import extract_sentences_from_words

import logging
if not logging.getLogger().hasHandlers():
    setup_logging()

__all__ = [
    "setup_logging",
    "get_logger",
    "check_device",
    "extract_sentences_from_words",
    "audio_utils",
]

from ._version import __version__
from .core import KairosASR, dtypes
from .utils import (
    audio_utils,
    check_device,
    extract_sentences_from_words,
    setup_logging,
)
from .models import ModelDownloader

setup_logging(logger_name="kairos_asr")

__all__ = [
    "KairosASR",
    "__version__",
    "check_device",
    "extract_sentences_from_words",
    "setup_logging",
    "dtypes",
    "audio_utils",
    "ModelDownloader",
]

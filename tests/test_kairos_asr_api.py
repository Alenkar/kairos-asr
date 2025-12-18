from pathlib import Path

import pytest
import shutil

from kairos_asr import KairosASR
from kairos_asr.models.utils.model_downloader import ModelDownloader
from kairos_asr.core import dtypes


TEST_WAV = Path(__file__).resolve().parents[1] / "test_data" / "record.wav"


def _ffmpeg_available() -> bool:
    """Проверка наличия ffmpeg в системе."""
    return shutil.which("ffmpeg") is not None


def _models_available() -> bool:
    """Проверка локального наличия всех требуемых моделей."""
    downloader = ModelDownloader()
    return all(downloader.check_local_file(name) is not None for name in downloader.model_files)


@pytest.fixture(scope="module")
def asr_cuda():
    """Фикстура для создания экземпляра KairosASR на CUDA (если доступно)."""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA не доступен на этой системе")
    except ImportError:
        pytest.skip("PyTorch не установлен")

    yield KairosASR(device="cuda")


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_python_api_transcribe(asr_cuda):
    result = asr_cuda.transcribe(str(TEST_WAV))
    assert result.full_text.strip(), "Текст транскрипции пустой"
    assert len(result.words) > 0, "Список слов пустой"
    assert len(result.sentences) > 0, "Список предложений пустой"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_python_api_transcribe_iterative_words(asr_cuda):
    generator = asr_cuda.transcribe_iterative(str(TEST_WAV), return_sentences=False)
    words = list(generator)
    assert len(words) > 0, "Слова не получены из генератора"
    assert isinstance(words[0], dtypes.word)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_python_api_transcribe_iterative_sentences(asr_cuda):
    generator = asr_cuda.transcribe_iterative(str(TEST_WAV), return_sentences=True)
    sentences = list(generator)
    assert len(sentences) > 0, "Предложения не получены из генератора"
    assert isinstance(sentences[0], dtypes.sentence)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_python_api_transcribe_iterative_with_progress(asr_cuda):
    generator = asr_cuda.transcribe_iterative(str(TEST_WAV), with_progress=True)
    items = list(generator)
    assert len(items) > 0
    item, progress = items[0]
    assert isinstance(item, dtypes.word)
    assert isinstance(progress, dtypes.progress)

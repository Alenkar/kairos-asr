import time
from pathlib import Path
from contextlib import contextmanager
from io import StringIO

import pytest
import cProfile
import pstats

from kairos_asr import KairosASR
from kairos_asr.models.utils.model_downloader import ModelDownloader

TEST_WAV = Path(__file__).resolve().parents[1] / "test_data" / "record.wav"


def _ffmpeg_available() -> bool:
    """Проверка наличия ffmpeg в системе."""
    import shutil
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


@contextmanager
def profile_context():
    """Контекстный менеджер для профилирования"""
    pr = cProfile.Profile()
    pr.enable()
    yield pr
    pr.disable()


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_profile_detailed_vad_segmentation(asr_cuda):
    """Тест профилирования VAD сегментации"""
    file_path = str(TEST_WAV)
    
    _ = asr_cuda.transcribe(wav_file=file_path)
    
    t1 = time.time()
    segments, boundaries = asr_cuda.silero_vad.segment_audio_file(file_path, sr=16000)
    t2 = time.time()
    
    assert len(segments) > 0, "Должен быть хотя бы один сегмент"
    assert len(segments) == len(boundaries), "Количество сегментов и границ должно совпадать"
    assert (t2 - t1) > 0, "Время выполнения должно быть положительным"
    assert (t2 - t1) < 1, "VAD не должен занимать слишком много времени (макс 2 секунды)"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_profile_detailed_process_segment(asr_cuda):
    """Тест профилирования обработки одного сегмента"""
    file_path = str(TEST_WAV)
    
    _ = asr_cuda.transcribe(wav_file=file_path)
    
    segments, boundaries = asr_cuda.silero_vad.segment_audio_file(file_path, sr=16000)
    
    if segments:
        segment = segments[0]
        t1 = time.time()
        result = asr_cuda._process_segment(segment, offset=0.0)
        t2 = time.time()
        
        assert isinstance(result, list), "Результат должен быть списком"
        assert (t2 - t1) > 0, "Время выполнения должно быть положительным"
        assert (t2 - t1) < 0.6, "Обработка сегмента не должна занимать слишком много времени (макс 30 сек)"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_profile_detailed_full_transcription(asr_cuda):
    """Тест профилирования полной транскрипции"""
    file_path = str(TEST_WAV)
    
    _ = asr_cuda.transcribe(wav_file=file_path)
    
    t1 = time.time()
    result = asr_cuda.transcribe(wav_file=file_path)
    t2 = time.time()
    
    assert result.full_text.strip(), "Текст транскрипции не должен быть пустым"
    assert len(result.words) > 0, "Должны быть слова"
    assert (t2 - t1) > 0, "Время выполнения должно быть положительным"
    assert (t2 - t1) < 3, "Полная транскрипция не должна занимать слишком много времени (макс 120 сек)"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_profile_detailed_multiple_runs(asr_cuda):
    """Тест профилирования множественных запусков"""
    file_path = str(TEST_WAV)
    
    _ = asr_cuda.transcribe(wav_file=file_path)
    
    times = []
    for i in range(3):
        t1 = time.time()
        result = asr_cuda.transcribe(wav_file=file_path)
        t2 = time.time()
        times.append(t2 - t1)
        assert result.full_text.strip(), f"Итерация {i+1}: текст не должен быть пустым"
    
    assert len(times) == 3, "Должно быть 3 измерения времени"
    assert all(t > 0 for t in times), "Все времена должны быть положительными"
    
    avg_time = sum(times) / len(times)
    assert avg_time > 0, "Среднее время должно быть положительным"
    assert avg_time < 3, "Среднее время не должно быть слишком большим (макс 120 сек)"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_profile_performance_cprofile(asr_cuda):
    """Тест профилирования с cProfile"""
    file_path = str(TEST_WAV)
    
    _ = asr_cuda.transcribe(wav_file=file_path)
    
    with profile_context() as pr:
        result = asr_cuda.transcribe(wav_file=file_path)
    
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    profile_output = s.getvalue()
    
    assert result.full_text.strip(), "Текст транскрипции не должен быть пустым"
    assert profile_output is not None, "Профиль должен быть создан"
    assert "transcribe" in profile_output or "kairos_asr" in profile_output, "Профиль должен содержать информацию о транскрипции"
    assert "function calls" in profile_output or "ncalls" in profile_output, "Профиль должен содержать статистику вызовов"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.skipif(not _models_available(), reason="Модели не найдены локально; выполните `kairos-asr download`")
@pytest.mark.integration
def test_profile_performance_timing(asr_cuda):
    """Тест измерения времени выполнения"""
    file_path = str(TEST_WAV)
    
    _ = asr_cuda.transcribe(wav_file=file_path)
    
    t1 = time.time()
    result = asr_cuda.transcribe(wav_file=file_path)
    t2 = time.time()
    
    elapsed = t2 - t1
    
    assert result.full_text.strip(), "Текст транскрипции не должен быть пустым"
    assert elapsed > 0, "Время выполнения должно быть положительным"
    assert elapsed < 3, "Транскрипция не должна занимать слишком много времени (макс 120 сек)"

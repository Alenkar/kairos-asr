import subprocess
from pathlib import Path

import pytest
import shutil

TEST_WAV = Path(__file__).resolve().parents[1] / "test_data" / "record.wav"


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


@pytest.mark.cli
def test_cli_help():
    cmd = ["python", "-m", "kairos_asr.core.cli", "--help"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Kairos-ASR — русское распознавание речи" in proc.stdout
    assert "download" in proc.stdout
    assert "list" in proc.stdout
    assert "doctor" in proc.stdout
    assert "transcribe" in proc.stdout


@pytest.mark.cli
def test_cli_no_command():
    cmd = ["python", "-m", "kairos_asr.core.cli"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Kairos-ASR — русское распознавание речи" in proc.stdout


@pytest.mark.cli
def test_cli_list():
    cmd = ["python", "-m", "kairos_asr.core.cli", "list"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Модели Kairos-ASR" in proc.stdout
    assert any(symbol in proc.stdout for symbol in ["✅", "❌"])


@pytest.mark.cli
def test_cli_doctor():
    cmd = ["python", "-m", "kairos_asr.core.cli", "doctor"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode in (0, 1)
    assert "Kairos-ASR Doctor" in proc.stdout
    assert "Overall status:" in proc.stdout


@pytest.mark.cli
def test_cli_download_all():
    cmd = ["python", "-m", "kairos_asr.core.cli", "download", "all"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert "encoder" in proc.stdout or "decoder" in proc.stdout or "❌" in proc.stderr


@pytest.mark.cli
def test_cli_download_invalid_model():
    cmd = ["python", "-m", "kairos_asr.core.cli", "download", "invalid_model"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 1
    assert "Неизвестная модель: invalid_model" in proc.stderr


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.integration
@pytest.mark.cli
def test_cli_transcribe_full_text():
    cmd = [
        "python",
        "-m",
        "kairos_asr.core.cli",
        "transcribe",
        str(TEST_WAV),
        "--device",
        "cuda",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"CLI ошибка: {proc.stderr}"
    assert proc.stdout.strip(), "Вывод транскрипции пустой"


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.integration
@pytest.mark.cli
def test_cli_transcribe_sentences():
    cmd = [
        "python",
        "-m",
        "kairos_asr.core.cli",
        "transcribe",
        str(TEST_WAV),
        "--sentences",
        "--device",
        "cuda",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert len(proc.stdout.strip().split("\n")) > 1


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg требуется для ASR-тестов")
@pytest.mark.integration
@pytest.mark.cli
def test_cli_transcribe_progress():
    cmd = [
        "python",
        "-m",
        "kairos_asr.core.cli",
        "transcribe",
        str(TEST_WAV),
        "--progress",
        "--device",
        "cuda",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0
    assert proc.stdout.strip() or proc.stderr.strip()

import os
import shutil
import unittest
import subprocess
from pathlib import Path

from kairos_asr import KairosASR
from kairos_asr.models.utils.model_downloader import ModelDownloader


TEST_WAV = Path(__file__).resolve().parents[2] / "test_data" / "record.wav"


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _models_available() -> bool:
    downloader = ModelDownloader()
    for name in downloader.model_files:
        if downloader.get_model_path(name) is None:
            return False
    return True


@unittest.skipUnless(_ffmpeg_available(), "ffmpeg is required for ASR smoke test")
@unittest.skipUnless(_models_available(), "Models not found locally; run `kairos-asr download` first")
class TestSmoke(unittest.TestCase):
    def test_python_api_transcribe(self):
        asr = KairosASR(device="cpu")
        result = asr.transcribe(str(TEST_WAV))
        self.assertTrue(result.full_text.strip(), "Transcription text is empty")
        self.assertGreater(len(result.words), 0, "Words list is empty")
        self.assertGreater(len(result.sentences), 0, "Sentences list is empty")

    def test_cli_transcribe_json(self):
        cmd = [
            "python",
            "-m",
            "kairos_asr.core.cli",
            "transcribe",
            str(TEST_WAV),
            "--device",
            "cpu",
            "--format",
            "json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            self.fail(f"CLI failed: {proc.stderr}")
        self.assertIn("{", proc.stdout, "CLI JSON output looks empty")


if __name__ == "__main__":
    unittest.main()


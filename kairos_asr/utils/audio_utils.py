import logging
import warnings
from subprocess import CalledProcessError, run
import torch

logger = logging.getLogger(__name__)

def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Loads audio and resamples to specified rate. Supports multi-channel.

    :param audio_path: Path to audio file.
    :param sample_rate: Target sample rate.
    :return: Tensor [channels, samples].
    """
    logger.debug(f"Load audio file")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        audio = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        raise RuntimeError(f"Failed to load audio: {audio_path}") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        audio_tensor = torch.frombuffer(audio, dtype=torch.int16).float() / 32768.0

    return audio_tensor

import logging
import warnings
from subprocess import CalledProcessError, run
from typing import Union, Optional
import torch
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)

def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Загружает аудиофайл и производит resample до указанной частоты. Поддерживает многоканальный звук.

    :param audio_path: Путь к аудиофайлу.
    :param sample_rate: Целевая частота дискретизации.
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


def prepare_audio_array(
    audio_array: Union[np.ndarray, torch.Tensor],
    sample_rate: Optional[int],
    target_sample_rate: int,
) -> torch.Tensor:
    """
    Подготавливает аудио массив для обработки: конвертирует в torch.Tensor,
    нормализует и выполняет ресемплинг при необходимости.

    :param audio_array: Numpy массив или torch.Tensor с аудио данными [samples] или [channels, samples]
    :param sample_rate: Исходная частота дискретизации (если None, используется target_sample_rate)
    :param target_sample_rate: Целевая частота дискретизации
    :return: Обработанный torch.Tensor с нормализованным аудио
    :raises ValueError: Если тип audio_array не поддерживается
    """
    if sample_rate is None:
        sample_rate = target_sample_rate

    if isinstance(audio_array, np.ndarray):
        audio_tensor = torch.from_numpy(audio_array).float()
    elif isinstance(audio_array, torch.Tensor):
        audio_tensor = audio_array.float()
    else:
        raise ValueError(
            f"Неподдерживаемый тип аудио: {type(audio_array)}. "
            "Ожидается numpy.ndarray или torch.Tensor"
        )

    if audio_tensor.abs().max() > 1.0:
        audio_tensor = audio_tensor / audio_tensor.abs().max()

    if sample_rate != target_sample_rate:
        logger.debug(f"Resampling from {sample_rate} Hz to {target_sample_rate} Hz")
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        audio_tensor = torchaudio.functional.resample(
            audio_tensor, sample_rate, target_sample_rate
        )

        if audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.squeeze(0)

    return audio_tensor

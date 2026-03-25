import logging

import torch

logger = logging.getLogger(__name__)


def _mps_available() -> bool:
    return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()


def normalize_device(device: str) -> str:
    """
    Нормализует устройство: 'auto' -> cuda/mps/cpu, 'metal' -> mps.
    :param device: 'auto', 'cuda', 'cuda:0', 'mps', 'metal', 'cpu'
    :return: нормализованная строка устройства
    """
    if not device:
        device = "auto"

    device_lower = device.lower()

    if device_lower in ("auto",):
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if device_lower.startswith("cuda"):
        return device_lower
    if device_lower in ("mps", "metal"):
        return "mps"
    if device_lower == "cpu":
        return "cpu"

    logger.warning(f"Неизвестное устройство '{device}', используется CPU.")
    return "cpu"


def check_device(device: str) -> torch.device:
    """
    Проверка доступности GPU.
    :param device: 'auto', 'cuda', 'cuda:0', 'mps', 'metal', 'cpu'.
    :return:
    """
    device_norm = normalize_device(device)

    if device_norm.startswith("cuda") and not torch.cuda.is_available():
        if _mps_available():
            logger.warning("CUDA недоступна, используется MPS.")
            return torch.device("mps")
        logger.warning("CUDA недоступна, используется CPU.")
        return torch.device("cpu")

    if device_norm == "mps" and not _mps_available():
        logger.warning("MPS (Metal) недоступен, используется CPU.")
        return torch.device("cpu")

    return torch.device(device_norm)


def prepare_audio_tensor(wav: torch.Tensor) -> torch.Tensor:
    """
    Приводит аудио к формату [1, samples].
    :param wav: Аудио.
    :return:
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() > 2:
        wav = wav.squeeze(0)
    return wav

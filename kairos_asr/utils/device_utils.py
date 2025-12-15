import logging

import torch

logger = logging.getLogger(__name__)

def check_device(device: str):
    """
    Проверка доступности GPU.
    :param device: 'cuda', 'cuda:0', 'cpu'.
    :return:
    """
    if not torch.cuda.is_available():
        logger.warning('GPU недоступен, используется CPU.')
        return torch.device("cpu")
    else:
        return torch.device(device)


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

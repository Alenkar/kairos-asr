import logging

import numpy as np
import torch
import torchaudio

from ..utils.device_utils import check_device, normalize_device

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Извлекает характеристики мел-спектрограммы из аудиофайла.
    """

    def __init__(
            self,
            sample_rate: int = 16000,
            features: int = 64,
            device: str = "auto"
    ):
        """
        Инициализирует экстрактор признаков.

        :param sample_rate: Частота дискретизации звука.
        :param features: Количество признаков Mel.
        :param device: Устройство Torch ('auto', 'cuda:0', 'mps', 'metal' или 'cpu').
        :return:
        """
        logger.debug("Starting initialization of FeatureExtractor")

        self.sample_rate = sample_rate
        self.hop_length = sample_rate // 100
        self.win_length = sample_rate // 40
        self.n_fft = sample_rate // 40

        device_norm = normalize_device(device)
        if device_norm == "mps":
            logger.info("MPS (Metal) не используется для MelSpectrogram, принудительно CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = check_device(device_norm)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=features,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            center=True,
            pad_mode="reflect",
            power=2.0
        ).to(self.device)

        logger.info(f"FeatureExtractor initialized on device: {self.device}")

    def __call__(self, audio: torch.Tensor) -> np.ndarray:
        if audio.device != self.device:
            audio = audio.to(self.device)
        x = self.mel_transform(audio)
        x = torch.log(x.clamp(min=1e-9, max=1e9))
        return x.cpu().numpy()

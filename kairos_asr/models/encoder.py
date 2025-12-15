import logging

import torch
import numpy as np

from ..core.feature_extractor import FeatureExtractor
from ..models.onnx_model import ONNXModel
from ..utils.device_utils import check_device, prepare_audio_tensor

logger = logging.getLogger(__name__)

class KairosEncoder:
    """
    Encoder для KairosASR.
    """
    def __init__(
        self,
        encoder_path: str,
        device: str = "cuda",
    ):
        """
        Инициализирует Encoder-модель.
        :param encoder_path: Путь к encoder.onnx (опционально)
        :param device: Устройство ('cuda', 'cuda:0' или 'cpu').
        """
        logger.debug("Initialization: KairosEncoder")

        self.sample_rate = 16000
        self.dtype = torch.float32
        self.max_letters_per_frame = 10

        self.device = check_device(device)

        logger.debug(f"Device: {self.device}")

        self.encoder = ONNXModel(encoder_path, device=device)
        self.feature_extractor = FeatureExtractor(sample_rate=self.sample_rate, device=device)

    def _encode(self, features: np.ndarray) -> np.ndarray:
        """
        Запуск encoder.
        :param features:
        :return:
        """
        length = np.array([features.shape[-1]], dtype=np.int64)
        inputs = [features.astype(np.float32), length]
        return self.encoder.run(
            self.encoder.output_names(),
            self.encoder.get_input_dict(inputs)
        )[0]

    def encode_segment(self, segment: torch.Tensor) -> [np.ndarray | None, float | None]:
        """
        Обработка сегмента: извлекает слова с точными timestamps.
        :param segment:
        :return:
        """
        logger.debug(f"Encode segment")
        wav = prepare_audio_tensor(segment)
        features = self.feature_extractor(wav)

        if features.shape[-1] == 0:
            return None, None

        enc_features = self._encode(features)
        if enc_features.shape[-1] == 0:
            return None, None

        audio_len_sec = wav.shape[-1] / self.sample_rate
        frame_duration = audio_len_sec / enc_features.shape[-1]

        return enc_features, frame_duration

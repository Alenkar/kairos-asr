import logging
from typing import List
import torch
import numpy as np

from ..models.onnx_model import ONNXModel
from ..utils.device_utils import check_device

logger = logging.getLogger(__name__)

class KairosDecoder:
    """
    Decoder для KairosASR.
    """
    def __init__(
        self,
        decoder_path: str,
        joint_path: str,
        blank_id: int,
        device: str = "cuda",
    ):
        """
        Инициализирует Decoder-модель.
        :param decoder_path: Путь к decoder.onnx (опционально).
        :param joint_path: Путь к joint.onnx (опционально).
        :param blank_id: Путь к tokenizer.model (опционально).
        :param device: Устройство ('cuda', 'cuda:0' или 'cpu').
        """
        logger.debug("Initialization: KairosDecoder")

        self.sample_rate = 16000
        self.dtype = torch.float32
        self.max_letters_per_frame = 10
        self.blank_id = blank_id
        self.device = check_device(device)

        logger.debug(f"Device: {self.device}")

        self.decoder = ONNXModel(decoder_path, device=device)
        self.joint = ONNXModel(joint_path, device=device)

        for inp in self.decoder.session.get_inputs():
            if len(inp.shape) == 3:
                self.num_layers, _, self.hidden_size = inp.shape
                break
        else:
            raise RuntimeError("Не удалось определить размер скрытого состояния декодера.")

    def _get_initial_states(self):
        """
        Возвращает пустые состояния для начала декодирования.
        :return:
        """
        return [
            np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32),
            np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32),
        ]

    def decode_segment(self, enc_features: torch.Tensor) -> [List[int], List[int]]:
        """
        Запуск decoder.
        :param enc_features:
        :return:
        """
        logger.debug(f"Decode segment")

        token_ids: List[int] = []
        token_frames: List[int] = []

        states = self._get_initial_states()
        prev_token = self.blank_id
        for t in range(enc_features.shape[-1]):
            emitted = 0
            while emitted < self.max_letters_per_frame:
                pred_in = [np.array([[prev_token]], dtype=np.int64)] + states
                pred_out = self.decoder.run(None, self.decoder.get_input_dict(pred_in))
                pred_h = pred_out[0].swapaxes(1, 2)

                joint_in = [enc_features[:, :, [t]], pred_h]
                logits = self.joint.run(None, self.joint.get_input_dict(joint_in))[0]
                token = logits.argmax(axis=-1).item()

                if token != self.blank_id:
                    token_ids.append(token)
                    token_frames.append(t)
                    prev_token = token
                    states = pred_out[1:]
                    emitted += 1
                else:
                    break

        return token_ids, token_frames

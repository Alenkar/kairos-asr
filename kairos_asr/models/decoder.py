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
        logger.debug("Starting initialization of KairosDecoder")

        self.sample_rate = 16000
        self.dtype = torch.float32
        self.max_letters_per_frame = 10
        self.blank_id = blank_id
        self.device = check_device(device)

        self.decoder = ONNXModel(decoder_path, device=device)
        self.joint = ONNXModel(joint_path, device=device)

        logger.info(f"KairosDecoder initialized on device: {self.device}")
        for inp in self.decoder.session.get_inputs():
            if len(inp.shape) == 3:
                self.num_layers, _, self.hidden_size = inp.shape
                break
        else:
            raise RuntimeError("Не удалось определить размер скрытого состояния декодера.")

    def _get_initial_states(self):
        """
        Возвращает пустые состояния для начала декодирования.

        :return: Список numpy-массивов состояний.
        """
        return [
            np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32),
            np.zeros((self.num_layers, 1, self.hidden_size), dtype=np.float32),
        ]

    def decode_segment(self, enc_features: np.ndarray) -> [List[int], List[int]]:
        """
        Декодирование encoder-выходов в последовательность токенов.

        :param enc_features: Encoder-выходы формы [B, D, T].
        :return:
            token_ids — список ID токенов,
            token_frames — список индексов encoder-фреймов.
        """
        logger.debug(f"Decode segment")

        token_ids: List[int] = []
        token_frames: List[int] = []

        states = self._get_initial_states()
        prev_token = self.blank_id
        
        num_decoder_inputs = len(self.decoder._cached_input_names)
        num_joint_inputs = len(self.joint._cached_input_names)
        
        for t in range(enc_features.shape[-1]):
            emitted = 0
            while emitted < self.max_letters_per_frame:
                pred_in = [np.array([[prev_token]], dtype=np.int64)] + states
                decoder_input_dict = {
                    self.decoder._cached_input_names[i]: pred_in[i]
                    for i in range(num_decoder_inputs)
                }
                
                pred_out = self.decoder.run(decoder_input_dict)
                pred_h = pred_out[0].swapaxes(1, 2)

                joint_in = [enc_features[:, :, t:t+1], pred_h]
                joint_input_dict = {
                    self.joint._cached_input_names[i]: joint_in[i]
                    for i in range(num_joint_inputs)
                }
                
                logits = self.joint.run(joint_input_dict)[0]
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

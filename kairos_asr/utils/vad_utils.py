import logging

import torch
from typing import List, Tuple

from ..utils.device_utils import check_device
from ..utils.audio_utils import load_audio

from silero_vad import load_silero_vad, get_speech_timestamps

logger = logging.getLogger(__name__)

class SileroVAD:
    def __init__(self, device: str='cuda'):
        """
        SileroVAD для разбиения длинного аудио на сегменты.
        :param device:
        """
        logger.debug("Initialization: SileroVAD")

        self.device = check_device(device)
        self.silero_model = self.get_silero_model()
        logger.info(f"Model SileroVAD loaded : {self.device}")

    def get_silero_model(self):
        """
        Загружает модель Silero VAD и необходимые утилиты.
        Использует pip-пакет silero-vad для фиксированной версии.
        """
        model = load_silero_vad()
        model = model.to(self.device)
        return model

    @staticmethod
    def preprocessing(audio: torch.Tensor) -> torch.Tensor:
        """
        Подготовка аудио для сегментации.
        :param audio:
        :return:
        """
        if audio.dim() > 1 and audio.shape[0] > 1:
            return audio.mean(dim=0, keepdim=False)
        elif audio.dim() > 1:
            return audio.squeeze()
        else:
            return audio

    @staticmethod
    def _update_segments(
            strict_limit_duration,
            segments,
            audio,
            boundaries,
            sr,
            curr_start: float,
            curr_end: float,
            curr_duration: float):
        """
        Обновление сегментов.
        :param strict_limit_duration:
        :param segments:
        :param boundaries:
        :param sr:
        :param curr_start:
        :param curr_end:
        :param curr_duration:
        :return:
        """
        if curr_duration > strict_limit_duration:
            max_segments = int(curr_duration / strict_limit_duration) + 1
            segment_duration = curr_duration / max_segments
            curr_end = curr_start + segment_duration
            for _ in range(max_segments - 1):
                segments.append(audio[int(curr_start * sr): int(curr_end * sr)])
                boundaries.append((curr_start, curr_end))
                curr_start = curr_end
                curr_end += segment_duration
        segments.append(audio[int(curr_start * sr): int(curr_end * sr)])
        boundaries.append((curr_start, curr_end))

    def segment_audio_file(
            self,
            wav_file: str,
            sr: int = 16000,
            max_duration: float = 22.0,
            min_duration: float = 15.0,
            strict_limit_duration: float = 30.0,
            new_chunk_threshold: float = 0.2,
    ) -> Tuple[List[torch.Tensor], List[Tuple[float, float]]]:
        """
        Сегментирует аудиофайл на чанки, используя Silero VAD 5/6.
        :param wav_file:
        :param sr:
        :param max_duration:
        :param min_duration:
        :param strict_limit_duration:
        :param new_chunk_threshold:
        :return:
        """
        logger.debug(f"VAD Segmentation : start")
        audio = load_audio(wav_file)
        audio_for_vad = self.preprocessing(audio)

        speech_timestamps_samples = get_speech_timestamps(
            audio_for_vad.to(self.device),
            self.silero_model,
            sampling_rate=sr,
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100
        )

        logger.debug(f"Processing of received segments")

        # ToDo segments находятся в памяти, возможно не стоит хранить их так

        segments: List[torch.Tensor] = []
        curr_duration = 0.0
        curr_start = 0.0
        curr_end = 0.0
        boundaries: List[Tuple[float, float]] = []

        for segment in speech_timestamps_samples:
            start = max(0, segment['start'] / sr)
            end = min(audio.shape[0] / sr, segment['end'] / sr)
            if curr_duration > new_chunk_threshold and (
                    curr_duration + (end - curr_end) > max_duration
                    or curr_duration > min_duration
            ):
                self._update_segments(
                    strict_limit_duration, segments, audio, boundaries, sr,
                    curr_start, curr_end, curr_duration)
                curr_start = start
            curr_end = end
            curr_duration = curr_end - curr_start

        if curr_duration > new_chunk_threshold:
            self._update_segments(
                strict_limit_duration, segments, audio, boundaries, sr,
                curr_start, curr_end, curr_duration)

        logger.debug(f"VAD Segmentation : complete")
        return segments, boundaries

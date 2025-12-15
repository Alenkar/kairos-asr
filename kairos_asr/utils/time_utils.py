import logging
import time

import torch

logger = logging.getLogger(__name__)

class CalculatedRemainingTime:
    """
    Дополнительный класс для расчета оставшегося времени обработки фала при итеративно обработке.
    """
    def __init__(self):
        self.start_seg = 0
        self.processed_time = 0.0
        self.processed_audio_duration = 0.0
        self.sample_rate = 0
        self.total_audio_duration = 0

    def load_info(self, sample_rate: int, total_audio_duration: float):
        """
        Загрузка информации об аудио.
        """
        self.start_seg = 0
        self.processed_time = 0.0
        self.processed_audio_duration = 0.0
        self.sample_rate = sample_rate
        self.total_audio_duration = total_audio_duration

    def step(self):
        """
        Шаг времени.
        """
        self.start_seg = time.time()

    def calc(self, segment_shape: int) -> float:
        """
        Расчет оставшегося времени обработки.
        """
        seg_time = time.time() - self.start_seg
        seg_duration = segment_shape / self.sample_rate
        self.processed_time += seg_time
        self.processed_audio_duration += seg_duration
        if self.processed_time > 0 and self.total_audio_duration > 0:
            avg_speed = self.processed_audio_duration / self.processed_time
            remaining_audio_duration = self.total_audio_duration - self.processed_audio_duration
            return remaining_audio_duration / avg_speed if avg_speed > 0 else 0.0
        else:
            return 0.0

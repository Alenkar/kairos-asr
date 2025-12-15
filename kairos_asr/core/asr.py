import logging
import sentencepiece as spm
import torch
from typing import List, Dict, Generator, Tuple, Optional

from ..core.data_classes import dtypes

from ..models.decoder import KairosDecoder
from ..models.encoder import KairosEncoder
from ..models.utils.model_downloader import ModelDownloader

from ..utils.vad_utils import SileroVAD
from ..utils.device_utils import check_device
from ..utils.text_processing import (
    extract_sentences_from_words, extract_words_from_tokens
)
from ..utils.time_utils import CalculatedRemainingTime

logger = logging.getLogger(__name__)


class KairosASR:
    """
    Модель автоматического распознавания речи на базе Gigaam.
    """
    def __init__(
        self,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        joint_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        force_download: bool = False
    ):
        """
        Инициализирует ASR-модель с опциональными путями к весам.

        :param encoder_path: Путь к encoder.onnx (если None, будет скачан автоматически, если его нет).
        :param decoder_path: Путь к decoder.onnx (если None, будет скачан автоматически, если его нет).
        :param joint_path: Путь к joint.onnx (если None, будет скачан автоматически, если его нет).
        :param tokenizer_path: Путь к tokenizer.model (если None, будет скачан автоматически, если его нет).
        :param device: Устройство ('cuda', 'cuda:0' или 'cpu').
        :param force_download: Принудительно скачивать модели и перезаписывать.
        """
        logger.debug("Initialization: KairosASR")

        self.sample_rate = 16000
        self.dtype = torch.float32
        self.max_letters_per_frame = 10
        self.device = check_device(device)

        model_paths = {
            "encoder": encoder_path,
            "decoder": decoder_path,
            "joint": joint_path,
            "tokenizer": tokenizer_path,
        }

        model_downloader = ModelDownloader()
        resolved_paths = model_downloader.resolve_models_path(
            model_paths=model_paths,
            force_download=force_download,
        )

        # Дополнительный класс для расчета оставшегося времени
        self.calculated_remaining_time = CalculatedRemainingTime()

        # Сегментатор Silero
        self.silero_vad = SileroVAD(device=device)

        # Токенизатор
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(model_file=resolved_paths["tokenizer"])
        blank_id = self.tokenizer.GetPieceSize()

        # Encoder и Decoder
        self.encoder = KairosEncoder(
            encoder_path=resolved_paths["encoder"], device=device
        )
        self.decoder = KairosDecoder(
            decoder_path=resolved_paths["decoder"], joint_path=resolved_paths["joint"],
            blank_id=blank_id, device=device
        )
        logger.debug(f"KairosASR initialized on device: {self.device}")

    def _process_segment(self, segment: torch.Tensor, offset: float = 0.0) -> List[dtypes.word]:
        """
        Обработка сегмента: извлекает слова с точными timestamps.
        :param segment:
        :param offset:
        :return:
        """
        logger.debug(f"Process segment")

        enc_features, frame_duration = self.encoder.encode_segment(segment)
        if enc_features is None:
            return []

        token_ids, token_frames = self.decoder.decode_segment(enc_features)

        pieces = [self.tokenizer.IdToPiece(tid) for tid in token_ids]
        return extract_words_from_tokens(pieces, token_frames, frame_duration, offset)

    def transcribe(
            self,
            wav_file: str,
            pause_threshold: float = 2.0,
            **vad_kwargs,
    ) -> dtypes.tts_result:
        """
        Основная функция для файла. Возвращает полную структуру данных.
        1. Полный текст (чистый).
        2. Слова с timestamps.
        3. Предложения с timestamps.

        :param wav_file: Путь к файлу
        :param pause_threshold:

        :return:
        """
        logger.debug(f"Audio transcription: start")

        # ToDo change wav_file to data numpy ( str | ndarray)
        segments, boundaries = self.silero_vad.segment_audio_file(
            wav_file, sr=self.sample_rate, **vad_kwargs
        )

        all_words: List[dtypes.word] = []

        for segment, (start_offset, _) in zip(segments, boundaries):
            segment_words = self._process_segment(segment, offset=start_offset)
            all_words.extend(segment_words)

        sentences_objs = extract_sentences_from_words(all_words, pause_threshold=pause_threshold)

        full_text_str = " ".join([s.text for s in sentences_objs])

        logger.debug(f"Audio transcription : complete")

        return dtypes.tts_result(
            full_text=full_text_str,
            words=all_words,
            sentences=sentences_objs
        )

    def transcribe_iterative(
            self,
            wav_file: str,
            return_sentences: bool = False,
            with_progress: bool = False,
            pause_threshold: float = 2.0,
            **vad_kwargs
    ) -> Generator[dtypes.word | dtypes.sentence | Tuple[dtypes.word | dtypes.sentence, Dict[str, float]], None, None]:
        """
        Генератор для потокового вывода.
        :param wav_file: Путь к аудиофайлу.
        :param return_sentences: Если True, собирает слова в предложения и выводит экземпляры Sentence.
        :param with_progress: Если True, yield (объект Word/Sentence, progress_dict),
            где progress = {'percent': float, 'segment': int, 'total_segments': int}.
        :param pause_threshold: Порог паузы для формирования предложений (в секундах).
        :param vad_kwargs: Параметры VAD.
        :return: Экземпляр Word или Sentence (или кортеж с progress, если with_progress=True).
        """
        logger.debug(f"Audio transcription (Yield) : start")
        segments, boundaries = self.silero_vad.segment_audio_file(
            wav_file, sr=self.sample_rate, **vad_kwargs
        )

        total_segments = len(segments)
        total_audio_duration = sum(segment.shape[0] for segment in segments) / self.sample_rate

        current_segment = 0
        buffer_words: List[dtypes.word] = []
        estimated_remaining_time = 0.0
        if with_progress:
           self.calculated_remaining_time.load_info(self.sample_rate, total_audio_duration)

        for segment, (start_offset, _) in zip(segments, boundaries):
            current_segment += 1

            if with_progress:
                estimated_remaining_time = self.calculated_remaining_time.step()

            segment_words = self._process_segment(segment, offset=start_offset)

            if with_progress:
                estimated_remaining_time = self.calculated_remaining_time.calc(segment.shape[0])

            if return_sentences:
                buffer_words.extend(segment_words)
                sentences = extract_sentences_from_words(buffer_words, pause_threshold=pause_threshold)
                for sent in sentences[:-1] if sentences else []:
                    if with_progress:
                        progress = dtypes.progres(
                            percent=round((current_segment / total_segments) * 100, 2),
                            segment=current_segment,
                            total_segments=total_segments,
                            time_remaining=round(estimated_remaining_time, 2)
                        )
                        yield sent, progress
                    else:
                        yield sent
                if sentences:
                    last_sent_words = len(sentences[-1].text.split())
                    buffer_words = buffer_words[-last_sent_words:] if len(sentences) > 1 else buffer_words
            else:
                for word in segment_words:
                    if with_progress:
                        progress = dtypes.progres(
                            percent=round((current_segment / total_segments) * 100, 2),
                            segment=current_segment,
                            total_segments=total_segments,
                            time_remaining=round(estimated_remaining_time, 2)
                        )
                        yield word, progress
                    else:
                        yield word
        if return_sentences and buffer_words:
            logger.debug(f"Audio transcription (Yield) : complete")
            remaining_sentences = extract_sentences_from_words(buffer_words, pause_threshold=pause_threshold)
            for sent in remaining_sentences:
                if with_progress:
                    progress = dtypes.progres(
                        percent=100.0,
                        segment=total_segments,
                        total_segments=total_segments,
                        time_remaining=0.0
                    )
                    yield sent, progress
                else:
                    yield sent
        else:
            logger.debug(f"Audio transcription (Yield) : complete")

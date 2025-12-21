import logging
import sentencepiece as spm
import torch
import numpy as np
from typing import List, Generator, Tuple, Optional, Union, overload, Literal

from ..core import dtypes

from ..models.decoder import KairosDecoder
from ..models.encoder import KairosEncoder
from ..models.utils.model_downloader import ModelDownloader

from ..utils.vad_utils import SileroVAD
from ..utils.device_utils import check_device
from ..utils.text_processing import (
    extract_sentences_from_words, extract_words_from_tokens
)
from ..utils.time_utils import CalculatedRemainingTime
from ..utils.audio_utils import prepare_audio_array

logger = logging.getLogger(__name__)

class KairosASR:
    """
    Модель автоматического распознавания речи на основе Gigaam.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        force_download: bool = False
    ):
        """
        Инициализирует модель ASR с необязательными путями к весам.

        :param model_path: Пользовательский путь к файлам *.onnx.
                           Если путь пуст, файлы будут загружены автоматически.
        :param device: Устройство ('cuda', 'cuda:0' или 'cpu').
        :param force_download: Принудительная загрузка моделей и перезапись.
        """
        logger.debug("Starting initialization of KairosASR")

        self.sample_rate = 16000
        self.dtype = torch.float32
        self.max_letters_per_frame = 10
        self.device = check_device(device)
        logger.debug(f"Device checked and set to: {self.device}")

        model_downloader = ModelDownloader(model_path=model_path)
        resolved_paths = model_downloader.resolve_models_path(force_download=force_download)
        logger.debug("Model paths resolved")

        self.calculated_remaining_time = CalculatedRemainingTime()
        logger.debug("CalculatedRemainingTime initialized")

        self.silero_vad = SileroVAD()
        logger.debug("SileroVAD initialized")

        self.tokenizer = spm.SentencePieceProcessor()
        tokenizer_path = resolved_paths["tokenizer"]
        logger.debug(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer.Load(model_file=tokenizer_path)
        blank_id = self.tokenizer.GetPieceSize()
        logger.debug(f"Tokenizer loaded, blank_id: {blank_id}")

        self.encoder = KairosEncoder(
            encoder_path=resolved_paths["encoder"], device=device
        )
        logger.debug("KairosEncoder initialized")

        self.decoder = KairosDecoder(
            decoder_path=resolved_paths["decoder"], joint_path=resolved_paths["joint"],
            blank_id=blank_id, device=device
        )
        logger.debug("KairosDecoder initialized")

        logger.info(f"KairosASR fully initialized on device: {self.device}")

    def _process_segment(self, segment: torch.Tensor, offset: float = 0.0) -> List[dtypes.word]:
        """
        Обрабатывает сегмент: извлекает слова с точными временными метками.

        :param segment: Сегмент аудио.
        :param offset:
        :return:
        """
        logger.debug(f"Processing segment with offset: {offset}")

        enc_features, frame_duration = self.encoder.encode_segment(segment)
        if enc_features is None:
            logger.warning("Encoder returned None features for segment")
            return []

        token_ids, token_frames = self.decoder.decode_segment(enc_features)
        logger.debug(f"Decoded {len(token_ids)} tokens")

        pieces = [self.tokenizer.IdToPiece(tid) for tid in token_ids]
        words = extract_words_from_tokens(pieces, token_frames, frame_duration, offset)
        logger.debug(f"Extracted {len(words)} words from tokens")

        return words

    def transcribe_chunk(
            self,
            audio_array: Union[np.ndarray, torch.Tensor],
            sample_rate: Optional[int] = None,
            use_vad: bool = False,
            pause_threshold: float = 2.0,
    ) -> dtypes.tts_result:
        """
        Обрабатывает короткий чанк аудио напрямую без VAD (для потоковой обработки).
        Для коротких чанков (< 5 секунд) рекомендуется use_vad=False для скорости.
        
        :param audio_array: Numpy массив или torch.Tensor с аудио данными [samples] или [channels, samples]
        :param sample_rate: Частота дискретизации (обязательно если указан audio_array)
        :param use_vad: Если True, использует VAD для сегментации (медленнее, но точнее для длинных чанков)
        :param pause_threshold: Порог паузы для формирования предложений (в секундах)
        :return: Результат транскрипции
        """
        audio_tensor = prepare_audio_array(
            audio_array, sample_rate, self.sample_rate
        )
        
        if use_vad:
            logger.debug("Processing chunk with VAD")
            segments, boundaries = self.silero_vad.segment_audio_tensor(
                audio_tensor, sr=self.sample_rate
            )
            all_words: List[dtypes.word] = []
            for segment, (start_offset, _) in zip(segments, boundaries):
                segment_words = self._process_segment(segment, offset=start_offset)
                all_words.extend(segment_words)
        else:
            logger.debug("Processing chunk directly without VAD")
            all_words = self._process_segment(audio_tensor, offset=0.0)
        
        sentences_objs = extract_sentences_from_words(all_words, pause_threshold=pause_threshold)
        full_text_str = " ".join([s.text for s in sentences_objs])
        
        return dtypes.tts_result(
            full_text=full_text_str,
            words=all_words,
            sentences=sentences_objs
        )

    def transcribe(
            self,
            wav_file: Optional[str] = None,
            audio_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
            sample_rate: Optional[int] = None,
            pause_threshold: float = 2.0,
    ) -> dtypes.tts_result:
        """
        Основная функция для работы с файлом или numpy/torch массивом. 
        Возвращает полную структуру данных:
        1. Полный текст (чистый).
        2. Слова с временными метками.
        3. Предложения с временными метками.

        :param wav_file: Путь к файлу (взаимоисключающе с audio_array)
        :param audio_array: Numpy массив или torch.Tensor с аудио данными [samples] или [channels, samples]
        :param sample_rate: Частота дискретизации для audio_array (обязательно если указан audio_array)
        :param pause_threshold: Порог паузы для формирования предложений (в секундах)
        :return: Результат транскрипции
        :raises ValidationError: Если параметры невалидны
        """
        if wav_file is not None:
            logger.info(f"Starting audio transcription for file: {wav_file}")
            segments, boundaries = self.silero_vad.segment_audio_file(
                wav_file, sr=self.sample_rate
            )
        else:
            audio_tensor = prepare_audio_array(
                audio_array, sample_rate, self.sample_rate
            )
            logger.info(f"Starting audio transcription from array (shape: {audio_tensor.shape})")
            segments, boundaries = self.silero_vad.segment_audio_tensor(
                audio_tensor, sr=self.sample_rate
            )
        
        logger.debug(f"Segmented audio into {len(segments)} segments")

        all_words: List[dtypes.word] = []

        for segment, (start_offset, _) in zip(segments, boundaries):
            segment_words = self._process_segment(segment, offset=start_offset)
            all_words.extend(segment_words)
            logger.debug(f"Processed segment, added {len(segment_words)} words")

        sentences_objs = extract_sentences_from_words(all_words, pause_threshold=pause_threshold)
        logger.debug(f"Extracted {len(sentences_objs)} sentences")

        full_text_str = " ".join([s.text for s in sentences_objs])

        logger.info("Audio transcription completed")

        return dtypes.tts_result(
            full_text=full_text_str,
            words=all_words,
            sentences=sentences_objs
        )

    @overload
    def transcribe_iterative(
            self,
            wav_file: Optional[str] = None,
            audio_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
            sample_rate: Optional[int] = None,
            return_sentences: Literal[False] = False,
            with_progress: Literal[False] = False,
            pause_threshold: float = 2.0,
    ) -> Generator[dtypes.word, None, None]:
        ...

    @overload
    def transcribe_iterative(
            self,
            wav_file: Optional[str] = None,
            audio_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
            sample_rate: Optional[int] = None,
            return_sentences: Literal[True] = True,
            with_progress: Literal[False] = False,
            pause_threshold: float = 2.0,
    ) -> Generator[dtypes.sentence, None, None]:
        ...

    @overload
    def transcribe_iterative(
            self,
            wav_file: Optional[str] = None,
            audio_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
            sample_rate: Optional[int] = None,
            return_sentences: Literal[False] = False,
            with_progress: Literal[True] = True,
            pause_threshold: float = 2.0,
    ) -> Generator[Tuple[dtypes.word, dtypes.Progress], None, None]:
        ...

    @overload
    def transcribe_iterative(
            self,
            wav_file: Optional[str] = None,
            audio_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
            sample_rate: Optional[int] = None,
            return_sentences: Literal[True] = True,
            with_progress: Literal[True] = True,
            pause_threshold: float = 2.0,
    ) -> Generator[Tuple[dtypes.sentence, dtypes.Progress], None, None]:
        ...

    def transcribe_iterative(
            self,
            wav_file: Optional[str] = None,
            audio_array: Optional[Union[np.ndarray, torch.Tensor]] = None,
            sample_rate: Optional[int] = None,
            return_sentences: bool = False,
            with_progress: bool = False,
            pause_threshold: float = 2.0,
    ) -> Generator[Union[dtypes.word, dtypes.sentence, Tuple[Union[dtypes.word, dtypes.sentence], dtypes.Progress]], None, None]:
        """
        Генератор для потокового вывода.

        :param wav_file: Путь к аудиофайлу (взаимоисключающе с audio_array).
        :param audio_array: Numpy массив или torch.Tensor с аудио данными.
        :param sample_rate: Частота дискретизации для audio_array.
        :param return_sentences: Если True, собирает слова в предложения и возвращает экземпляры предложений.
        :param with_progress: Если True, возвращает (объект Word/Sentence, progress_dict),
                              где progress = {'percent': float, 'segment': int, 'total_segments': int}.
        :param pause_threshold: Порог паузы для формирования предложений (в секундах).
        :return: Экземпляр слова или предложения (или кортеж с progress, если with_progress=True).
        """
        if wav_file is not None:
            logger.info(f"Starting iterative audio transcription for file: {wav_file} (return_sentences={return_sentences}, with_progress={with_progress})")
            segments, boundaries = self.silero_vad.segment_audio_file(
                wav_file, sr=self.sample_rate
            )
        else:
            audio_tensor = prepare_audio_array(
                audio_array, sample_rate, self.sample_rate
            )
            logger.info(f"Starting iterative audio transcription from array (return_sentences={return_sentences}, with_progress={with_progress})")
            segments, boundaries = self.silero_vad.segment_audio_tensor(
                audio_tensor, sr=self.sample_rate
            )
        logger.debug(f"Segmented audio into {len(segments)} segments")

        total_segments = len(segments)
        total_audio_duration = sum(segment.shape[0] for segment in segments) / self.sample_rate
        logger.debug(f"Total audio duration: {total_audio_duration} seconds")

        current_segment = 0
        buffer_words: List[dtypes.word] = []
        estimated_remaining_time = 0.0
        if with_progress:
            self.calculated_remaining_time.load_info(self.sample_rate, total_audio_duration)
            logger.debug("Progress calculation initialized")

        for segment, (start_offset, _) in zip(segments, boundaries):
            current_segment += 1
            logger.debug(f"Processing segment {current_segment}/{total_segments}")

            if with_progress:
                estimated_remaining_time = self.calculated_remaining_time.step()
                logger.debug(f"Estimated remaining time after step: {estimated_remaining_time}")

            segment_words = self._process_segment(segment, offset=start_offset)

            if with_progress:
                estimated_remaining_time = self.calculated_remaining_time.calc(segment.shape[0])
                logger.debug(f"Estimated remaining time after calc: {estimated_remaining_time}")

            if return_sentences:
                buffer_words.extend(segment_words)
                sentences = extract_sentences_from_words(buffer_words, pause_threshold=pause_threshold)
                logger.debug(f"Extracted {len(sentences)} sentences from buffer")

                for sent in sentences[:-1] if sentences else []:
                    if with_progress:
                        progress = dtypes.progress(
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
                    logger.debug(f"Updated buffer_words to last {len(buffer_words)} words")
            else:
                for word in segment_words:
                    if with_progress:
                        progress = dtypes.progress(
                            percent=round((current_segment / total_segments) * 100, 2),
                            segment=current_segment,
                            total_segments=total_segments,
                            time_remaining=round(estimated_remaining_time, 2)
                        )
                        yield word, progress
                    else:
                        yield word

        if return_sentences and buffer_words:
            remaining_sentences = extract_sentences_from_words(buffer_words, pause_threshold=pause_threshold)
            logger.debug(f"Processing remaining {len(remaining_sentences)} sentences")
            for sent in remaining_sentences:
                if with_progress:
                    progress = dtypes.progress(
                        percent=100.0,
                        segment=total_segments,
                        total_segments=total_segments,
                        time_remaining=0.0
                    )
                    yield sent, progress
                else:
                    yield sent

        logger.info("Iterative audio transcription completed")

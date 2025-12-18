import logging
from typing import List

from ..core import dtypes

logger = logging.getLogger(__name__)


def clean_and_validate_word(text: str) -> str | None:
    """
    Убирает тире и мусор. Возвращает None, если слово невалидно.
    """
    clean = text.lstrip("—").strip()
    if not clean or clean == "—":
        return None
    return clean


def extract_words_from_tokens(
        pieces: List,
        token_frames: List,
        frame_duration: float,
        offset: float
) -> List[dtypes.word]:
    """
    Извлечение слов из токенов.
    """
    words: List[dtypes.word] = []

    current_word_str = ""
    word_start_sec = 0.0
    word_end_sec = 0.0

    for piece, frame_idx in zip(pieces, token_frames):
        timestamp = frame_idx * frame_duration

        if piece.startswith("▁"):
            cleaned_word = clean_and_validate_word(current_word_str)
            if cleaned_word:
                words.append(dtypes.word(
                    text=cleaned_word,
                    start=round(word_start_sec + offset, 3),
                    end=round(word_end_sec + offset, 3)
                ))

            current_word_str = piece[1:]
            word_start_sec = timestamp
            word_end_sec = timestamp + frame_duration
        else:
            current_word_str += piece
            word_end_sec = timestamp + frame_duration

    cleaned_word = clean_and_validate_word(current_word_str)
    if cleaned_word:
        words.append(dtypes.word(
            text=cleaned_word,
            start=round(word_start_sec + offset, 3),
            end=round(word_end_sec + offset, 3)
        ))
    return words


def extract_sentences_from_words(
        word_data: List[dtypes.word],
        pause_threshold: float = 2.0
) -> List[dtypes.sentence]:
    """
    Создание предложений из слов.
    """
    logger.debug(f"Extract sentences from words")
    end_marks = ['.', '!', '?', '...']
    sentences: List[dtypes.sentence] = []
    current_sentence: List[dtypes.word] = []

    for word in word_data:
        if not current_sentence:
            current_sentence.append(word)
            continue

        prev_end = current_sentence[-1].end
        current_start = word.start
        pause_duration = current_start - prev_end

        is_end_of_sentence = (
                pause_duration > pause_threshold or
                any(current_sentence[-1].text.endswith(mark) for mark in end_marks)
        )

        if is_end_of_sentence:
            s_text = ' '.join(w.text for w in current_sentence).strip()

            sentences.append(dtypes.sentence(
                text=s_text,
                start=current_sentence[0].start,
                end=current_sentence[-1].end
            ))
            current_sentence = [word]
        else:
            current_sentence.append(word)

    if current_sentence:
        s_text = ' '.join(w.text for w in current_sentence).strip()
        sentences.append(dtypes.sentence(
            text=s_text,
            start=current_sentence[0].start,
            end=current_sentence[-1].end
        ))

    logger.debug(f"Extract sentences from words : complete")
    return sentences

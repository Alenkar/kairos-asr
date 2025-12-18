from dataclasses import dataclass
from typing import List, Dict#, Type


@dataclass
class Progress:
    """
    Структура для состояния обработки
    """
    percent: float
    segment: int
    total_segments: int
    time_remaining: float

    def to_dict(self) -> Dict:
        return {
            "percent": self.percent,
            "segment": self.segment,
            "total_segments": self.total_segments,
            "time_remaining": self.time_remaining
        }

@dataclass
class Word:
    """
    Структура для слов
    """
    text: str
    start: float
    end: float

    def to_dict(self) -> Dict:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass
class Sentence:
    """
    Структура для предложений
    """
    text: str
    start: float
    end: float

    def to_dict(self) -> Dict:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass
class TranscriptionResult:
    """
    Структура для общего результата обработки
    """
    full_text: str
    words: List[Word]
    sentences: List[Sentence]


# class DataTypes:
#
#     @property
#     def word(self) -> Type[Word]:
#         return Word
#
#     @property
#     def sentence(self) -> Type[Sentence]:
#         return Sentence
#
#     @property
#     def tts_result(self) -> Type[TranscriptionResult]:
#         return TranscriptionResult
#
#     @property
#     def progress(self) -> Type[Progress]:
#         return Progress


word = Word
sentence = Sentence
tts_result = TranscriptionResult
progress = Progress

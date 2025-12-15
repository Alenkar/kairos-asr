from dataclasses import dataclass
from typing import List


@dataclass
class Progress:
    """
    Структура для состояния обработки
    """
    percent: float
    segment: int
    total_segments: int
    time_remaining: float

    def to_dict(self):
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

    def to_dict(self):
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass
class Sentence:
    """
    Структура для предложений
    """
    text: str
    start: float
    end: float

    def to_dict(self):
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass
class TranscriptionResult:
    """
    Структура для общего результата обработки
    """
    full_text: str
    words: List[Word]
    sentences: List[Sentence]


class DataTypes:
    @property
    def word(self):
        return Word

    @property
    def sentence(self):
        return Sentence

    @property
    def tts_result(self):
        return TranscriptionResult

    @property
    def progres(self):
        return Progress


dtypes = DataTypes()

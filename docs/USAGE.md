# Kairos ASR — расширенное использование

## Содержание
- [Быстрый старт](#быстрый-старт)
- [Подготовка весов](#подготовка-весов-моделей)
- [Требования к аудио](#требования-к-аудио-wav_file)
- [Обычная транскрипция](#обычная-транскрипция)
- [Итеративная обработка](#итеративная-обработка)
- [Прогресс и метаданные](#прогресс-и-метаданные)
- [CLI режим](#cli-режим)

## Быстрый старт
```python
from kairos_asr import KairosASR

asr = KairosASR()  # авто-загрузка весов
print(asr.transcribe("audio.wav").full_text)
```

## Подготовка весов моделей

**Kairos ASR** использует предобученные веса, происходящие 
из проекта **GigaAM**, которые были конвертированы
и оптимизированы для **ONNX**-инференса.

По умолчанию веса загружаются автоматически. При 
необходимости можно указать путь вручную:
```python
from kairos_asr import KairosASR

asr = KairosASR(model_path="<ваш путь до весов>")
```

Автозагрузка:
```python
from kairos_asr import KairosASR
asr = KairosASR()
```

Явный контроль загрузок через утилиту `ModelDownloader`:
```python
from kairos_asr.models.utils.model_downloader import ModelDownloader

downloader = ModelDownloader()  # также можно указать model_path="<ваш путь до весов>"

# Скачать все модели
paths = downloader.download_all()

# Скачать конкретную модель
encoder_path = downloader.download_file("encoder")

# Получить путь до конкретной модели
encoder_path = downloader.check_local_file("encoder")

```

## Требования к аудио (`wav_file`)
- Поддерживаются форматы, которые умеет `ffmpeg`. Файл автоматически ресемплируется до 16 kHz.
- Рекомендуемый вход — WAV PCM 16-bit, mono, 16 kHz (стерео будет приведено к моно).
- Длинные записи обрабатываются: Silero VAD режет на сегменты ~15–25 с (жёсткий лимит ~30 с), затем объединяются.

## Обычная транскрипция
```python
result = asr.transcribe(wav_file="audio.wav")
print(result.full_text)
print(result.words)
print(result.sentences)
```

### Структура объектов
```python
from typing import List

class Word:
    text: str
    start: float
    end: float

class Sentence:
    text: str
    start: float
    end: float

class TranscriptionResult:
    full_text: str
    words: List[Word]
    sentences: List[Sentence]
```

## Итеративная обработка
Итеративный режим нужен, если требуется выводить результаты по мере готовности сегментов.

### Word:
```python
for item in asr.transcribe_iterative(wav_file="example.wav"):
    print(f"{item.text} ({item.start} - {item.end})")
```

### Sentence (сборка слов в предложения на лету):
```python
for item in asr.transcribe_iterative(wav_file="example.wav", return_sentences=True):
    print(f"{item.text} ({item.start} - {item.end})")
```

## Прогресс и метаданные
Режим с прогрессом возвращает кортеж `(item, progress)`, где `progress` содержит ``[процент, номер сегмента, оценку оставшегося времени]``.
```python
for item, progress in asr.transcribe_iterative(
    wav_file="example.wav", return_sentences=False, with_progress=True
):
    print(f"{item.text} | {progress.percent}% "
          f"({progress.segment}/{progress.total_segments}), "
          f"ETA: {progress.time_remaining}s")
```

## CLI режим

Быстрый запуск:
```bash
kairos-asr transcribe path/to/audio.wav
```

Команды:
- `kairos-asr doctor` — проверка окружения (Python, Torch, CUDA, Onnxruntime, Models dir).
- `kairos-asr list` — показывает локальное наличие весов и пути.
- `kairos-asr download [model]` — скачивает все веса или только указанные (`all|encoder|decoder|joint|tokenizer`).
- `kairos-asr transcribe <wav_file>` — транскрипция файла.

Полезные опции:
- `--device cpu|cuda` — выбрать устройство.
- `--progress` — печатать прогресс при транскрипции (если реализовано в CLI).
- `--sentences` — печатать текст в виде отдельных предложений. 

Пример с явным устройством и прогрессом:
```bash
kairos-asr transcribe example.wav --device cpu --progress
```

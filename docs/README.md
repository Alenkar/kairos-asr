 # Kairos Automatic Speech Recognition

## 📄 Описание

**Kairos ASR** — высокопроизводительная библиотека распознавания русской речи на базе [GigaAM-style RNN-T](https://github.com/salute-developers/GigaAM) и **ONNX**. Фокус: скорость, точность и простая интеграция в микросервисы и десктопы.

Основные возможности:
- Оптимизированный ONNX-инференс
- **CPU**, **GPU (CUDA, extra `[gpu]`)** и **Metal (MPS, extra `[metal]`)**
- Временные метки (**word-level**, **sentence-level**)
- Итеративная обработка с прогрессом и ETA
- Встроенный **Voice-Activity-Detection (VAD)**
- Поддержка длинных аудио
- Поддержка **Windows**, **Linux** и **macOS**

## ⚡ Быстрый старт

```bash
pip install kairos-asr[cpu]
# для GPU (Windows/Linux): pip install kairos-asr[gpu]
# для macOS (Metal/MPS): pip install kairos-asr[metal]
```

Если нужна конкретная сборка Torch под вашу CUDA:

```bash
# пример под CUDA 12.1/12.2 (cu121)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --upgrade
```

## 🚀 Использование (Python)

Минимальный пример:

```python
from kairos_asr import KairosASR

asr = KairosASR()  # авто-загрузка весов с HF
result = asr.transcribe(wav_file="audio.wav")
print(result.full_text)
```

Требования к аудио:
- Любые форматы, поддерживаемые `ffmpeg`; ресемплинг до 16 kHz.
- WAV PCM 16-bit mono (рекомендуется); стерео приводится к моно.
- Длинные записи режутся Silero VAD на ~15–25 c (жёсткий лимит ~30 c) и объединяются.

## 🖥️ Использование (CLI)

Установите пакет, затем:

```bash
# Проверить окружение
kairos-asr doctor

# Список локальных/доступных моделей
kairos-asr list

# Скачать все модели заранее
kairos-asr download

# Перевести файл в текст
kairos-asr transcribe <wav_file>
```

## ⚙️ Системные требования
- `ffmpeg` должен быть доступен в `PATH` (загрузка и ресемплинг аудио).
- Доступ в интернет. При первом запуске скачиваются веса моделей. Для ускорения и избежания лимитов задайте `HF_TOKEN` (`huggingface-cli login` или экспорт переменной окружения).

 ![image](assets/logo.png)
 
 # Kairos Automatic Speech Recognition (Hugging Face)
 
 [GitHub: Alenkar/kairos-asr](https://github.com/Alenkar/kairos-asr)

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

## 📦 Файлы в этом репозитории HF

Модельный репозиторий `Alenkar/KairosASR` содержит:
- `kairos_asr_encoder.onnx`
- `kairos_asr_decoder.onnx`
- `kairos_asr_joint.onnx`
- `kairos_asr_tokenizer.model`

`kairos-asr` загружает эти файлы через `huggingface_hub` автоматически. Можно скачать вручную:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download("Alenkar/KairosASR", "kairos_asr_encoder.onnx")
print(path)
```

## ⚡ Быстрый старт

```bash
pip install kairos-asr[cpu]
# для GPU (Windows/Linux): pip install kairos-asr[gpu]
# для macOS (Metal/MPS): pip install kairos-asr[metal]
```

Если нужна конкретная сборка Torch под вашу CUDA:

```bash
# пример под CUDA 12.1/12.2 (cu121)
pip install torch==2.6.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu121 --upgrade
```

## 🚀 Использование (Python)

Минимальный пример:

```python
from kairos_asr import KairosASR

asr = KairosASR()  # device="auto" по умолчанию
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
- Доступ в интернет при первом запуске: скачивание весов моделей. Для ускорения и избежания лимитов задайте `HF_TOKEN` (`huggingface-cli login` или экспорт переменной окружения).

## Больше информации

Для получения дополнительной информации об исходной модели смотрите ее [карточку модели](https://huggingface.co/ai-sage/GigaAM-v3).

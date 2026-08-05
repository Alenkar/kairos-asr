

![image](assets/logo.png)

# Kairos Automatic Speech Recognition

[![PyPI](https://img.shields.io/pypi/v/kairos-asr)](https://pypi.org/project/kairos-asr/)
[![Python](https://img.shields.io/pypi/pyversions/kairos-asr)](https://pypi.org/project/kairos-asr/)
[![Downloads](https://static.pepy.tech/badge/kairos-asr)](https://pepy.tech/project/kairos-asr)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-KairosASR-yellow?logo=huggingface)](https://huggingface.co/Alenkar/KairosASR)


---

**Kairos ASR** — una biblioteca de reconocimiento de voz en ruso de alto rendimiento basada en la arquitectura **RNN-T estilo GigaAM** y **ONNX**.

El proyecto se centra en la velocidad, precisión y facilidad de integración en microservicios y aplicaciones de escritorio.

## 📄 Descripción

---

- Inferencia ONNX optimizada
- Funciona en **CPU**, **GPU (CUDA, extra `[gpu]`)** y **Metal (MPS, extra `[metal]`)**
- Soporte de marcas de tiempo (**nivel palabra**, **nivel oración**)
- Procesamiento iterativo con visualización de progreso y **ETA**
- Detección de actividad de voz (VAD) integrada ([Silero VAD](https://github.com/snakers4/silero-vad))
- Soporte de **audio de larga duración**
- Instalación y uso sencillos
- Compatibilidad con **Windows**, **Linux** y **macOS**
## ⚡ TL;DR
- `pip install kairos-asr[gpu]` (Windows/Linux) o `pip install kairos-asr[metal]` (macOS)
- Ejecutar: `kairos-asr transcribe example.wav` o ver el fragmento de Python a continuación.
- Guía completa: `docs/USAGE.md`.

## 📦 Instalación

Los pesos están disponibles en Hugging Face: [Alenkar/KairosASR](https://huggingface.co/Alenkar/KairosASR)

## 🖥️ Requisitos del sistema
- `ffmpeg` debe estar disponible en `PATH` (se usa para cargar y reamplear audio).
- Conexión a internet. En la primera ejecución se descargan los pesos del modelo.
- Para acelerar las descargas y evitar límites, se recomienda configurar un token de HF: `export HF_TOKEN=...` (o `huggingface-cli login`).

### Inicio rápido (CPU)

```bash
pip install kairos-asr[cpu]
```

### macOS (Metal/MPS)

```bash
pip install kairos-asr[metal]
```

En macOS, la inferencia ONNX se ejecuta en CPU, mientras que la parte de Torch (extracción de características) utiliza MPS si está disponible.

### Soporte GPU (CUDA)

1) Paquete con opciones GPU:
```bash
pip install kairos-asr[gpu]
```

2) Torch/Torchaudio para tu versión de CUDA:
```bash
# ejemplo para CUDA 12.1/12.2 (cu121)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 --upgrade
```
Encuentra tu índice (cu118, cu121, etc.) en pytorch.org y sustitúyelo en el comando.

## 🚀 Uso (Python)

Ejemplo mínimo:
```python
from kairos_asr import KairosASR

asr = KairosASR()  # device="auto" por defecto
result = asr.transcribe(wav_file="audio.wav")
print(result.full_text)
```

Requisitos del audio de entrada:
- Archivos compatibles con `ffmpeg`; reampleo automático a 16 kHz.
- Se recomienda WAV PCM de 16 bits, mono, 16 kHz; el estéreo se convierte a mono.
- Las grabaciones largas se segmentan con Silero VAD en bloques de ~15–25 s (límite máximo ~30 s) y se unen.

## Uso (CLI)
Existen varios comandos para trabajar desde la terminal.

```bash
# Mostrar información
kairos-asr info

# Verificar el entorno y mostrar información
kairos-asr doctor

# Lista de modelos (muestra disponibilidad local y ruta)
kairos-asr list

# Descargar todos los modelos
kairos-asr download

# Descargar solo el encoder
kairos-asr download encoder

# Transcribir archivo a texto
kairos-asr transcribe <wav_file>
```

## 🧠 Cómo funciona

---

1. El VAD divide el audio en segmentos (hasta 25 segundos).
2. Cada segmento se convierte en un espectrograma Mel.
3. Modelo RNN-T (Encoder + Predictor + Joint).
4. Alineación de palabras por frames.
5. Unión de segmentos en la transcripción final.

La arquitectura y el flujo general están inspirados en el proyecto **GigaAM**.

## 📜 Licencia y origen

---

### Código
Parte del código fuente se basa en el proyecto [**GigaAM**](https://github.com/salute-developers/GigaAM)
y se utiliza conforme a los términos de la licencia MIT.
Proyecto original [**GigaAM**](https://github.com/salute-developers/GigaAM):
* License: MIT
* Copyright (c) 2024 GigaChat Team

### Modelos
Los pesos del modelo utilizados provienen del proyecto **GigaAM**, pero han sido:
* convertidos al formato **ONNX**,
* optimizados para **inferencia en CPU/GPU**,
* adaptados para su uso en **Kairos ASR**,
* parcialmente afinados (**Lora**) con datos personalizados.

Los derechos sobre los pesos originales pertenecen a los titulares del proyecto **GigaAM**. 
El uso se realiza bajo los términos de la licencia MIT.

Si utilizas **KairosASR** con fines de investigación o comerciales, verifica la compatibilidad con la licencia
del modelo original (**GigaAM**).

## 📊 Planes de desarrollo

---

* [ ] Inferencia por lotes (batch).
* [ ] Inferencia en tiempo real (micrófono) y fragmentos (chunks) con ndarray.
* [ ] Diarización (separación de hablantes).
* [ ] Pruebas (tests).
* [ ] Entrenamiento adicional con datos especializados.
* [ ] Script para afinación (finetuning) Lora con datos propios.
* [ ] Expansión del vocabulario.
* [x] Paquete para instalación.
* [x] Pesos en Hugging Face.

## 👤 Autor

---

Desarrollado y mantenido por: **([Alexey Shimokhin / Alenkar](https://github.com/Alenkar))**. \
Si utilizas **Kairos ASR** en tus proyectos, se agradece cualquier referencia o comentario.

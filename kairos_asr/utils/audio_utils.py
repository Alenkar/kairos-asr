import logging
import warnings
from subprocess import CalledProcessError, run
import torch
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import butter, lfilter

logger = logging.getLogger(__name__)

def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """
    Loads audio and resamples to specified rate. Supports multi-channel.

    Args:
        audio_path: Path to audio file.
        sample_rate: Target sample rate.

    Returns:
        Tensor [channels, samples].
    """
    logger.debug(f"Load audio file")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        audio = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        raise RuntimeError(f"Failed to load audio: {audio_path}") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        audio_tensor = torch.frombuffer(audio, dtype=torch.int16).float() / 32768.0

    # Handle channels (assume interleaved if multi-channel)
    # ffmpeg with -ac 1 for mono, but if need multi, remove -ac 1 and reshape
    return audio_tensor  # [samples] for mono

# ToDo встроить в pipeline
def peak_normalize(waveform: torch.Tensor, target_db: float = -3.0) -> torch.Tensor:
    """
    Peak-normalizes audio to target dB.

    Args:
        waveform: Audio tensor.
        target_db: Target peak dB.

    Returns:
        Normalized tensor.
    """
    peak = torch.abs(waveform).max()
    if peak == 0:
        return waveform
    target_level = 10 ** (target_db / 20)
    return (waveform / peak) * target_level

# ToDo встроить в pipeline
def normalize_audio(input_file: str, output_file: str = 'normalized_audio.wav') -> str:
    """
    Normalizes and filters audio file.

    Args:
        input_file: Input WAV path.
        output_file: Output path.

    Returns:
        Output file path.
    """
    fs, data = wavfile.read(input_file)
    if data.dtype == 'int16':
        data = data / 32768.0
    else:
        data = data / np.max(np.abs(data))

    # Butterworth lowpass
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    filtered_data = lowpass_filter(data, cutoff=4000, fs=fs)

    # FFT filter (optional, can remove if not needed)
    f = fft(filtered_data)
    freq = np.fft.fftfreq(len(filtered_data), 1 / fs)
    f[np.abs(freq) > 4000] = 0
    filtered_data = np.real(ifft(f))

    wavfile.write(output_file, fs, np.int16(filtered_data * 32767))
    return output_file

# ToDo встроить в pipeline
# def segment_audio_for_embedding(
#         wav_file: str,
#         sr: int,
#         # Параметры Silero
#         vad_model,
#         vad_utils,
#         device: torch.device,
#         # Параметры препроцессинга
#         target_sr: int = 16000,
#         pad_seconds: float = 0.1,  # Добавляем 100мс контекста
#         min_duration_sec: float = 0.5,  # Игнорируем короче 500мс
# ) -> List[torch.Tensor]:
#     # 1. Загрузка (предположим load_audio возвращает Tensor [Channels, Time])
#     audio = load_audio(wav_file)
#
#     # 2. Ресемплинг для VAD и Эмбеддинга (обычно 16k)
#     if sr != target_sr:
#         audio = F.resample(audio, sr, target_sr)
#         sr = target_sr
#
#     # Подготовка моно для VAD
#     if audio.shape[0] > 1:
#         audio_mono = audio.mean(dim=0)
#     else:
#         audio_mono = audio.squeeze()
#
#     # 3. Получение таймстемпов (Silero)
#     get_speech_timestamps = vad_utils[0]
#     speech_timestamps = get_speech_timestamps(
#         audio_mono.to(device),
#         vad_model,
#         sampling_rate=sr,
#         min_speech_duration_ms=int(min_duration_sec * 1000)
#     )
#
#     embedding_chunks = []
#
#     total_samples = audio_mono.shape[0]
#     pad_samples = int(pad_seconds * sr)
#
#     for seg in speech_timestamps:
#         start_sample = seg['start']
#         end_sample = seg['end']
#
#         # --- PREPROCESSING STEP 1: PADDING ---
#         # Расширяем границы, но не выходим за пределы файла
#         start_padded = max(0, start_sample - pad_samples)
#         end_padded = min(total_samples, end_sample + pad_samples)
#
#         # Вырезаем кусок
#         chunk = audio[:, start_padded:end_padded]  # Сохраняем каналы если есть
#
#         # --- PREPROCESSING STEP 2: DURATION CHECK ---
#         # Проверяем длительность уже ПОСЛЕ паддинга или ДО (зависит от логики)
#         # Лучше проверять "чистую" речь (как сделано в параметрах VAD),
#         # но убедиться, что итоговый чанк не слишком мал.
#         if (end_padded - start_padded) / sr < min_duration_sec:
#             continue
#
#         # --- PREPROCESSING STEP 3: SLIDING WINDOW (Опционально здесь или позже) ---
#         # Если чанк длинный (> 4 сек), его лучше нарезать на окна внутри
#         # Для простоты здесь возвращаем полный чанк, но для продакшена лучше
#         # вернуть список окон: [win1, win2, win3]
#
#         embedding_chunks.append(chunk)
#
#     return embedding_chunks

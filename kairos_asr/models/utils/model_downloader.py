import logging
from pathlib import Path

from typing import Dict, Optional

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import LocalEntryNotFoundError

logger = logging.getLogger(__name__)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)


class ModelDownloader:
    """
    Класс для загрузки и разрешения путей к моделям из Hugging Face Hub.

    Основные задачи:
    - хранение соответствия между логическими именами моделей и именами файлов;
    - попытка найти модели в локальном кеше;
    - при необходимости — скачивание файлов моделей из репозитория HF.
    """

    def __init__(self):
        self.repo_id = "Alenkar/KairosASR"
        self.model_files = {
            "encoder": "kairos_asr_encoder.onnx",
            "decoder": "kairos_asr_decoder.onnx",
            "joint": "kairos_asr_joint.onnx",
            "tokenizer": "kairos_asr_tokenizer.model",
        }

    @staticmethod
    def get_models_dir() -> Path:
        """
        Возвращает директорию кеша Hugging Face, где физически лежат модели.

        :return: Объект `Path` с путем до директории кеша HF.
        """
        return Path(HF_HUB_CACHE)

    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Возвращает путь до локального файла модели, если он уже есть в кеше.

        :param model_name: Логическое имя модели (encoder, decoder, joint, tokenizer).
        :return: Строка с путем до файла или `None`, если файл не найден в кеше.
        """
        logger.debug(f"Trying to resolve local path for model '{model_name}' in cache.")
        filename = self._get_filename(model_name)
        try:
            path = self._download_file(filename, local_files_only=True)
            logger.debug(f"Model '{model_name}' found in local cache at '{path}'.")
            return path
        except LocalEntryNotFoundError:
            logger.debug(f"Model '{model_name}' is not found in local cache.")
            return None

    def _get_filename(self, model_name: str) -> str:
        """
        Возвращает имя файла для переданного логического имени модели.

        :param model_name: Логическое имя модели.
        :return: Имя файла модели в репозитории HF.
        :raises ValueError: Если указанное имя модели неизвестно.
        """
        if model_name not in self.model_files:
            logger.error(f"Unknown model name requested: '{model_name}'")
            raise ValueError(f"Неизвестная модель: {model_name}")

        filename = self.model_files[model_name]
        logger.debug(f"Resolved filename '{filename}' for model '{model_name}'")
        return filename

    def _download_file(
        self,
        filename: str,
        force_download: bool = False,
        local_files_only: bool = False,
    ):
        """
        Внутренний метод для загрузки отдельного файла модели с Hugging Face Hub.

        :param filename: Имя файла в репозитории HF.
        :param force_download: Если `True`, файл будет скачан заново, даже если уже есть в кеше.
        :param local_files_only: Если `True`, попытка будет сделана только через локальный кеш,
                                 без обращения к сети.
        :return: Путь до локального файла модели.
        """
        if local_files_only:
            logger.debug(
                f"Checking local cache for file '{filename}' (force_download={force_download})."
            )
        else:
            logger.debug(
                "Downloading file '%s' from Hugging Face Hub (force_download=%s).",
                filename,
                force_download,
            )

        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="model",
            force_download=force_download,
            local_files_only=local_files_only,
        )
        logger.debug(f"File '{filename}' is available locally at '{path}'.")
        return path

    def download_model(self, model_name: str, force_download: bool = False):
        """
        Скачивает одну модель и возвращает путь до локального файла.

        Если модель уже есть в кеше и `force_download=False`, будет возвращен существующий путь.

        :param model_name: Логическое имя модели.
        :param force_download: Если `True`, принудительно перекачать файл.
        :return: Строка с путем до локального файла модели.
        """
        logger.debug(f"Request to download model '{model_name}' (force_download={force_download}).")
        filename = self._get_filename(model_name)
        path = self._download_file(filename=filename, force_download=force_download)
        logger.debug(f"Model '{model_name}' is ready at '{path}'.")
        return path

    def download_all_models(self):
        """
        Загружает все модели, указанные в `self.model_files`.

        Используется для предварительного скачивания всех необходимых файлов.
        """
        logger.debug("Starting download of all models (force_download=True).")
        for model_name in self.model_files.keys():
            logger.debug("Downloading model '{model_name}'.")
            filename = self._get_filename(model_name)
            self._download_file(filename=filename, force_download=True)
        logger.debug("All models have been downloaded successfully.")

    def resolve_models_path(self, model_paths: Dict[str, Optional[str]], force_download: bool = False):
        """
        Проверяет переданные пользователем пути к моделям и при необходимости загружает недостающие.

        :param model_paths: Словарь вида {логическое_имя_модели: путь_к_файлу_или_None}.
                            Если путь указан, он будет проверен на существование.
                            Если путь не указан (`None` или пустая строка), модель будет
                            загружена автоматически.
        :param force_download: Если `True`, недостающие файлы будут скачаны заново.
        :return: Словарь с разрешёнными путями {логическое_имя_модели: путь_к_файлу}.
        """
        logger.debug("Resolving model paths (force_download=%s).", force_download)
        resolved: Dict[str, str] = {}
        for model_name, user_path in model_paths.items():
            if user_path:
                path_obj = Path(user_path)
                logger.debug(f"User provided path '{user_path}' for model '{model_name}'. Checking existence.")
                if not path_obj.exists():
                    logger.error(f"Provided path for model '{model_name}' does not exist: '{user_path}'.")
                    raise FileNotFoundError(
                        f"Файл модели '{model_name}' не найден: {user_path}"
                    )
                logger.debug(f"Using user provided path for model '{model_name}': '{user_path}'.")
                resolved[model_name] = str(path_obj)
                continue
            else:
                logger.debug("No path provided for model '{model_name}'. Downloading or resolving from cache.")
                resolved_path = self.download_model(model_name, force_download)
                resolved[model_name] = resolved_path
        logger.debug("All model paths have been resolved successfully.")
        return resolved

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
    def __init__(self, model_path: Optional[str] = None):
        """
        Класс для управления весами модели из Hugging Face.

        :param model_path: Пользовательский каталог для хранения весов.
                           Если None, используется кэш HF по умолчанию.
        """
        self.repo_id = "Alenkar/KairosASR"
        self.model_files = {
            "encoder": "kairos_asr_encoder.onnx",
            "decoder": "kairos_asr_decoder.onnx",
            "joint": "kairos_asr_joint.onnx",
            "tokenizer": "kairos_asr_tokenizer.model",
        }
        self.model_path = Path(model_path) if model_path else None
        logger.debug(f"ModelDownloader initialized with model_path: {self.model_path}")

    def get_storage_dir(self) -> Path:
        """
        Возвращает директорию, где хранятся (или будут храниться) веса.

        :return: Абсолютный путь к папке.
        """
        if self.model_path:
            storage_dir = self.model_path.absolute()
        else:
            repo_parts = self.repo_id.replace("/", "--")
            storage_dir = Path(HF_HUB_CACHE) / f"models--{repo_parts}"
        logger.debug(f"Storage directory: {storage_dir}")
        return storage_dir

    def check_local_file(self, file_key: str) -> Optional[str]:
        """
        Проверяет наличие файла на диске без доступа к сети.

        :param file_key: Имя ключ для получения имени модели.
        :return: Путь к файлу или None, если файл не найден.
        """
        filename = self.model_files.get(file_key)
        if not filename:
            logger.warning(f"File key '{file_key}' not found in model files.")
            return None

        if self.model_path:
            target_path = self.model_path / filename
            if target_path.exists():
                logger.debug(f"Local file found: {target_path.absolute()}")
                return str(target_path.absolute())
            else:
                logger.debug(f"Local file not found: {target_path.absolute()}")
                return None
        else:
            try:
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    local_files_only=True,
                )
                logger.debug(f"Local file found: {Path(path).absolute()}")
                return str(Path(path).absolute())
            except (LocalEntryNotFoundError, Exception) as e:
                logger.debug(f"Local file not found for '{filename}': {e}")
                return None

    def download_file(self, file_key: str, force_download: bool = False) -> str:
        """
        Загружает определенный файл весов или возвращает его путь, если он уже существует.

        :param file_key: Ключ из self.model_files (например, 'encoder').
        :param force_download: Принудительная загрузка.
        :return: Абсолютный путь к файлу.
        """
        if file_key not in self.model_files:
            logger.error(f"File key '{file_key}' not found in model files.")
            raise ValueError(f"File key '{file_key}' not found in model files.")

        filename = self.model_files[file_key]
        logger.info(f"Starting download for file: {filename} (force_download={force_download})")

        try:
            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                local_dir=self.model_path,
                force_download=force_download,
            )
            logger.debug(f"File {filename} ready: {path}")
            return str(Path(path).absolute())
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            raise

    def download_all(self, force_download: bool = False) -> Dict[str, Optional[str]]:
        """
        Загружает все файлы весов, указанные в конфигурации.

        :param force_download: Принудительная загрузка.
        :return: Словарь {logical_name: absolute_path}.
        """
        storage_dir = self.get_storage_dir()
        logger.debug(f"Starting download of all weights to: {storage_dir}")
        resolved_paths = {}
        for key in self.model_files.keys():
            try:
                resolved_paths[key] = self.download_file(key, force_download=force_download)
                logger.info(f"Successfully downloaded {key}")
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")
        return resolved_paths

    def resolve_models_path(self, force_download: bool = False) -> Dict[str, Optional[str]]:
        """
        Получает или загружает модели.

        :param force_download: Принудительная загрузка.
        :return: Dict с путями до файлов моделей.
        """
        logger.debug(f"Resolving model paths (force_download={force_download})")
        resolved_paths = {}
        for key in self.model_files.keys():
            local_path = self.check_local_file(key)
            if force_download or local_path is None:
                logger.info(f"Downloading {key} due to force_download or not found locally")
                local_path = self.download_file(key, force_download=force_download)
            else:
                logger.debug(f"Using local path for {key}: {local_path}")

            resolved_paths[key] = local_path
        return resolved_paths

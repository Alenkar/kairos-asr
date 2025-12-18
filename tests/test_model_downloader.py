from pathlib import Path

import pytest
from unittest.mock import patch

from huggingface_hub.errors import LocalEntryNotFoundError
from kairos_asr.models.utils.model_downloader import ModelDownloader


REPO_ID = "Alenkar/KairosASR"
MODEL_FILES = {
    "encoder": "kairos_asr_encoder.onnx",
    "decoder": "kairos_asr_decoder.onnx",
    "joint": "kairos_asr_joint.onnx",
    "tokenizer": "kairos_asr_tokenizer.model",
}


def test_init_default():
    downloader = ModelDownloader()
    assert downloader.model_path is None
    assert downloader.repo_id == REPO_ID
    assert downloader.model_files == MODEL_FILES


def test_init_custom_path():
    custom_path = "/tmp/models"
    downloader = ModelDownloader(model_path=custom_path)
    assert downloader.model_path == Path(custom_path)
    assert downloader.repo_id == REPO_ID
    assert downloader.model_files == MODEL_FILES


@patch("kairos_asr.models.utils.model_downloader.HF_HUB_CACHE", "/default/cache")
def test_get_storage_dir_default():
    downloader = ModelDownloader()
    expected_dir = Path("/default/cache") / "models--Alenkar--KairosASR"
    assert downloader.get_storage_dir() == expected_dir


def test_get_storage_dir_custom():
    custom_path = "/tmp/models"
    downloader = ModelDownloader(model_path=custom_path)
    assert downloader.get_storage_dir() == Path(custom_path).absolute()


def test_check_local_file_custom_exists(tmp_path):
    custom_path = tmp_path / "models"
    custom_path.mkdir()
    file_name = "kairos_asr_encoder.onnx"
    file_path = custom_path / file_name
    file_path.touch()  # Создаём пустой файл

    downloader = ModelDownloader(model_path=str(custom_path))
    result = downloader.check_local_file("encoder")
    assert result == str(file_path.absolute())


def test_check_local_file_custom_not_exists(tmp_path):
    custom_path = tmp_path / "models"
    custom_path.mkdir()

    downloader = ModelDownloader(model_path=str(custom_path))
    result = downloader.check_local_file("encoder")
    assert result is None


@patch("kairos_asr.models.utils.model_downloader.hf_hub_download")
def test_check_local_file_default_exists(mock_hf_download):
    downloader = ModelDownloader()
    mock_hf_download.return_value = "/cache/path/model.onnx"

    result = downloader.check_local_file("encoder")
    assert result == "/cache/path/model.onnx"
    mock_hf_download.assert_called_once_with(
        repo_id=downloader.repo_id,
        filename="kairos_asr_encoder.onnx",
        local_files_only=True,
    )


@patch("kairos_asr.models.utils.model_downloader.hf_hub_download")
def test_check_local_file_default_not_exists(mock_hf_download):
    downloader = ModelDownloader()
    mock_hf_download.side_effect = LocalEntryNotFoundError("Not found")

    assert downloader.check_local_file("encoder") is None


def test_check_local_file_invalid_key():
    downloader = ModelDownloader()
    assert downloader.check_local_file("invalid") is None


@patch("kairos_asr.models.utils.model_downloader.hf_hub_download")
def test_download_file(mock_hf_download):
    downloader = ModelDownloader()
    mock_hf_download.return_value = "/downloaded/path/model.onnx"

    result = downloader.download_file("encoder")
    assert result == "/downloaded/path/model.onnx"
    mock_hf_download.assert_called_once_with(
        repo_id=downloader.repo_id,
        filename="kairos_asr_encoder.onnx",
        local_dir=None,
        force_download=False,
    )


def test_download_file_invalid_key():
    downloader = ModelDownloader()
    with pytest.raises(ValueError):
        downloader.download_file("invalid")


@patch("kairos_asr.models.utils.model_downloader.ModelDownloader.download_file")
def test_download_all(mock_download_file):
    downloader = ModelDownloader()
    mock_download_file.side_effect = [
        "/path/encoder",
        "/path/decoder",
        "/path/joint",
        "/path/tokenizer",
    ]

    result = downloader.download_all()
    assert len(result) == 4
    assert result["encoder"] == "/path/encoder"
    assert mock_download_file.call_count == 4


@patch("kairos_asr.models.utils.model_downloader.ModelDownloader.check_local_file")
@patch("kairos_asr.models.utils.model_downloader.ModelDownloader.download_file")
def test_resolve_models_path(mock_download_file, mock_check_local_file):
    downloader = ModelDownloader()
    mock_check_local_file.side_effect = [None, "/path/decoder", None, "/path/tokenizer"]
    mock_download_file.side_effect = ["/path/encoder", "/path/joint"]

    result = downloader.resolve_models_path()
    assert len(result) == 4
    assert result["encoder"] == "/path/encoder"
    assert result["decoder"] == "/path/decoder"
    assert mock_download_file.call_count == 2


@patch("kairos_asr.models.utils.model_downloader.ModelDownloader.check_local_file")
@patch("kairos_asr.models.utils.model_downloader.ModelDownloader.download_file")
def test_resolve_models_path_force_download(mock_download_file, mock_check_local_file):
    downloader = ModelDownloader()
    mock_check_local_file.side_effect = [
        "/old/encoder",
        "/old/decoder",
        "/old/joint",
        "/old/tokenizer",
    ]
    mock_download_file.side_effect = [
        "/new/encoder",
        "/new/decoder",
        "/new/joint",
        "/new/tokenizer",
    ]

    result = downloader.resolve_models_path(force_download=True)
    assert len(result) == 4
    assert result["encoder"] == "/new/encoder"
    assert mock_download_file.call_count == 4


@patch("kairos_asr.models.utils.model_downloader.ModelDownloader.check_local_file")
@patch("kairos_asr.models.utils.model_downloader.ModelDownloader.download_file")
def test_resolve_models_path_no_force(mock_download_file, mock_check_local_file):
    downloader = ModelDownloader()
    mock_check_local_file.side_effect = [None, "/path/decoder", None, "/path/tokenizer"]
    mock_download_file.side_effect = ["/path/encoder", "/path/joint"]

    result = downloader.resolve_models_path()
    assert result["encoder"] == "/path/encoder"
    assert result["decoder"] == "/path/decoder"
    assert mock_download_file.call_count == 2

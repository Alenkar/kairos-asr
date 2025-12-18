import argparse
import logging
import os
import sys
import platform
from typing import Dict, Any

logger = logging.getLogger(__name__)

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

def setup_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.NOTSET
    logging.basicConfig(level=level)


def doctor_command(args: argparse.Namespace) -> int:
    from ..models.utils.model_downloader import ModelDownloader
    import torch
    import onnxruntime

    downloader = ModelDownloader()
    report: Dict[str, Any] = {"status": "ok", "checks": {}}

    python_ok = sys.version_info >= (3, 10)
    report["checks"]["python"] = {
        "version": platform.python_version(),
        "required": ">=3.10",
        "ok": python_ok,
    }
    if not python_ok:
        report["status"] = "error"

    try:
        cuda_available = torch.cuda.is_available()
        report["checks"]["torch"] = {
            "version": torch.__version__,
            "cuda_available": cuda_available,
        }
    except Exception as e:
        report["checks"]["torch"] = {"error": str(e)}
        report["status"] = "error"

    try:
        providers = onnxruntime.get_available_providers()
        report["checks"]["onnxruntime"] = {
            "version": onnxruntime.__version__,
            "providers": providers,
        }
    except Exception as e:
        report["checks"]["onnxruntime"] = {"error": str(e)}
        report["status"] = "error"

    models_dir = downloader.get_storage_dir()
    report["checks"]["models_dir"] = {
        "path": str(models_dir),
        "exists": models_dir.exists(),
        "writable": os.access(str(models_dir), os.W_OK) if models_dir.exists() else False,
    }

    model_files = downloader.model_files
    models_info = []
    for name in model_files:
        path = downloader.check_local_file(name)
        exists_local = path is not None
        models_info.append(
            {
                "name": name,
                "exists": exists_local,
                "path": str(path) if path else None,
            }
        )
        if not exists_local:
            report["status"] = "warning"

    report["checks"]["models"] = models_info

    print("Kairos-ASR Doctor")
    print("=" * 50)

    for key, value in report["checks"].items():
        print(f"[{key}]")
        if isinstance(value, list):
            for m in value:
                status = "✅" if m["exists"] else "❌"
                print(f"  {status} {m['name']}")
        else:
            for k, v in value.items():
                print(f"  {k}: {v}")
        print()

    print(f"Overall status: {report['status']}")

    return 0 if report["status"] == "ok" else 1


def download_command(args: argparse.Namespace) -> int:
    from ..models.utils.model_downloader import ModelDownloader

    downloader = ModelDownloader()

    all_models = list(downloader.model_files.keys())
    targets = all_models if args.model == "all" else [args.model]

    if args.model != "all" and args.model not in all_models:
        print(f"❌ Неизвестная модель: {args.model}. Доступные: {', '.join(all_models)}", file=sys.stderr)
        return 1

    results: Dict[str, Any] = {}

    for model_name in targets:
        try:
            path = downloader.download_file(model_name, force_download=args.force)
            results[model_name] = {"status": "success", "path": str(path)}
            print(f"✅ {model_name}")
        except Exception as e:
            results[model_name] = {"status": "error", "error": str(e)}
            print(f"❌ {model_name}: {e}", file=sys.stderr)

    return 0 if all(v["status"] == "success" for v in results.values()) else 1


def list_command(args: argparse.Namespace) -> int:
    from ..models.utils.model_downloader import ModelDownloader

    downloader = ModelDownloader()

    model_files = downloader.model_files
    result = []
    for name in model_files:
        path = downloader.check_local_file(name)
        exists_local = path is not None
        result.append(
            {
                "name": name,
                "exists": exists_local,
                "path": str(path) if path else None,
            }
        )

    print("Модели Kairos-ASR")
    print("=" * 50)
    for m in result:
        symbol = "✅" if m["exists"] else "❌"
        print(f"{symbol} {m['name']}")

    return 0


def transcribe_command(args: argparse.Namespace) -> int:
    from ..core.asr import KairosASR
    asr = KairosASR(device=args.device)

    if args.progress:
        for item in asr.transcribe_iterative(
            args.audio,
            return_sentences=args.sentences,
            with_progress=True,
        ):
            obj, progress = item
            print(obj.text)
            logger.info(f"{progress.percent}% | {progress.time_remaining} s")
    else:
        result = asr.transcribe(args.audio)
        if args.sentences:
            for sentence in result.sentences:
                print(sentence.text)
        else:
            print(result.full_text)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kairos-ASR — русское распознавание речи",
    )
    subparsers = parser.add_subparsers(dest="command")

    # download
    p_download = subparsers.add_parser("download", help="Скачать модели")
    p_download.add_argument("model", nargs="?", default="all")
    p_download.add_argument("--force", "-f", action="store_true")
    p_download.add_argument("--verbose", "-v", action="store_true")

    # list
    p_list = subparsers.add_parser("list", help="Список моделей")
    p_list.add_argument("--verbose", "-v", action="store_true")

    # doctor
    p_doctor = subparsers.add_parser("doctor", help="Диагностика окружения")
    p_doctor.add_argument("--verbose", "-v", action="store_true")

    # transcribe
    p_transcribe = subparsers.add_parser("transcribe", help="Распознать аудио")
    p_transcribe.add_argument("audio")
    p_transcribe.add_argument("--device", default="cuda")
    p_transcribe.add_argument("--sentences", action="store_true")
    p_transcribe.add_argument("--progress", action="store_true")
    p_transcribe.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    if args.command == "download":
        return download_command(args)
    if args.command == "list":
        return list_command(args)
    if args.command == "doctor":
        return doctor_command(args)
    if args.command == "transcribe":
        return transcribe_command(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

import logging
from pathlib import Path
from typing import List

import onnxruntime as ort

logger = logging.getLogger(__name__)

class ONNXModel:
    def __init__(
            self,
            model_path: str,
            device: str = "cuda",
        ):
        """
        Обертка для ONNX моделей.
        :param model_path: Путь к файлу или имя модели из реестра
        :param device: 'cuda' или 'cpu'
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 16
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.log_severity_level = 3

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device.lower() and "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(model_path.resolve(), sess_options=opts, providers=providers)
        logger.debug(f"Model loaded {model_path.name} : {self.session.get_providers()[0]}")

    def run(self, output_names, input_dict):
        """
        Запуск inference.
        :param output_names:
        :param input_dict:
        :return:
        """
        return self.session.run(output_names, input_dict)

    def input_names(self) -> List[str]:
        """
        Получение имен слоёв input.
        :return:
        """
        return [i.name for i in self.session.get_inputs()]

    def output_names(self) -> List[str]:
        """
        Получение имен слоёв output.
        :return:
        """
        return [o.name for o in self.session.get_outputs()]

    def get_input_dict(self, data_list):
        """
        Создание входного словаря.
        :param data_list:
        :return:
        """
        return {
            node.name: data
            for (node, data) in zip(self.session.get_inputs(), data_list)
        }

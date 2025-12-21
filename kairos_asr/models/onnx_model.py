import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any

import onnxruntime as ort

logger = logging.getLogger(__name__)

class ONNXModel:
    def __init__(
            self,
            model_path: str,
            device: str = "cuda",
            optimize: bool = True,
            gpu_mem_limit: Optional[int] = None,
            enable_tunable_ops: bool = False,
        ):
        """
        Обертка для ONNX моделей с оптимизациями.

        :param model_path: Путь к файлу или имя модели из реестра
        :param device: 'cuda' или 'cpu'
        :param optimize: Включить оптимизацию графа (по умолчанию True)
        :param gpu_mem_limit: Лимит памяти GPU в байтах (None = без лимита)
        :param enable_tunable_ops: Включить tunable operators для автоматической оптимизации (по умолчанию False, т.к. может замедлять первый запуск)
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        opts = ort.SessionOptions()
        
        if "cpu" in device.lower():
            opts.intra_op_num_threads = min(16, os.cpu_count() or 1)
            opts.inter_op_num_threads = 1
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.enable_mem_pattern = True
            opts.enable_cpu_mem_arena = True
        else:
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
        
        if optimize:
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        opts.log_severity_level = 3

        providers = self._get_providers(device)
        provider_options = self._get_provider_options(device, gpu_mem_limit, enable_tunable_ops, providers)
        
        session_kwargs = {
            "sess_options": opts,
            "providers": providers,
        }
        if provider_options:
            session_kwargs["provider_options"] = provider_options
        
        self.session = ort.InferenceSession(
            model_path.resolve(), 
            **session_kwargs
        )
        
        self.use_io_binding = (
            "cuda" in device.lower() and 
            "CUDAExecutionProvider" in providers
        )
        self.device = device.lower()
        
        self._cached_input_names = [node.name for node in self.session.get_inputs()]
        self._cached_output_names = [node.name for node in self.session.get_outputs()]
        
        logger.debug(
            f"Model loaded {model_path.name} : {self.session.get_providers()[0]} "
            f"(optimize={optimize}, io_binding={self.use_io_binding}, "
            f"tunable_ops={enable_tunable_ops})"
        )

    @staticmethod
    def _get_providers(device: str) -> List[str]:
        """Получить список провайдеров для устройства"""
        available = ort.get_available_providers()
        
        if "cuda" in device.lower() and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "tensorrt" in device.lower() and "TensorrtExecutionProvider" in available:
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
    
    @staticmethod
    def _get_provider_options(
        device: str, 
        gpu_mem_limit: Optional[int], 
        enable_tunable_ops: bool,
        providers: List[str]
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Получить опции провайдеров для оптимизации GPU.
        
        :param device: Устройство ('cuda' или 'cpu')
        :param gpu_mem_limit: Лимит памяти GPU в байтах
        :param enable_tunable_ops: Включить tunable operators
        :param providers: Список провайдеров
        :return: Список словарей опций для каждого провайдера или None
        """
        if "cuda" not in device.lower():
            return None
        
        provider_options_list = []
        
        for provider in providers:
            if provider == "CUDAExecutionProvider":
                cuda_options: Dict[str, Any] = {}
                
                cuda_options["arena_extend_strategy"] = "kNextPowerOfTwo"
                
                if gpu_mem_limit is not None:
                    cuda_options["gpu_mem_limit"] = str(gpu_mem_limit)
                
                if enable_tunable_ops:
                    cuda_options["tunable_op_enable"] = "1"
                    cuda_options["tunable_op_tuning_enable"] = "1"
                
                provider_options_list.append(cuda_options)
            else:
                provider_options_list.append({})
        
        return provider_options_list if provider_options_list else None

    def run(self, input_dict):
        """
        Запуск inference с оптимизацией через IO Binding для GPU.

        :param input_dict: Словарь входных данных.
        :return: Список выходных numpy-массивов.
        """
        if self.use_io_binding:
            try:
                io_binding = self.session.io_binding()
                
                for name, value in input_dict.items():
                    io_binding.bind_input(name, value)
                
                for name in self._cached_output_names:
                    io_binding.bind_output(name)
                
                self.session.run_with_iobinding(io_binding)
                return io_binding.copy_outputs_to_cpu()
            except Exception as e:
                logger.debug(f"IO Binding failed, using standard run: {e}")
                return self.session.run(self._cached_output_names, input_dict)
        else:
            return self.session.run(self._cached_output_names, input_dict)

    def input_names(self) -> List[str]:
        """
        Получение имен слоёв input.

        :return: Список имён входных узлов модели.
        """
        return self._cached_input_names.copy()

    def output_names(self) -> List[str]:
        """
        Получение имен слоёв output.

        :return: Список имён выходных узлов модели.
        """
        return self._cached_output_names.copy()

    def get_input_dict(self, data_list):
        """
        Создание входного словаря.

        :param data_list: Список numpy-массивов.
        :return: Dict[name, data].
        """
        return {
            self._cached_input_names[i]: data
            for i, data in enumerate(data_list)
        }

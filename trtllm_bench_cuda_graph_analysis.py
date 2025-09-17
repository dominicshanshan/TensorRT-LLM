#!/usr/bin/env python3
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import yaml


class CudaGraphBenchmark:

    def __init__(self):
        self.llm_models_root = os.environ.get(
            'LLM_MODELS_ROOT', '/scratch.trt_llm_data/llm-models')
        self.llm_root = os.environ.get('LLM_ROOT',
                                       '/scratch_gpu/fork/TensorRT-LLM')
        self.model_path = os.environ.get(
            'MODEL_PATH',
            '/scratch.trt_llm_data/llm-models/DeepSeek-V3-Lite/fp8')
        # self.model_path = "/tmp/DeepSeek-V3-Lite/fp8"
        self.model_name = "deepseek_v3_lite_fp8_hf"
        # Specify the output directory for different experiments
        # E.g., _gh200, _h200, etc.
        self.output_dir = Path(
            os.environ.get(
                'OUTPUT_DIR',
                '/scratch_gpu/fork/TensorRT-LLM/cuda_graph_testing_logs_gh200'))
        self.data_gen_path = Path(
            os.environ.get('DATA_GEN_PATH',
                           '/scratch_gpu/fork/TensorRT-LLM/benchmarks/cpp'))

        self.batch_sizes = [
            1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 33, 48, 49, 64, 65, 96, 97, 128,
            160, 192, 200, 256, 320, 384, 390, 396, 400, 412, 416, 420, 500,
            512, 528, 576, 608, 640, 672, 704, 736, 768, 900, 1024, 1032, 1040,
            1280, 1408, 1536, 1792, 1920, 2000
        ]
        # self.batch_sizes = [1, 2, 4, 8, 16, 24, 32, 33, 48, 49]
        # self.batch_sizes = [
        #     128, 160, 192, 200, 256, 320, 384, 390, 396, 400, 412, 416, 420,
        #     500, 512, 528, 576, 608, 640, 672, 704, 736, 768, 900, 1024, 1032,
        #     1040, 1280, 1408, 1536, 1792, 1920, 2000
        # ]
        self.input_length = int(os.environ.get('INPUT_LENGTH', 500))
        self.output_length = int(os.environ.get('OUTPUT_LENGTH', 2000))
        self.num_requests = int(os.environ.get('NUM_REQUESTS', 2048))
        self.max_batch_size = int(os.environ.get('MAX_BATCH_SIZE', 2048))
        self.max_num_tokens = int(os.environ.get('MAX_NUM_TOKENS', 8192))
        self.tp_size = int(os.environ.get('TP_SIZE', 1))
        self.pp_size = int(os.environ.get('PP_SIZE', 1))

        self.gpu_id = int(os.environ.get('GPU_ID', 0))
        self.monitor_interval = int(os.environ.get('MONITOR_INTERVAL', 1))
        self.monitor_process: Optional[subprocess.Popen] = None

        self.setup_logging()

        # SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, self.signal_handler)
        # SIGTERM - terminate the process gracefully
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                            format='[%(asctime)s] %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def signal_handler(self, signum, frame):
        self.logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        self.logger.info("Cleaning up...")
        if self.monitor_process and self.monitor_process.poll() is None:
            self.logger.info(
                f"Stopping GPU monitoring (PID: {self.monitor_process.pid})")
            self.monitor_process.terminate()
            try:
                self.monitor_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.monitor_process.kill()
                self.monitor_process.wait()
            self.monitor_process = None

    def create_directories(self):
        self.logger.info("Creating output directories...")
        directories = ['configs', 'logs', 'reports', 'gpu_logs']
        for directory in directories:
            (self.output_dir / directory).mkdir(parents=True, exist_ok=True)

    def create_cuda_graph_configs(self):
        self.logger.info("Creating CUDA graph configuration files...")

        # Config 1: Default padding enabled
        config_default = {
            'print_iter_log': False,
            'cuda_graph_config': {
                'enable_padding': True,
                'max_batch_size': self.max_batch_size,
                # Default batch sizes with padding: [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
                # > 128 it will add powers of 2 up to max_batch_size, e.g., (128, 256, 512, 1024, 2048)
            },
            'kv_cache_config': {
                'dtype': 'auto',
                'free_gpu_memory_fraction': 0.9
            },
            'enable_chunked_prefill': True
        }

        with open(self.output_dir / 'configs' / 'padding_enabled_default.yaml',
                  'w') as f:
            yaml.dump(config_default, f, default_flow_style=False)

        # Config 2: Padding disabled (more comprehensive batch sizes)
        config_disabled = {
            'print_iter_log': False,
            'cuda_graph_config': {
                'enable_padding': False,
                'max_batch_size': self.max_batch_size,
                # Default batch sizes without padding: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 64, 128]
                # > 128 it will add powers of 2 up to max_batch_size, e.g., (128, 256, 512, 1024, 2048)
            },
            'kv_cache_config': {
                'dtype': 'auto',
                'free_gpu_memory_fraction': 0.9
            },
            'enable_chunked_prefill': True
        }

        # Config 3: Padding with slide size 64
        config_slide_64 = {
            'print_iter_log': False,
            'cuda_graph_config': {
                'enable_padding':
                True,
                'batch_sizes': [
                    128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832,
                    896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472,
                    1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048
                ]
            },
            'kv_cache_config': {
                'dtype': 'auto',
                'free_gpu_memory_fraction': 0.9
            },
            'enable_chunked_prefill': True
        }

        with open(self.output_dir / 'configs' / 'padding_disabled.yaml',
                  'w') as f:
            yaml.dump(config_disabled, f, default_flow_style=False)

        with open(self.output_dir / 'configs' / 'padding_slide_64.yaml',
                  'w') as f:
            yaml.dump(config_slide_64, f, default_flow_style=False)

        self.logger.info("Created CUDA graph configuration files:")
        config_files = list((self.output_dir / 'configs').glob('*.yaml'))
        for config_file in config_files:
            self.logger.info(f"  {config_file}")

    def generate_dataset(self) -> Path:
        dataset_path = self.output_dir / f"dataset_{self.input_length}_{self.output_length}_{self.num_requests}.txt"
        if not dataset_path.exists():
            self.logger.info(
                f"Generating dataset: ISL={self.input_length}, OSL={self.output_length}, requests={self.num_requests}"
            )

            cmd = [
                'python',
                str(self.data_gen_path / 'prepare_dataset.py'), '--stdout',
                '--tokenizer',
                str(self.model_path), 'token-norm-dist', '--num-requests',
                str(self.num_requests), '--input-mean',
                str(self.input_length), '--input-stdev', '0', '--output-mean',
                str(self.output_length), '--output-stdev', '0'
            ]

            with open(dataset_path, 'w') as f:
                result = subprocess.run(cmd,
                                        stdout=f,
                                        stderr=subprocess.PIPE,
                                        text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Dataset generation failed: {result.stderr}")

            self.logger.info(f"Dataset generated: {dataset_path}")
        else:
            self.logger.info(f"Using existing dataset: {dataset_path}")

        return dataset_path

    def start_gpu_monitoring(self, config_name: str, batch_size: int):
        gpu_log_path = self.output_dir / 'gpu_logs' / f'gpu_monitor_{config_name}_bs{batch_size}.log'

        self.logger.info(
            f"Starting GPU monitoring for {config_name}, batch size {batch_size}"
        )
        self.logger.info(f"GPU log: {gpu_log_path}")

        # nvidia-smi dmon: -s um (utilization,memory), -i (GPU ID), -o T (timestamp), -f (file)
        cmd = [
            'nvidia-smi', 'dmon', '-s', 'um', '-i',
            str(self.gpu_id), '-o', 'T', '-f',
            str(gpu_log_path)
        ]

        self.monitor_process = subprocess.Popen(cmd)
        self.logger.info(
            f"GPU monitoring started with PID: {self.monitor_process.pid}")
        time.sleep(2)

    def stop_gpu_monitoring(self):
        if self.monitor_process and self.monitor_process.poll() is None:
            self.logger.info(
                f"Stopping GPU monitoring (PID: {self.monitor_process.pid})")
            self.monitor_process.terminate()
            try:
                self.monitor_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.monitor_process.kill()
                self.monitor_process.wait()
            self.monitor_process = None

    def run_benchmark(self, config_name: str, config_path: Path,
                      batch_size: int, dataset_path: Path) -> bool:
        benchmark_log = self.output_dir / 'logs' / f'benchmark_{config_name}_bs{batch_size}.log'
        report_json = self.output_dir / 'reports' / f'report_{config_name}_bs{batch_size}.json'

        self.logger.info(
            f"Running benchmark: {config_name}, batch size: {batch_size}")
        self.logger.info(f"Config: {config_path}")
        self.logger.info(f"Dataset: {dataset_path}")
        self.logger.info(f"Benchmark log: {benchmark_log}")
        self.logger.info(f"Report: {report_json}")

        self.start_gpu_monitoring(config_name, batch_size)

        # Build benchmark command
        iteration_log = self.output_dir / 'logs' / f'iteration_{config_name}_bs{batch_size}.log'
        cmd = [
            'trtllm-bench', f'--model={self.model_name}',
            f'--model_path={self.model_path}', 'throughput',
            f'--dataset={dataset_path}', '--backend=pytorch',
            f'--max_batch_size={self.max_batch_size}',
            f'--max_num_tokens={self.max_num_tokens}',
            f'--concurrency={batch_size}',
            f'--num_requests={self.num_requests}',
            f'--extra_llm_api_options={config_path}',
            f'--report_json={report_json}', f'--iteration_log={iteration_log}'
        ]

        # Add parallelism options if specified
        if self.tp_size > 1:
            cmd.append(f'--tp={self.tp_size}')
        if self.pp_size > 1:
            cmd.append(f'--pp={self.pp_size}')

        self.logger.info(f"Executing: {' '.join(cmd)}")

        try:
            with open(benchmark_log, 'w') as f:
                result = subprocess.run(cmd,
                                        stdout=f,
                                        stderr=subprocess.STDOUT,
                                        text=True)

            if result.returncode == 0:
                self.logger.info(
                    f"Benchmark completed successfully for {config_name}, batch size {batch_size}"
                )
                success = True
            else:
                self.logger.error(
                    f"Benchmark failed for {config_name}, batch size {batch_size}. Check {benchmark_log}"
                )
                success = False

        except Exception as e:
            self.logger.error(f"Exception during benchmark: {e}")
            success = False
        finally:
            self.stop_gpu_monitoring()
            time.sleep(5)

        return success

    def run(self):
        self.logger.info("Starting CUDA Graph Padding Analysis")
        self.logger.info(f"Output directory: {self.output_dir}")

        try:
            self.create_directories()
            self.create_cuda_graph_configs()

            dataset_path = self.generate_dataset()
            configs = {
                # "default_padding": self.output_dir / "configs" / "padding_enabled_default.yaml",
                "no_padding":
                self.output_dir / "configs" / "padding_disabled.yaml",
                # "padding_slide_64":
                # self.output_dir / "configs" / "padding_slide_64.yaml",
            }

            total_tests = len(configs) * len(self.batch_sizes)
            current_test = 0

            for config_name, config_path in configs.items():
                self.logger.info(f"Testing configuration: {config_name}")

                for batch_size in self.batch_sizes:
                    current_test += 1
                    self.logger.info(
                        f"Progress: {current_test}/{total_tests} - Testing {config_name} with batch size {batch_size}"
                    )

                    if self.run_benchmark(config_name, config_path, batch_size,
                                          dataset_path):
                        self.logger.info(
                            f"✓ Completed: {config_name}, batch size {batch_size}"
                        )
                    else:
                        self.logger.error(
                            f"✗ Failed: {config_name}, batch size {batch_size}")

            self.logger.info("CUDA Graph Padding Analysis completed!")
            self.logger.info(f"Results available in: {self.output_dir}")

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            raise
        finally:
            self.cleanup()


def main():
    benchmark = CudaGraphBenchmark()
    benchmark.run()


if __name__ == "__main__":
    main()

<div align="center">

# Integrate verl into TensorRT-LLM

</div>

<div align="left">

Introduce how to integrate verl and use TensorRT-LLM as rollout backend.

## Quick Start
```shell
cd examples/verl_integration
pip install -r requirements.txt
```

In [here](/examples/verl_integraion/requirments.txt) ONLY enabled verl basic functionality without any training/inference package dependency, e.g., megatron, vllm, sglang, etc. This demonstration focus on FSDP (Training) + TensorRT-LLM (Inference) backend. According [verl/setup.py](https://github.com/volcengine/verl/blob/main/setup.py), for FSDP depends on torch package and TensorRT-LLM docker based on [NVIDIA official image PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html).

To test the basic setup:
```shell
chmod +x test_trtllm_verl.py 
./test_trtllm_verl.py > test_trtllm_verl.log 2>&1
```




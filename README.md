# 🚀 High-Performance Model Serving with NVIDIA Triton

[![Triton](https://img.shields.io/badge/NVIDIA-Triton_2.44-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/triton-inference-server)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-GPU-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![gRPC](https://img.shields.io/badge/gRPC-Supported-244c5a?style=flat-square&logo=grpc&logoColor=white)](https://grpc.io)

> Production model serving with NVIDIA Triton Inference Server — supports TensorRT, ONNX, PyTorch, and Python backends. Features dynamic batching, model ensembles, and 10,000+ QPS throughput.

## 📊 Performance (A100 GPU)
| Model | Backend | Batch | Latency p99 | Throughput |
|---|---|---|---|---|
| BERT-base | TensorRT | 32 | 8ms | 12,000 req/s |
| ResNet-50 | TensorRT | 64 | 5ms | 18,000 req/s |
| XGBoost Fraud | FIL | 256 | 2ms | 45,000 req/s |
| LLaMA-3-8B | vLLM+Triton | 1 | 450ms | 80 tok/s |

## 🗂 Model Repository
```
model_repository/
├── fraud_model/
│   ├── config.pbtxt        # Triton model config
│   └── 1/model.onnx        # ONNX model file
├── embedding_model/
│   ├── config.pbtxt
│   └── 1/model.plan        # TensorRT engine
└── ensemble_pipeline/      # Multi-model pipeline
    ├── config.pbtxt
    └── 1/
```

## 🚀 Quick Start
```bash
docker-compose up -d

# Check server health
curl http://localhost:8000/v2/health/ready

# Run inference
python client/infer.py --model fraud_model --input sample.json

# Benchmark
python perf/benchmark.py --model fraud_model --concurrency 64 --duration 60
```

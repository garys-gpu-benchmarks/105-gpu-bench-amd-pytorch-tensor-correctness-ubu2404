# PyTorch Tensor Op Correctness Suite Benchmark

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-host--safe-green.svg)](.github/workflows/ci.yml)

Target: Ubuntu 24.04 · AMD · see Hardware Requirements. This is a host benchmark, not a laptop `pip install` project.

## Quick Start

```bash
git clone https://github.com/garys-gpu-benchmarks/105-gpu-bench-amd-pytorch-tensor-correctness-ubu2404.git
cd 105-gpu-bench-amd-pytorch-tensor-correctness-ubu2404
sudo bash setup.sh --assume-yes
bash run_benchmark.sh --profile smoke --validate
```
Results are written to `results/benchmark.db` and `results/summary.json`.

This workload is executed on the validation host after the repository is copied there. `setup.sh` and `run_benchmark.sh` do not open an outbound SSH session.

Prerequisites: Ubuntu 24.04; AMD; Python 3.12.3; root or sudo for `setup.sh`. Framework: Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, HIP/ROCm. This is a host benchmark, not a laptop `pip install` project.

```mermaid
flowchart LR
  setup.sh --> run_benchmark.sh --> parse_results.py --> results/benchmark.db
```

## 1. Overview

Compares GPU PyTorch matmul/conv2d against a CPU FP32 reference for each yaml op/dtype/shape. op_name: matmul or conv2d. dtype and shape: sweep values. rtol/atol: comparison gates (defaults 0.02 / 0.1). Sweep dimensions: device_id, op_name, dtype, shape, rtol, atol, num_iterations, seed.

## 2. What It Validates

- Validates PyTorch GPU matmul/conv2d against CPU FP32 within rtol/atol. Does not run pytest
- #1: Max absolute error across correctness cases (max_abs_error); is present and physically sensible.
- #2: Maximum relative L2 error (max_rel_error); is present and physically sensible.
- #3: Tolerance-compliant case percentage (tolerance_compliance); is present and physically sensible.
- #4: Test case coverage count (coverage_count); is present and physically sensible.
- #5: Test case failure count (failure_count) is present and physically sensible.

## 3. Metrics Captured

- **#1: Max absolute error across correctness cases** — stored as `max_abs_error`.
- **#2: Maximum relative L2 error** — stored as `max_rel_error`.
- **#3: Tolerance-compliant case percentage** — stored as `tolerance_compliance`.
- **#4: Test case coverage count** — stored as `coverage_count`.
- **#5: Test case failure count** — stored as `failure_count`.

## 4. Hardware Requirements

### Supported environment

- OS: Ubuntu 24.04
- GPU vendor: AMD
- Framework family: Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, HIP/ROCm
- Python: Python 3.12.3

### Reference validation environment

The tables below describe the machine used to generate the reference results. They are not a requirement that every user buy that exact cloud instance.

### System

Ubuntu 24.04 / AMD / Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, HIP/ROCm

### GPU

Ubuntu 24.04 / AMD / Bash, SQLite, Python, PyYAML, ROCm Runtime, PyTorch-ROCm, HIP/ROCm

## 5. Software Requirements

| Component | Version |
|---|---|
| OS | Ubuntu 24.04 |
| Kernel | kernel 6.8.0 |
| Python | Python 3.12.3 |
| ROCm | ROCm 7.2.1 |
| rocBLAS | rocBLAS 5.2.0 |

Compares GPU PyTorch matmul/conv2d against a CPU FP32 reference for each yaml op/dtype/shape. op_name: matmul or conv2d. dtype and shape: sweep values.

## 6. Installation

```bash
Run scripts/run_tensor_correctness.py in the repository .venv. Does not run pytest or official test_ops.py
```

## 7. Running the Benchmark

```bash
Run scripts/run_tensor_correctness.py in the repository .venv. Does not run pytest or official test_ops.py
```

**Validating results separately:**

```bash
export BENCHMARK_PYTHON=/usr/bin/python3.13  # optional
python3 -m venv .venv
source ".venv/bin/activate"
".venv/bin/python" scripts/validate_results.py
```

## 8. Output

### `results/benchmark.db` (SQLite)

CSV with one row per op/dtype/shape case

sample_index,status,op_name,dtype,shape,max_abs_error,max_rel_error,tolerance_compliance,coverage_count,failure_count,error_message
0,ok,matmul,fp32,128x128,1.2e-4,8e-6,100,1,0,

```bash
Run scripts/run_tensor_correctness.py in the repository .venv. Does not run pytest or official test_ops.py
```

### `results/summary.json`

Consolidated metrics from the most recent run — suitable for CI artifact upload or dashboard ingestion.

### `results/raw/<timestamp>.txt`

CSV with one row per op/dtype/shape case

sample_index,status,op_name,dtype,shape,max_abs_error,max_rel_error,tolerance_compliance,coverage_count,failure_count,error_message
0,ok,matmul,fp32,128x128,1.2e-4,8e-6,100,1,0,

## 9. Baselines / Thresholds

Expected ranges and gates live in `config/benchmark_config.yaml` under `baselines:` or `thresholds:`. To update them, edit that file — never edit validation code directly.

## 10. Troubleshooting

**`setup.sh` missing collector**
Create cannot finish without `scripts/collect_workload.py`.

**`self_check` overlay rewritten**
Do not overwrite files listed in `results/overlay_lock.json`.

**Remote SSH drop during setup**
Reconnect and resume `bash setup.sh --assume-yes`. Do not wipe `.venv` or `.cache`.

## 11. NVIDIA H100 Coding Differences

Primary target is AMD ROCm. NVIDIA notes in this section are reference only and are not the execution path.

## Repository layout

```text
.
├── setup.sh
├── run_benchmark.sh
├── benchmark_specification.json
├── config/
├── scripts/
├── src/
├── tests/
├── docs/
├── results/
└── LICENSE
```

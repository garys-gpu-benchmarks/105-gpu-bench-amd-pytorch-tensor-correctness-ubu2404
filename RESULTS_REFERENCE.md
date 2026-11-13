# Results: 105 PyTorch Tensor Op Correctness Suite

**Main goal:** Analyze tensor op numerical correctness  
**Tools:** Run scripts/run_tensor_correctness.py in the repo .venv: GPU matmul/conv2d vs CPU FP32 per op, dtype, and shape

Reference results measured by the maintainer on the system shown in each section (generator TEMPLATE_00_139 · recorded 2026-11-13). Your own runs write `results/RESULTS_ACTUAL.md` in the same format.

> Smoke runs use deliberately small sizes. They check that the workload installs and runs end to end; their numbers are not performance results. Baseline and extended are.

<!-- profile:smoke -->
## smoke

- **System:** 1× AMD Instinct MI300X VF (196288 MB) · INTEL(R) XEON(R) PLATINUM 8568Y+ · Ubuntu 24.04.5 LTS · kernel 6.8.0-147-generic · driver 7.1.9.31600000
- **Run:** 2026-11-13T13:53:52Z · status **PASS** (exit 0)
- **Time (mm:ss):** setup 00:55 · benchmark 00:13 · whole run 01:22
- **Source:** run 105_20261113_135352
- **Command:** `bash run_benchmark.sh --smoke --device-id 0 --op-name matmul,conv2d --dtype f16_r --shape matmul=1x128x128;conv2d=1x3x32x32 --rtol 0.02 --atol 0.1 --num-iterations 2 --seed 42`

| # | Metric | Result |
|---|---|---|
| 1 | Max absolute error across correctness cases (max_abs_error) | 0.018320083618164062 |
| 2 | Maximum relative L2 error (max_rel_error) | 0.0008462772765480011 |
| 3 | Tolerance-compliant case percentage (tolerance_compliance) | 100.0 |
| 4 | Test case coverage count (coverage_count) | 2.0 |
| 5 | Test case failure count (failure_count) | 0.0 |
<!-- /profile:smoke -->

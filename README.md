# PatchSentry

Unsupervised visual anomaly detection for industrial inspection: a from-scratch PatchCore
implementation with statistically calibrated alarm thresholds, ONNX-optimised inference and a
FastAPI serving layer with dynamic micro-batching.

> **This project is under construction.** Nothing below is a verified result yet. No metric or
> benchmark will appear in this file until a script in this repository has produced it.

---

## Project status and handoff

**To continue this work (human or AI): read this section, then carry on from "Next step".**
Update this section whenever a step is finished, including anything that failed.

| # | Step | Status |
|---|------|--------|
| 1 | Choose project and write this plan | Done |
| 2 | Inspect environment (Python, Docker, `gh`, network, disk) | Done |
| 3 | Scaffold repository, dependency management, git | Done |
| 4 | Download dataset (see "Dataset plan") | In progress |
| 5 | Data ingestion and preprocessing | Not started |
| 6 | Baselines: pixel autoencoder, PaDiM, image-level k-NN | Not started |
| 7 | PatchCore: feature extractor, greedy coreset, FAISS index, re-weighted scoring | Not started |
| 8 | Evaluation: image AUROC, pixel AUROC, AUPRO; MLflow tracking; hyperparameter sweep | Not started |
| 9 | Conformal threshold calibration, with empirical false-alarm-rate validation | Not started |
| 10 | ONNX export, quantisation, latency and throughput benchmarks | Not started |
| 11 | FastAPI service: validation, micro-batching, audit log, metrics | Not started |
| 12 | pytest suite, ruff, mypy | Not started |
| 13 | Dockerfile, Docker Compose, GitHub Actions | Not started |
| 14 | Final README with real results; architecture diagrams | Not started |
| 15 | Secret scan, frozen dependencies, publish public GitHub repo | Not started |

**Next step:** finish step 4, run the tests, then the sweep (`pipeline/sweep.py`).

**Machine notes (read before running anything on the author's Mac)**
- This folder is on the iCloud-synced Desktop and the disk is nearly full, so macOS evicts files to the
  cloud. A virtualenv here hangs on import, with 0% CPU, while files download again. So heavy files live
  outside iCloud in `~/.local/share/patchsentry` (`venv/`, `data/`, `artifacts/`); `data` and `artifacts`
  here are symlinks to it.
- Run commands with `export UV_PROJECT_ENVIRONMENT=~/.local/share/patchsentry/venv`, and call
  `~/.local/share/patchsentry/venv/bin/python` directly. Do not create `.venv` in this folder.
- 16 GB RAM (the brief says 24), about 8 GB free disk. Docker's daemon was not running at inspection.

**Rules for this project**
- Never fabricate or estimate a result. Every number in this README comes from a committed script
  and a saved file under `reports/`.
- Never claim something works unless it was run. If Docker or a GPU is unavailable, say so here.
- Never commit secrets, `.env` files, datasets or large model weights.

---

## Why this project

- **Problem:** real production lines have plenty of images of good parts and very few of defects,
  and new defect types keep appearing. Supervised detectors cannot cover defects they never saw.
- **Approach:** learn only what "normal" looks like and flag deviations.
- **Distinctive parts:**
  - PatchCore is implemented from scratch, not wrapped from a library, so each decision can be
    explained in an interview.
  - Thresholds are set with split conformal prediction, which gives a finite-sample bound on the
    false alarm rate instead of a threshold tuned by hand on test data.
  - The serving layer is engineered and measured: ONNX Runtime, quantisation and dynamic
    micro-batching, with latency and throughput reported per configuration.

## Planned architecture

```
Normal training images
   -> pretrained CNN backbone (mid-level layers)
   -> locally aware patch features
   -> greedy k-center coreset (memory bank)
   -> FAISS nearest-neighbour index

Held-out normal images -> anomaly scores -> conformal threshold (target false alarm rate)

Client -> FastAPI (validation) -> micro-batcher -> ONNX Runtime backbone -> FAISS lookup
       -> anomaly score, heatmap, decision -> audit log and metrics
```

## Dataset plan

Preferred order, to be confirmed by checking availability and licence during step 4:

1. **VisA** (Visual Anomaly dataset, Amazon): licensed CC BY 4.0, which suits a public repository.
2. **MVTec AD**: the standard benchmark, licensed CC BY-NC-SA 4.0 (non-commercial).

Datasets are downloaded by a script and never committed.

## Planned technology

Python, PyTorch, torchvision, FAISS, scikit-learn, NumPy, OpenCV, ONNX, ONNX Runtime, MLflow,
FastAPI, Pydantic, PostgreSQL, Docker, Docker Compose, GitHub Actions, pytest.

Local development targets a MacBook Air M3 (24 GB RAM) on CPU/MPS. The inference backend is
configurable so that the same code can run on a CUDA GPU in the cloud.

<!--
  Unified Semi-Supervised Object Detection Suite
  README.md (structured, sellable, and actionable)
-->

# Unified Semi-Supervised Object Detection Suite

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#requirements)

Production-ready, label-efficient object detection: a complete toolkit to train a teacher model, generate pseudo-labels on unlabeled images, train a student model on combined data, and export optimized models for edge deployment (TFLite / ONNX / NCNN).

## Table of contents
- [Why this project](#why-this-project)
- [Why semi-supervised detection](#why-semi-supervised-detection)
- [Why quantize & TFLite for edge](#why-quantize--tflite-for-edge)
- [What the pipeline does](#what-the-pipeline-does)
- [Files & layout](#files--layout)
- [Quick start (try it)](#quick-start-try-it)
- [Training / conversion flow (details)](#training--conversion-flow-details)
- [Calibration & quantization guidance](#calibration--quantization-guidance)
- [Evaluation, limitations, and recommendations](#evaluation-limitations-and-recommendations)
- [Contributing & next steps](#contributing--next-steps)
- [License & contact](#license--contact)

## Why this project

- Reduces annotation cost by leveraging unlabeled images to produce high-quality detectors with far fewer manual boxes.
- Practical teacher-student pseudo-labeling pipeline that scales to off-the-shelf detectors.
- End-to-end: dataset prep, training, pseudo-labeling, student retraining, model export and calibration artifacts for quantized deployment.

## Why semi-supervised detection

Labeling bounding boxes is expensive and slow. For many applied tasks (industrial inspection, retail shelving, custom surveillance) you can collect huge amounts of unlabeled images cheaply. Semi-supervised learning (teacher -> pseudo-label -> student) lets you convert that cheap data into additional training signal with minimal human effort.

Key benefits:
- Label-efficiency: reduce the number of manual boxes required.
- Faster iteration: prototype and iterate with a small labeled set and expand dataset programmatically.
- Cost-effective model improvements: much of the performance gap to fully supervised training can be closed using pseudo-labels.

## Why quantize & TFLite for edge

Edge devices (mobile phones, embedded boards, IoT cameras) have tight constraints on storage, memory, CPU, power and thermal envelope. To make models usable on such devices, we convert and quantize models:

- Smaller models (FP32 -> INT8 reduces size ~4x) fit limited flash/storage.
- Integer or mixed-precision inference runs faster on CPUs and accelerators and uses less power.
- TFLite is an edge-first runtime with wide vendor support (Android, iOS, Raspberry Pi, vendor NN accelerators) and hardware delegates.

This repo includes conversion scripts and calibration images (see `semi_supervised_output/exports/calibration_data`) to perform safe post-training quantization while minimizing accuracy drop.

## What the pipeline does

1. Train a teacher on the labeled dataset (`teacher_train.py`).
2. Generate pseudo-labels (confidence filtering + NMS) on unlabeled images using the teacher (`pseudo.py`).
3. Combine labeled + pseudo-labeled data, retrain student (`pipeline.py` / `final.py`).
4. Export final model to ONNX, TFLite, NCNN (`tflite.py`, `ncnn.py`, `unified.py`) and optionally quantize.

## Files & layout (important)

- `class.py` — model helpers.
- `data_prep.py` — dataset organization and conversion.
- `teacher_train.py` — train teacher model.
- `pseudo.py` — generate pseudo-labels.
- `pipeline.py` — orchestrate student training and postprocessing.
- `final.py` — evaluation and finalization.
- `tflite.py`, `ncnn.py`, `unified.py` — conversion/export helpers.
- `utilities.py` — shared utilities.
- `final_model/` — example final artifacts and training arguments.
- `prepared_data/` — example dataset layout.
- `semi_supervised_output/exports/calibration_data` — calibration images + calib.yaml for quantization.

Notable artifacts in this repo (already included):
- `final_model/final/weights/best.pt` — example final weights.
- `teacher_model/teacher/weights/best.pt` — teacher weights.
- `yolo11n.pt` — an additional model checkpoint present in the workspace.

## Quick start (short)

1. Create a virtual environment and install dependencies (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirment.txt
```

2. Train a teacher (example):

```powershell
python teacher_train.py
```

3. Generate pseudo-labels on unlabeled images:

```powershell
python pseudo.py
```

4. Run the pipeline (student training + export):

```powershell
python pipeline.py
python final.py
python tflite.py  # optional export
```

See script docstrings and `final_model/*/args.yaml` for example CLI options and advanced configs.

## Training / conversion flow (details)

- Teacher training produces a robust checkpoint and validation metrics.
- Pseudo-label generation applies conservative filtering and writes YOLO-style `.txt` files alongside unlabeled images.
- Student training uses combined data with weighting strategies to mitigate noisy pseudo-labels.
- Export paths convert PyTorch -> ONNX -> TFLite/NCNN and optionally apply post-training quantization using representative calibration images.

## Calibration & quantization guidance

- Use `semi_supervised_output/exports/calibration_data/images` as representative calibration images.
- Calibrate INT8 quantization on images that match deployment conditions (lighting, object size, viewpoint).
- Validate quantized model on a holdout set and compare per-class mAP—if drop is large, try mixed-precision or smaller specialized models.

## Evaluation, limitations, and recommendations

- Bias propagation: the teacher biases can be amplified via pseudo-labels. Manually review low-performing classes.
- Class imbalance: use loss weighting or class-aware sampling for rare classes.
- Domain shift: if unlabeled data is from a different distribution, add a small labeled seed from that domain.



## License & contact

MIT License — reuse and adapt freely. Open issues or PRs for questions and improvements.



# Unified Semi-Supervised Object Detection Suite

Make high-quality object detectors with less labeling. This repository is a complete, end-to-end toolkit for semi-supervised object detection, training, pseudo-labeling, export, and deployment. It bundles scripts, data preparation helpers, teacher/student training pipelines, model exports (PyTorch/ONNX/TFLite/NCNN), and reusable utilities — all wired together so you can reproduce results quickly or integrate the pipeline into production.



Why this project matters
- Reduces the need for dense labels with a pragmatic pseudo-labeling pipeline.
- Ships ready-to-run scripts for teacher training, pseudo-label generation, student training, and model export.
- Includes export pathways for TFLite / ONNX / NCNN so models can be deployed on edge/embedded hardware.
- Clean dataset layout and utilities for preparing YOLO-style datasets.

Highlights / Quick snapshot
- Fully scripted teacher-first semi-supervised training (see `teacher_train.py`).
- Pseudo-label generation and pipeline orchestration (`pseudo.py`, `pipeline.py`).
- Final model conversion and export scripts (`final.py`, `tflite.py`, `ncnn.py`, `unified.py`).
- Utilities for dataset preparation, augmentation and evaluation (`data_prep.py`, `utilities.py`).
- Example pretrained weights included at `final_model/final/weights/best.pt` so you can test inference quickly.

Who this is for
- ML engineers who want to accelerate dataset creation and increase data efficiency.
- Researchers exploring practical semi-supervised object detection techniques.
- Embedded/edge developers who need exportable models (TFLite/ONNX/NCNN).

Repository contents (important files & directories)
- `class.py` — model definition helpers and lightweight class utilities.
- `data_prep.py` — dataset preparation utilities for converting/organizing images & labels.
- `teacher_train.py` — training script to produce a robust teacher model.
- `pseudo.py` — pseudo-label generation logic and confidence filtering.
- `pipeline.py` — high-level orchestration that stitches the pipeline steps together.
- `final.py` — finalization and evaluation utilities used to create production-ready models.
- `tflite.py`, `ncnn.py`, `unified.py` — model export and conversion helpers.
- `utilities.py` — assorted helper functions used throughout the scripts.
- `final_model/` — example final artifacts and trained weights (look under `final_model/final/weights`).
- `prepared_data/` and `pseudo_labeled/` — dataset partitions and outputs from the pipeline.
- `requirment.txt` — Python dependencies (note: name is `requirment.txt` in repo).

Quick demo — run a full pipeline (fast sanity check)
1. Create or activate a Python virtual environment (recommended Python 3.11-3.12).
2. Install dependencies: this repo lists packages in `requirment.txt`.

	powershell
	python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirment.txt

3. Train a small teacher model (example)

	powershell
	python teacher_train.py

4. Generate pseudo labels and run pipeline

	powershell
	python pseudo.py
	python pipeline.py

5. Create final model and export

	powershell
	python final.py
	python tflite.py
	python ncnn.py

These commands are intentionally high-level — the scripts accept configuration files and flags for advanced control. See the script docstrings and `final_model/*/args.yaml` for example settings.

Installation & dependencies
- Recommended: Python 3.10 or 3.11.
- Create a virtual environment and install dependencies from `requirment.txt`.
- Troubleshooting: if a package fails to install on Windows, try upgrading pip and wheel and re-running the install step.

# Unified Semi-Supervised Object Detection Suite — Detailed Overview

This repository is a production-oriented toolkit that implements a practical semi-supervised object detection workflow. It is designed to reduce the manual labeling burden while producing accurate, deployable object detectors that can be exported for edge devices (TFLite, ONNX, NCNN).

This file explains clearly what the project does, why semi-supervised learning was chosen, who benefits from it, and how the pipeline is organized so engineers, researchers, and product teams can evaluate and reuse the work.

What this project does (concise)
- Trains a high-quality teacher detector on a relatively small labeled dataset.
- Uses the teacher to generate pseudo-labels on a larger unlabeled dataset, including confidence filtering and heuristics to reduce label noise.
- Retrains (or fine-tunes) a student model on the combined labeled + pseudo-labeled data to achieve performance comparable to fully supervised training at a fraction of labeling cost.
- Converts the final model into ONNX, TFLite, and NCNN formats for deployment on cloud, mobile, and embedded hardware.

Problem statement — why this exists
Labeling large-scale object detection datasets is expensive and slow: bounding boxes require per-image, per-object annotations and expert reviewers for quality control. For many applied problems — inventory monitoring, low-volume product defect detection, custom surveillance classes — teams cannot afford to label tens of thousands of images.

Why semi-supervised object detection? (the core rationale)
- Label-efficiency: Semi-supervised learning (SSL) can significantly reduce annotated data requirements by leveraging a large pool of unlabeled images to provide extra training signal.
- Practicality: Pseudo-labeling (teacher-student) is simple to implement, scales to off-the-shelf detectors, and integrates well with existing augmentation and training schedules.
- Cost vs. performance tradeoff: For many domains, the marginal performance gap between a fully supervised model and a pseudo-labeled student is small relative to the annotation cost saved. This repository is built around that pragmatic tradeoff.
- Deployment-ready: The pipeline focuses not only on model quality but on generating compact, exported models suitable for edge inference.

Who should use this
- ML engineers and data scientists who need to accelerate labeling and iterate quickly on detector development.
- Startups and product teams that have limited labeling budgets but access to lots of unlabeled image data.
- Researchers who want a reproducible baseline for teacher-student pseudo-labeling and conversion to production formats.
- Edge developers needing exported models and calibration artifacts for quantized deployments.

High-level pipeline and technical approach
1) Teacher training
	- Train a standard object detector (PyTorch-based scripts) on the available labeled dataset. Training includes augmentation and validation checkpoints.
	- Result: a robust teacher model stored under `teacher_model/teacher/weights/`.

2) Pseudo-labeling
	- Use the teacher to run inference over unlabeled images.
	- Apply confidence thresholding, non-maximum suppression, and simple heuristics to remove low-quality predictions.
	- Output: YOLO-style label files for pseudo-labeled images placed under `pseudo_labeled/` (see `pseudo.py`).

3) Student training
	- Combine ground-truth labeled data with filtered pseudo-labels.
	- Retrain a student detector (often the same architecture, but could be smaller) with a curriculum or weighted loss to down-weight noisy pseudo-labels.
	- Result: final weights located under `final_model/final/weights/`.

4) Export and deployment
	- Convert the final PyTorch model to ONNX and then to TFLite/NCNN as needed.
	- Optionally apply post-training quantization or calibration using provided calibration images in `semi_supervised_output/exports/calibration_data`.

Key design decisions and why they matter
- Teacher-first pseudo-labeling: simple, effective, and compatible with many detector backbones. It avoids complex consistency losses and is straightforward to debug.
- Confidence filtering + heuristics: reduces label noise which is the primary failure mode of pseudo-labeling. This repository favors robust, conservative filtering over aggressive label expansion.
- Modular scripts: every step is scriptable (`teacher_train.py`, `pseudo.py`, `pipeline.py`, `final.py`) so teams can replace components (different backbone, augmentations, or post-processing) without reworking the entire pipeline.

Data requirements and formats
- Labeled set: YOLO-style image + one .txt per image with class and normalized bbox coordinates.
- Unlabeled set: images folder; pseudo-labeling produces matching .txt files.
- Example dataset layouts are present under `prepared_data/` and `final_model/unified_dataset/`.

Benefits and expected outcomes
- Reduce annotation cost: Expect to cut manual labeling needs by a large factor depending on problem complexity and available unlabeled data.
- Faster iteration: Developers can prototype detectors with fewer labeled images and expand the dataset automatically via pseudo-labels.
- Deployment-ready artifacts: Exported ONNX/TFLite/NCNN models and calibration data enable rapid rollout to production.

Concrete use cases
- Industrial inspection where collecting unlabeled images is cheap but labeling defects is expensive.
- Retail/merchandising detection for new SKUs — unlabeled shelf photos can be pseudo-labeled quickly.
- Domain adaptation: bootstrap detectors for a new camera or environment using a small labeled seed set plus many unlabeled frames.

How to get started (short and practical)
1. Create a virtual environment and install packages (Windows PowerShell example):

	powershell
	python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirment.txt

2. Train a teacher on your labeled data:

	powershell
	python teacher_train.py

3. Generate pseudo-labels on your unlabeled images:

	powershell
	python pseudo.py

4. Run the pipeline to train the student and export final artifacts:

	powershell
	python pipeline.py
	python final.py
	python tflite.py  # optional export

Configuration and customization
- Each script accepts configuration settings (check docstrings and YAML args under `final_model/*/args.yaml`). Adjust confidence thresholds, loss weights, and augmentation according to dataset characteristics.

Included artifacts and reproducibility
- Example trained weights: `final_model/final/weights/best.pt`.
- Example configs and training args: `final_model/unified_data.yaml`, `final_model/final/args.yaml`.
- Calibration images and exports for quantized flows in `semi_supervised_output/exports/`.

Evaluation
- Standard metrics (mAP, precision, recall) are produced during validation; results are accessible in `final_model/final/results.csv` where available.

Limitations and failure modes (be explicit)
- If teacher predictions are systematically biased, pseudo-labels will propagate that bias to the student. Monitor per-class metrics and consider manual curation for under-performing classes.
- Highly imbalanced classes may require class-aware sampling or loss weighting during student training.
- Extreme domain shift between labeled and unlabeled data reduces the benefits of pseudo-labeling — consider unsupervised domain adaptation or additional labeled samples in that case.

Contributing and recommended next work
- Add a small CI job that runs a quick end-to-end smoke test on a tiny dataset.
- Add a Jupyter tutorial that demonstrates a minimal end-to-end example with visualizations of pseudo-label quality.

License and reuse
- MIT License. Use freely; attribution appreciated.

If you want this turned into a rendered `README.md` (with badges and a short demo GIF) I can convert and add images and badges next.

Quantization & TFLite — why we export and why this matters for edge
---------------------------------------------------------------

1) Why quantize models?
- Smaller model size: Quantization reduces model size (for example, FP32 -> INT8) by ~4x which matters when shipping models to devices with limited storage.
- Faster inference: INT8 and other reduced-precision arithmetic runs faster on many mobile and embedded CPUs and on specialized accelerators that support integer ops.
- Lower power consumption: fewer bits and optimized integer kernels consume less power — vital for battery-powered devices.
- Practical tradeoff: Post-training quantization often introduces only a small accuracy drop if calibration and representative datasets are used; the size and latency benefits usually outweigh the small performance hit for edge applications.

2) Why TFLite?
- Edge-first format: TFLite is designed specifically for mobile and embedded inference with optimized kernels for ARM CPUs, DSPs, and NPUs.
- Broad device support: TFLite works across Android, iOS, Raspberry Pi, and many vendor-specific accelerators with consistent runtime APIs.
- Quantization and delegates: TFLite provides robust post-training quantization tools and support for hardware delegates (e.g., NNAPI, GPU delegate) to accelerate inference without changing model code.
- Easy integration: TFLite models can be bundled directly into mobile apps or flashed onto devices, and there are well-established tools for converting from ONNX or TensorFlow to TFLite.

3) How this repo uses quantization and TFLite
- The `tflite.py` script converts and optionally quantizes the trained PyTorch model (via ONNX or TensorFlow conversion paths). Calibration images are included under `semi_supervised_output/exports/calibration_data` to improve INT8 quantization quality.
- We aim for conservative quantization: calibrate with representative images, validate the quantized model on a holdout set, and monitor mAP drop per-class. The README's calibration artifacts and example scripts demonstrate this flow.

4) Practical notes and recommendations
- Always validate quantized models on a validation set that matches your deployment domain.
- Use representative calibration images that reflect the distribution of your target environment (lighting, scale, viewpoint).
- If INT8 degrades accuracy too much, try mixed-precision (e.g., weight quantization but keep activations in FP16) or smaller model architectures tuned for edge.



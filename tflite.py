from ultralytics import YOLO
from pathlib import Path
import torch

def export_tflite_int8(weights_path, calib_yaml_path, out_dir, imgsz=640, device=None):
    """
    Exports the given weights to a TFLite INT8 quantized model using Ultralytics export API.
    - weights_path: path to final PyTorch weights (.pt)
    - calib_yaml_path: path to the calibration data.yaml (created above)
    - out_dir: directory where exported artifacts will be written
    - imgsz: export image size
    - device: 0 (gpu) or 'cpu'
    Returns path to the exported .tflite file.
    """
    weights_path = Path(weights_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'

    model = YOLO(str(weights_path))

    # Ultralytics exporter: enable int8 and pass the calibration data yaml via `data`.
    # The exporter will write an output file like <model>_int8.tflite into cwd/project folder.
    model.export(format="tflite", int8=True, data=str(calib_yaml_path), imgsz=imgsz, device=device, project=str(out_dir), name="tflite_int8", exist_ok=True)

    # find the produced tflite file (search out_dir)
    candidates = list(out_dir.rglob("*.tflite"))
    if not candidates:
        raise FileNotFoundError("TFLite export completed but no .tflite file found in out_dir.")
    # return the most recently modified .tflite (usually the correct one)
    tflite_path = max(candidates, key=lambda p: p.stat().st_mtime)
    return Path(tflite_path)

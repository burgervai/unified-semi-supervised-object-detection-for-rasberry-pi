from ultralytics import YOLO
from pathlib import Path
import torch

def export_ncnn(weights_path, out_dir, imgsz=640, device=None):
    """
    Exports the model to NCNN format (produces .param and .bin files).
    Note: INT8 quantization for NCNN is NOT supported via Ultralytics exporter at time of writing.
    """
    weights_path = Path(weights_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'

    model = YOLO(str(weights_path))
    # NCNN export (no int8 parameter available)
    model.export(format="ncnn", imgsz=imgsz, device=device, project=str(out_dir), name="ncnn", exist_ok=True)

    param_files = list(out_dir.rglob("*.param"))
    bin_files = list(out_dir.rglob("*.bin"))
    if not param_files or not bin_files:
        raise FileNotFoundError("NCNN export did not produce .param/.bin files in out_dir.")
    # return tuple (param_path, bin_path) choosing most recent
    param_path = max(param_files, key=lambda p: p.stat().st_mtime)
    bin_path = max(bin_files, key=lambda p: p.stat().st_mtime)
    return param_path, bin_path

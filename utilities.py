from pathlib import Path
import random
import shutil
import yaml
from tflite import export_tflite_int8
from ncnn import export_ncnn
import torch

def collect_representative_images(source_images_dir, calib_dir, num_images=300, seed=42):
    """
    Copies up to `num_images` representative images from source_images_dir (recursive)
    into calib_dir/images. Overwrites existing folder if present.
    """
    src = Path(source_images_dir)
    dst_images = Path(calib_dir) / "images"
    if dst_images.exists():
        shutil.rmtree(dst_images)
    dst_images.mkdir(parents=True, exist_ok=True)

    candidates = [p for p in src.glob("**/*") if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
    random.Random(seed).shuffle(candidates)
    selected = candidates[:min(num_images, len(candidates))]

    for p in selected:
        shutil.copy(p, dst_images / p.name)

    return dst_images  # path with calibration images


def create_calibration_yaml(calib_dir, out_yaml_path, names=None, nc=None):
    """
    Writes a minimal data.yaml pointing to the calib_dir so Ultralytics exporter can use it for INT8 calibration.
    The structure will be:
      <out_yaml_parent>/
        calibration/
          images/   # images placed here by collect_representative_images
    """
    calib_dir = Path(calib_dir)
    cfg = {
        'path': str(calib_dir.parent.absolute()),   # parent so 'calibration/images' resolves
        'train': 'calibration/images',
        'val': 'calibration/images'   # val can be same for calibration
    }
    if nc is not None:
        cfg['nc'] = int(nc)
    if names is not None:
        cfg['names'] = names

    out_yaml_path = Path(out_yaml_path)
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml_path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return out_yaml_path

def prepare_and_export_final_model(final_weights, calibration_source, output_dir, num_calibration_images=300):
    """
    Prepares calibration data and exports the final model to TFLite and NCNN formats.
    """
    final_weights = Path(final_weights)
    calibration_source = Path(calibration_source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- TFLite INT8 Export ---
    print("\n[1/2] Preparing for TFLite INT8 export...")
    calib_dir = output_dir / "calibration_data"
    
    # Collect representative images for calibration
    collect_representative_images(
        source_images_dir=calibration_source,
        calib_dir=calib_dir,
        num_images=num_calibration_images
    )
    
    # Create the calibration YAML file
    calib_yaml_path = create_calibration_yaml(calib_dir, calib_dir / "calib.yaml")
    print(f"  - Calibration data prepared at: {calib_dir}")

    # Determine device for export: use 'cpu' if CUDA is not available
    device = 'cpu' if not torch.cuda.is_available() else '0'

    # Export to TFLite INT8
    tflite_path = export_tflite_int8(
        weights_path=final_weights,
        calib_yaml_path=calib_yaml_path,
        out_dir=output_dir,
        device=device
    )
    print(f"  - TFLite model exported to: {tflite_path}")

    # --- NCNN Export ---
    print("\n[2/2] Exporting to NCNN format...")
    ncnn_paths = export_ncnn(weights_path=final_weights, out_dir=output_dir, device=device)
    print(f"  - NCNN model exported.")

    return tflite_path, ncnn_paths

from pathlib import Path
from utilities import prepare_and_export_final_model

if __name__ == "__main__":
    final_weights = Path("semi_supervised_output/final_model/final/weights/best.pt")
    calibration_source = Path("semi_supervised_output/final_model/unified_dataset/train/images")
    output_dir = Path("semi_supervised_output/exports")

    print("\n🔹 Phase 5: Export & Deployment Preparation Started")
    print("--------------------------------------------------")

    print(f"📂 Using trained model: {final_weights}")
    print(f"📸 Collecting representative calibration images from: {calibration_source}")

    tflite_path, ncnn_paths = prepare_and_export_final_model(
        final_weights=final_weights,
        calibration_source=calibration_source,
        output_dir=output_dir,
        num_calibration_images=50
    )

    print("\n✅ Export completed successfully.")
    print("--------------------------------------------------")

    print(f"📦 TFLite INT8 model saved at: {tflite_path}")
    print(f"📦 NCNN model files saved at: {ncnn_paths[0]} and {ncnn_paths[1]}")
    print("--------------------------------------------------")
    print("🚀 Model is now ready for deployment on Raspberry Pi.")

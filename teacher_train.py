from ultralytics import YOLO
from pathlib import Path
import torch

class TeacherTrainer:
    def __init__(self, data_yaml, model_name="yolo11n.pt", output_dir="teacher_model"):
        self.data_yaml = data_yaml
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train(self, epochs=100, imgsz=640, batch=16, patience=20, device=None):
        model = YOLO(self.model_name)

        if device is None:
            device = 'cpu' if not torch.cuda.is_available() else 0
        
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=device,
            project=str(self.output_dir),
            name="teacher",
            save=True,
            save_period=10,
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            box=7.5,
            cls=0.5,
            dfl=1.5,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=0.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            val=True,
            plots=True,
            verbose=False
        )
        
        best_weights = self.output_dir / "teacher" / "weights" / "best.pt"
        return best_weights, results
    
    def evaluate(self, weights_path, device=None):
        model = YOLO(weights_path)

        if device is None:
            device = 'cpu' if not torch.cuda.is_available() else 0

        metrics = model.val(data=self.data_yaml, device=device, verbose=False)
        
        # Convert numpy arrays to lists for JSON serialization
        precision = metrics.box.p
        recall = metrics.box.r
        
        return {
            'mAP50': metrics.box.map50,
            'mAP50-95': metrics.box.map,
            'precision': precision.tolist() if hasattr(precision, 'tolist') else precision,
            'recall': recall.tolist() if hasattr(recall, 'tolist') else recall
        }


if __name__ == "__main__":
    trainer = TeacherTrainer(data_yaml="prepared_data/data.yaml")
    best_weights, results = trainer.train(epochs=100, batch=16, device='cpu')
    metrics = trainer.evaluate(best_weights, device='cpu')
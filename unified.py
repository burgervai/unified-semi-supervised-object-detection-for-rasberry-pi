from ultralytics import YOLO
from pathlib import Path
import shutil
import yaml
import torch
from sklearn.model_selection import train_test_split

class UnifiedTrainer:
    def __init__(self, labeled_dir, pseudo_labeled_dir, background_dir, output_dir="final_model"):
        self.labeled_dir = Path(labeled_dir)
        self.pseudo_labeled_dir = Path(pseudo_labeled_dir)
        self.background_dir = Path(background_dir)
        self.output_dir = Path(output_dir)
        # Create a clean unified dataset directory under the output folder
        self.unified_dir = self.output_dir / "unified_dataset"
        self.unified_dir.mkdir(parents=True, exist_ok=True)
        
    def combine_datasets(self, val_ratio=0.15):
        train_dir = self.unified_dir / "train"
        val_dir = self.unified_dir / "val"
        
        for split_dir in [train_dir, val_dir]:
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        all_labeled_images = []
        
        for split in ["train", "val"]:
            src_img_dir = self.labeled_dir / split / "images"
            src_lbl_dir = self.labeled_dir / split / "labels"
            
            if src_img_dir.exists():
                for img_path in src_img_dir.glob("*"):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        all_labeled_images.append((img_path, src_lbl_dir / img_path.with_suffix('.txt').name))
        
        pseudo_img_dir = self.pseudo_labeled_dir / "images"
        pseudo_lbl_dir = self.pseudo_labeled_dir / "labels"
        
        if pseudo_img_dir.exists():
            for img_path in pseudo_img_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    label_path = pseudo_lbl_dir / img_path.with_suffix('.txt').name
                    if label_path.exists():
                        all_labeled_images.append((img_path, label_path))
        
        train_data, val_data = train_test_split(all_labeled_images, test_size=val_ratio, random_state=42)
        
        for img_path, label_path in train_data:
            shutil.copy(img_path, train_dir / "images" / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, train_dir / "labels" / label_path.name)
        
        for img_path, label_path in val_data:
            shutil.copy(img_path, val_dir / "images" / img_path.name)
            if label_path.exists():
                shutil.copy(label_path, val_dir / "labels" / label_path.name)
        
        bg_img_dir = self.background_dir / "images"
        if bg_img_dir.exists():
            for img_path in bg_img_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    shutil.copy(img_path, train_dir / "images" / f"bg_{img_path.name}")
        
        return len(train_data), len(val_data)
    
    def create_yaml_config(self, original_yaml_path):
        with open(original_yaml_path, 'r') as f:
            original_config = yaml.safe_load(f)
        
        config = {
            'path': str(self.unified_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': original_config['nc'],
            'names': original_config['names']
        }
        
        yaml_path = self.output_dir / "unified_data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return yaml_path
    
    def train_final_model(self, data_yaml, teacher_weights=None, epochs=150, 
                          imgsz=640, batch=16, patience=30, device='cpu'):
        if teacher_weights and Path(teacher_weights).exists():
            model = YOLO(teacher_weights)
        else:
            model = YOLO("yolo11n.pt")

        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'
        
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=device,
            project=str(self.output_dir),
            name="final",
            save=True,
            save_period=15,
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.005,
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
            mixup=0.1,
            copy_paste=0.0,
            val=True,
            plots=True,
            verbose=False
        )
        
        best_weights = self.output_dir / "final" / "weights" / "best.pt"
        return best_weights, results
    
    def evaluate(self, weights_path, data_yaml, device=None):
        model = YOLO(weights_path)

        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'

        metrics = model.val(data=data_yaml, device=device, verbose=False)
        
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
    trainer = UnifiedTrainer(
        labeled_dir="C:/Users/Niloy/Downloads/object detection/dataset with lebel/indoor obstacles.v1i.yolov8",
        pseudo_labeled_dir="pseudo_labeled",
        background_dir="pseudo_labeled/background"
    )
    
    train_count, val_count = trainer.combine_datasets(val_ratio=0.15)
    yaml_path = trainer.create_yaml_config("prepared_data/data.yaml")
    
    best_weights, results = trainer.train_final_model(
        data_yaml=yaml_path,
        teacher_weights="C:/Users/Niloy/Downloads/object detection/codes/teacher_model/teacher/weights/best.pt",
        epochs=150,
        batch=16,
        device='cpu'
    )
    
    final_metrics = trainer.evaluate(best_weights, yaml_path, device='cpu')
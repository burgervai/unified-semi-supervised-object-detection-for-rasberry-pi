import os
import shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

class DataPreparation:
    def __init__(self, output_dir="prepared_data"):
        self.output_dir = Path(output_dir)
        self.labeled_dir = self.output_dir / "labeled"
        self.unlabeled_dir = self.output_dir / "unlabeled"
        self.temp_dir = self.output_dir / "temp"  # Temporary staging area
        
        # Create directories
        self.labeled_dir.mkdir(parents=True, exist_ok=True)
        self.unlabeled_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def consolidate_labeled_data(self, dataset_paths, class_mappings, train_ratio=0.8):
        """Consolidate multiple datasets and directly organize into train/val splits"""
        
        # Create unified class mapping
        unified_classes = {}
        class_id = 0
        
        for dataset_name, classes in class_mappings.items():
            for cls in classes:
                if cls not in unified_classes:
                    unified_classes[cls] = class_id
                    class_id += 1
        
        print(f"Unified classes: {unified_classes}")
        
        # Collect all image-label pairs in temporary location
        temp_images = self.temp_dir / "images"
        temp_labels = self.temp_dir / "labels"
        temp_images.mkdir(exist_ok=True)
        temp_labels.mkdir(exist_ok=True)
        
        all_image_paths = []
        
        for dataset_name, dataset_path in dataset_paths.items():
            dataset_path = Path(dataset_path)
            old_to_new = {i: unified_classes[cls] for i, cls in enumerate(class_mappings[dataset_name])}
            
            print(f"\nProcessing {dataset_name}...")
            print(f"Class mapping: {old_to_new}")
            
            # Search for images in train, valid, and test splits
            for split in ["train", "valid", "test"]:
                image_source_dir = dataset_path / split / "images"
                label_source_dir = dataset_path / split / "labels"
                
                if not image_source_dir.exists():
                    print(f"  Skipping {split} - directory not found")
                    continue
                
                images_found = 0
                for img_path in image_source_dir.glob("*"):
                    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                        continue
                        
                    label_path = label_source_dir / f"{img_path.stem}.txt"
                    
                    if label_path.exists():
                        # Create unique filename
                        new_img_name = f"{dataset_name}_{split}_{img_path.name}"
                        new_label_name = f"{dataset_name}_{split}_{img_path.stem}.txt"
                        
                        new_img_path = temp_images / new_img_name
                        new_label_path = temp_labels / new_label_name
                        
                        # Copy image
                        shutil.copy(img_path, new_img_path)
                        
                        # Convert and copy label
                        with open(label_path, 'r') as f_in, open(new_label_path, 'w') as f_out:
                            for line in f_in:
                                parts = line.strip().split()
                                if parts:
                                    old_id = int(parts[0])
                                    if old_id not in old_to_new:
                                        print(f"  WARNING: Class ID {old_id} not in mapping for {dataset_name}")
                                        print(f"           Found in: {img_path.name}")
                                        print(f"           Available mappings: {old_to_new}")
                                        continue  # Skip this annotation
                                    new_id = old_to_new[old_id]
                                    f_out.write(f"{new_id} {' '.join(parts[1:])}\n")
                        
                        all_image_paths.append(new_img_path)
                        images_found += 1
                
                print(f"  {split}: {images_found} images processed")
        
        print(f"\nTotal images collected: {len(all_image_paths)}")
        
        # Split into train and validation
        if len(all_image_paths) == 0:
            raise ValueError("No images found! Check your dataset paths.")
        
        train_imgs, val_imgs = train_test_split(
            all_image_paths, 
            train_size=train_ratio, 
            random_state=42
        )
        
        print(f"Train set: {len(train_imgs)} images")
        print(f"Val set: {len(val_imgs)} images")
        
        # Create train and val directories
        train_dir = self.labeled_dir / "train"
        val_dir = self.labeled_dir / "val"
        
        for split_dir in [train_dir, val_dir]:
            (split_dir / "images").mkdir(parents=True, exist_ok=True)
            (split_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # Move files to final locations
        for img_path in train_imgs:
            label_path = temp_labels / f"{img_path.stem}.txt"
            shutil.move(str(img_path), train_dir / "images" / img_path.name)
            if label_path.exists():
                shutil.move(str(label_path), train_dir / "labels" / label_path.name)
        
        for img_path in val_imgs:
            label_path = temp_labels / f"{img_path.stem}.txt"
            shutil.move(str(img_path), val_dir / "images" / img_path.name)
            if label_path.exists():
                shutil.move(str(label_path), val_dir / "labels" / label_path.name)
        
        # Clean up temporary directory
        shutil.rmtree(self.temp_dir)
        
        return unified_classes
    
    def prepare_unlabeled_data(self, unlabeled_source_path):
        """Copy unlabeled images to unlabeled directory"""
        unlabeled_source = Path(unlabeled_source_path)
        images_dir = self.unlabeled_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        count = 0
        for img_path in unlabeled_source.glob("**/*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                shutil.copy(img_path, images_dir / img_path.name)
                count += 1
        
        print(f"\nUnlabeled images copied: {count}")
    
    def create_yaml_config(self, unified_classes):
        """Create YAML configuration file for YOLO training"""
        
        config = {
            'path': str(self.labeled_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(unified_classes),
            'names': {v: k for k, v in unified_classes.items()}
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\nYAML config saved to: {yaml_path}")
        print(f"Classes: {config['names']}")
        
        return yaml_path
    
    def get_dataset_stats(self):
        """Print statistics about the prepared dataset"""
        train_imgs = list((self.labeled_dir / "train" / "images").glob("*"))
        val_imgs = list((self.labeled_dir / "val" / "images").glob("*"))
        unlabeled_imgs = list((self.unlabeled_dir / "images").glob("*"))
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Training images: {len(train_imgs)}")
        print(f"Validation images: {len(val_imgs)}")
        print(f"Unlabeled images: {len(unlabeled_imgs)}")
        print(f"Total labeled: {len(train_imgs) + len(val_imgs)}")
        print("="*50)


if __name__ == "__main__":
    # Initialize data preparation
    prep = DataPreparation(output_dir="prepared_data")
    
    # Define your datasets
    dataset_paths = {
        "dataset1": "C:/Users/Niloy/Downloads/object detection/dataset with lebel/indoor obstacles.v1i.yolov8"
    }
    
    # Define class mappings for each dataset
    # Order matters! Index must match the class ID in the dataset
    class_mappings = {
        "dataset1": [
            "Chair",              # class 0 - 284 instances
            "Desk",               # class 1 - 157 instances
            "Door",               # class 2 - 72 instances
            "Elevator",           # class 3 - 6 instances
            "Low Obstacle",       # class 4 - 74 instances
            "Wall",               # class 5 - 60 instances
            "person",             # class 6 - 31 instances
            "stairs",             # class 7 - 5 instances
            # "unknown obstacle"    # class 8 - 50 instances
            "unknown obstacle"
        ],
        # Add more datasets as needed:
        # "dataset2": ["door", "stair", "chair", "window"]
    }
    
    # Process labeled data (includes train/val split)
    print("Starting data preparation...")
    unified_classes = prep.consolidate_labeled_data(
        dataset_paths, 
        class_mappings, 
        train_ratio=0.8
    )
    
    # Process unlabeled data
    prep.prepare_unlabeled_data(
        "C:/Users/Niloy/Downloads/object detection/dataset without lebels/obstacles dataset"
    )
    
    # Create YAML configuration
    yaml_path = prep.create_yaml_config(unified_classes)
    
    # Show statistics
    prep.get_dataset_stats()
    
    print(f"\n✓ Data preparation complete!")
    print(f"✓ Use '{yaml_path}' for YOLO training")
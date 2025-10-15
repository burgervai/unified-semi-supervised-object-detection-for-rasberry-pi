import os
from pathlib import Path
from collections import defaultdict

def inspect_dataset_classes(dataset_path):
    """Inspect a YOLO dataset to find all class IDs used"""
    dataset_path = Path(dataset_path)
    
    class_ids = set()
    class_counts = defaultdict(int)
    file_count = 0
    
    print(f"\nInspecting dataset: {dataset_path.name}")
    print("="*60)
    
    # Search through all splits
    for split in ["train", "valid", "test"]:
        label_dir = dataset_path / split / "labels"
        
        if not label_dir.exists():
            print(f"  {split}: Directory not found")
            continue
        
        label_files = list(label_dir.glob("*.txt"))
        print(f"  {split}: {len(label_files)} label files")
        
        for label_file in label_files:
            file_count += 1
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_ids.add(class_id)
                        class_counts[class_id] += 1
    
    print(f"\n  Total label files processed: {file_count}")
    print(f"  Unique class IDs found: {sorted(class_ids)}")
    print(f"  Number of classes: {len(class_ids)}")
    
    print("\n  Class distribution:")
    for class_id in sorted(class_ids):
        print(f"    Class {class_id}: {class_counts[class_id]} instances")
    
    # Check for data.yaml
    yaml_path = dataset_path / "data.yaml"
    if yaml_path.exists():
        print(f"\n  Found data.yaml - checking class names...")
        import yaml
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    print(f"    Class names: {data['names']}")
                    if isinstance(data['names'], dict):
                        for idx, name in sorted(data['names'].items()):
                            print(f"      {idx}: {name}")
                    elif isinstance(data['names'], list):
                        for idx, name in enumerate(data['names']):
                            print(f"      {idx}: {name}")
        except Exception as e:
            print(f"    Error reading yaml: {e}")
    
    print("="*60)
    return sorted(class_ids)


if __name__ == "__main__":
    # Inspect your dataset
    dataset_path = "C:/Users/Niloy/Downloads/object detection/dataset with lebel/indoor obstacles.v1i.yolov8"
    
    class_ids = inspect_dataset_classes(dataset_path)
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    print("Update your class_mappings to include all classes:")
    print(f'class_mappings = {{')
    print(f'    "dataset1": [')
    for i, class_id in enumerate(class_ids):
        print(f'        "class_{class_id}",  # Replace with actual class name')
    print(f'    ]')
    print(f'}}')
    print("="*60)
from ultralytics import YOLO
from pathlib import Path
import cv2, torch
from tqdm import tqdm

class PseudoLabeler:
    def __init__(self, teacher_weights, unlabeled_dir, output_dir="pseudo_labeled"):
        self.model = YOLO(teacher_weights)
        self.unlabeled_dir = Path(unlabeled_dir)
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.background_dir = self.output_dir / "background" / "images"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.background_dir.mkdir(parents=True, exist_ok=True)

    def generate_pseudo_labels(self, conf_threshold=0.85, iou_threshold=0.45,
                               device=None, batch_size=32, background_ratio=0.1):
        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'

        image_paths = list(self.unlabeled_dir.glob("**/*.jpg")) \
                    + list(self.unlabeled_dir.glob("**/*.jpeg")) \
                    + list(self.unlabeled_dir.glob("**/*.png"))
        total_images = len(image_paths)
        images_with_detections = 0
        background_count = 0
        max_backgrounds = int(total_images * background_ratio)

        for i in tqdm(range(0, total_images, batch_size), desc="Generating Pseudo-Labels"):
            batch_paths = image_paths[i:i+batch_size]
            results = self.model.predict(
                batch_paths,
                conf=conf_threshold,
                iou=iou_threshold,
                device=device,
                verbose=False
            )

            for img_path, result in zip(batch_paths, results):
                if result.boxes is not None and len(result.boxes) > 0:
                    self._save_pseudo_label(img_path, result)
                    images_with_detections += 1
                elif background_count < max_backgrounds:
                    self._save_background_image(img_path)
                    background_count += 1

        return {
            'total_images': total_images,
            'pseudo_labeled': images_with_detections,
            'background_images': background_count
        }

    def _save_pseudo_label(self, img_path, result):
        img = cv2.imread(str(img_path))
        if img is None:
            return
        cv2.imwrite(str(self.images_dir / img_path.name), img)

        with open(self.labels_dir / f"{img_path.stem}.txt", 'w') as f:
            for xywhn, cls in zip(result.boxes.xywhn, result.boxes.cls):
                x_center, y_center, width, height = xywhn.tolist()
                f.write(f"{int(cls)} {x_center} {y_center} {width} {height}\n")

    def _save_background_image(self, img_path):
        img = cv2.imread(str(img_path))
        if img is None:
            return
        cv2.imwrite(str(self.background_dir / img_path.name), img)


if __name__ == "__main__":
    labeler = PseudoLabeler(
        teacher_weights=r"C:\Users\Niloy\Downloads\object detection\codes\teacher_model\teacher\weights\best.pt",
        unlabeled_dir=r"C:\Users\Niloy\Downloads\object detection\dataset without lebels\obstacles dataset"
    )

    stats = labeler.generate_pseudo_labels(conf_threshold=0.85, batch_size=32, background_ratio=0.1)
    print(stats)

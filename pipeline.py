import torch
from pathlib import Path
import numpy as np
import json
from datetime import datetime

class SemiSupervisedPipeline:
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def run_phase1(self):
        from data_prep import DataPreparation
        
        prep = DataPreparation(output_dir=self.config['output_base'] / "prepared_data")
        
        unified_classes = prep.consolidate_labeled_data(
            self.config['dataset_paths'],
            self.config['class_mappings']
        )
        
        prep.prepare_unlabeled_data(self.config['unlabeled_source'])
        
        yaml_path = prep.create_yaml_config(unified_classes)
        
        self.results['phase1'] = {
            'unified_classes': unified_classes,
            'yaml_path': str(yaml_path),
            'num_classes': len(unified_classes)
        }
        
        return yaml_path
    
    def run_phase2(self, data_yaml):
        from teacher_train import TeacherTrainer
        
        trainer = TeacherTrainer(
            data_yaml=data_yaml,
            model_name=self.config['base_model'],
            output_dir=self.config['output_base'] / "teacher_model"
        )
        
        best_weights, results = trainer.train(
            epochs=self.config['teacher_epochs'],
            imgsz=self.config['imgsz'],
            batch=self.config['batch_size'],
            patience=self.config['patience'],
            device=self.config['device']
        )
        
        metrics = trainer.evaluate(best_weights, device=self.config['device'])
        
        self.results['phase2'] = {
            'teacher_weights': str(best_weights),
            'metrics': metrics
        }
        
        return best_weights
    
    def run_phase3(self, teacher_weights):
        from pseudo import PseudoLabeler
        
        labeler = PseudoLabeler(
            teacher_weights=teacher_weights,
            unlabeled_dir=self.config['output_base'] / "prepared_data" / "unlabeled",
            output_dir=self.config['output_base'] / "pseudo_labeled"
        )
        
        stats = labeler.generate_pseudo_labels(
            conf_threshold=self.config['pseudo_conf_threshold'],
            iou_threshold=self.config['pseudo_iou_threshold'],
            device=self.config['device'],
            batch_size=self.config['batch_size'],
            background_ratio=self.config['background_ratio']
        )
        
        self.results['phase3'] = stats
        
        return self.config['output_base'] / "pseudo_labeled"
    
    def run_phase4(self, pseudo_labeled_dir, original_yaml):
        from unified import UnifiedTrainer
        
        trainer = UnifiedTrainer(
            labeled_dir=self.config['output_base'] / "prepared_data" / "labeled",
            pseudo_labeled_dir=pseudo_labeled_dir,
            background_dir=pseudo_labeled_dir / "background",
            output_dir=self.config['output_base'] / "final_model"
        )
        
        train_count, val_count = trainer.combine_datasets(val_ratio=self.config['val_ratio'])
        
        yaml_path = trainer.create_yaml_config(original_yaml)
        
        best_weights, results = trainer.train_final_model(
            data_yaml=yaml_path,
            teacher_weights=self.results['phase2']['teacher_weights'],
            epochs=self.config['final_epochs'],
            imgsz=self.config['imgsz'],
            batch=self.config['batch_size'],
            patience=self.config['final_patience'],
            device=self.config['device']
        )
        
        final_metrics = trainer.evaluate(best_weights, yaml_path, device=self.config['device'])
        
        self.results['phase4'] = {
            'train_samples': train_count,
            'val_samples': val_count,
            'final_weights': str(best_weights),
            'metrics': final_metrics
        }
        
        return best_weights
    
    def run_full_pipeline(self):
        start_time = datetime.now()
        
        yaml_path = self.run_phase1()
        teacher_weights = self.run_phase2(yaml_path)
        pseudo_labeled_dir = self.run_phase3(teacher_weights)
        final_weights = self.run_phase4(pseudo_labeled_dir, yaml_path)
        
        end_time = datetime.now()
        self.results['total_time'] = str(end_time - start_time)
        
        results_path = self.config['output_base'] / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4, cls=NumpyEncoder)
        
        return final_weights, self.results
    
    def compare_metrics(self):
        teacher_metrics = self.results['phase2']['metrics']
        final_metrics = self.results['phase4']['metrics']
        
        # Ensure metrics are numpy arrays for subtraction
        t_precision = np.array(teacher_metrics['precision'])
        f_precision = np.array(final_metrics['precision'])
        t_recall = np.array(teacher_metrics['recall'])
        f_recall = np.array(final_metrics['recall'])
        
        comparison = {
            'teacher_model': teacher_metrics,
            'final_model': final_metrics,
            'improvements': {
                'mAP50': final_metrics['mAP50'] - teacher_metrics['mAP50'],
                'mAP50-95': final_metrics['mAP50-95'] - teacher_metrics['mAP50-95'],
                'precision': (f_precision - t_precision).tolist(),
                'recall': (f_recall - t_recall).tolist()
            }
        }
        
        return comparison

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    config = {
        'output_base': Path("./semi_supervised_output"),
        # Updated paths from your other scripts
        'dataset_paths': {
            "dataset1": r"C:\Users\Niloy\Downloads\object detection\dataset with lebel\indoor obstacles.v1i.yolov8"
        },
        # Updated class mappings from data_prep.py
        'class_mappings': {
            "dataset1": [
                "Chair",
                "Desk",
                "Door",
                "Elevator",
                "Low Obstacle",
                "Wall",
                "person",
                "stairs",
                "unknown obstacle"
            ],
        },
        'unlabeled_source': r"C:\Users\Niloy\Downloads\object detection\dataset without lebels\obstacles dataset",
        'base_model': "yolo11n.pt",
        'device': 'cpu',  # Changed from 0 to 'cpu'
        'imgsz': 640,
        'batch_size': 16,
        'train_ratio': 0.8,
        'val_ratio': 0.15,
        'teacher_epochs': 1, # Reduced for a quick test run
        'patience': 20,
        'final_epochs': 1, # Reduced for a quick test run
        'final_patience': 30,
        'pseudo_conf_threshold': 0.85,
        'pseudo_iou_threshold': 0.45,
        'background_ratio': 0.1
    }
    
    pipeline = SemiSupervisedPipeline(config)
    final_weights, results = pipeline.run_full_pipeline()
    comparison = pipeline.compare_metrics()
import argparse
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from config.config import get_config, Config
from src.data.data_loader import DataLoader
from src.features.feature_selector import FeatureSelectionPipeline
from src.models.ensemble_model import EnsembleModel


class HARPipeline:
    """
    Main Human Activity Recognition Pipeline.
    
    Implements Facade Pattern: provides a simple interface to complex subsystems.
    Follows Single Responsibility: orchestrates the entire ML pipeline.
    """
    
    def __init__(self, config: Config, output_dir: str = "results"):
        """
        Initialize the HAR pipeline.
        
        Args:
            config (Config): Configuration object
            output_dir (str): Directory to save results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.data_loader = DataLoader(config.data)
        self.feature_pipeline = FeatureSelectionPipeline(config.feature_selection)
        self.ensemble_model = EnsembleModel(config.model)
        
        # Data storage
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_selected = None
        self.X_test_selected = None
    
    def load_data(self) -> None:
        """Load and prepare data."""
        print("=" * 50)
        print("STEP 1: Loading Data")
        print("=" * 50)
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_loader.load_training_data()
        
        # Generate activity distribution plot
        self._plot_activity_distribution()
    
    def select_features(self) -> None:
        """Apply feature selection pipeline."""
        print("=" * 50)
        print("STEP 2: Feature Selection")
        print("=" * 50)
        
        self.X_train_selected = self.feature_pipeline.fit_transform(self.X_train, self.y_train)
        self.X_test_selected = self.feature_pipeline.transform(self.X_test)
        
        # Generate feature selection summary
        self._plot_feature_selection_summary()
    
    def train_model(self) -> None:
        """Train the ensemble model."""
        print("=" * 50)
        print("STEP 3: Model Training")
        print("=" * 50)
        
        self.ensemble_model.train(self.X_train_selected, self.y_train)
        
        # Save the trained model
        self._save_model()
    
    def evaluate_model(self) -> dict:
        """Evaluate the model and generate reports."""
        print("=" * 50)
        print("STEP 4: Model Evaluation")
        print("=" * 50)
        
        # Evaluate on test set
        results = self.ensemble_model.evaluate(self.X_test_selected, self.y_test)
        
        print(f"Test Accuracy: {results['accuracy']:.4f}")
        print("\\nIndividual Model Scores:")
        for model_name, score in self.ensemble_model.get_model_scores().items():
            print(f"{model_name.upper()}: {score:.4f}")
        
        # Generate evaluation plots and reports
        self._generate_evaluation_report(results)
        
        return results
    
    def evaluate_on_external_data(self) -> dict:
        """Evaluate on external test data."""
        print("=" * 50)
        print("STEP 5: External Data Evaluation")
        print("=" * 50)
        
        # Load external test data
        X_test_external, y_test_external = self.data_loader.load_test_data()
        
        # Apply feature selection
        X_test_external_selected = self.feature_pipeline.transform(X_test_external)
        
        # Evaluate
        y_pred_external = self.ensemble_model.predict(X_test_external_selected)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        external_results = {
            'accuracy': accuracy_score(y_test_external, y_pred_external),
            'predictions': y_pred_external,
            'confusion_matrix': confusion_matrix(y_test_external, y_pred_external),
            'classification_report': classification_report(y_test_external, y_pred_external)
        }
        
        print(f"External Test Accuracy: {external_results['accuracy']:.4f}")
        
        # Generate external evaluation report
        self._generate_external_evaluation_report(external_results)
        
        return external_results
    
    def run_full_pipeline(self) -> dict:
        """Run the complete training and evaluation pipeline."""
        print("Starting Human Activity Recognition Pipeline...")
        
        # Execute pipeline steps
        self.load_data()
        self.select_features()
        self.train_model()
        results = self.evaluate_model()
        external_results = self.evaluate_on_external_data()
        
        print("=" * 50)
        print("Pipeline completed successfully!")
        print(f"Results saved to: {self.output_dir}")
        print("=" * 50)
        
        return {'internal_test': results, 'external_test': external_results}
    
    def load_pretrained_model(self, model_path: str) -> None:
        """Load a pretrained model for evaluation only."""
        print(f"Loading pretrained model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        self.ensemble_model = saved_data['model']
        self.feature_pipeline = saved_data['feature_pipeline']
        
        print("Pretrained model loaded successfully!")
    
    def _plot_activity_distribution(self) -> None:
        """Plot activity distribution for each subject."""
        # Reconstruct the original data for plotting
        data = pd.read_csv(self.config.data.train_data_path)
        
        plt.figure(figsize=self.config.plot.figure_size_large)
        sns.countplot(x='subject', hue='Activity', data=data)
        plt.title('Activity Count for Each Subject')
        plt.xlabel('Subject')
        plt.ylabel('Activity Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = self.output_dir / 'activity_distribution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Activity distribution plot saved to: {plot_path}")
    
    def _plot_feature_selection_summary(self) -> None:
        """Plot feature selection summary."""
        feature_counts = self.feature_pipeline.get_feature_counts()
        
        stages = ['Original', 'After Correlation', 'After Variance', 'After RFE']
        counts = [
            self.X_train.shape[1],
            self.X_train.shape[1] - feature_counts['correlated_features_removed'],
            feature_counts['variance_features_selected'],
            feature_counts['rfe_features_selected']
        ]
        
        plt.figure(figsize=self.config.plot.figure_size_medium)
        plt.bar(stages, counts, color=['blue', 'orange', 'green', 'red'])
        plt.title('Feature Selection Pipeline Progress')
        plt.xlabel('Selection Stage')
        plt.ylabel('Number of Features')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            plt.text(i, v + 5, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'feature_selection_summary.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature selection summary saved to: {plot_path}")
    
    def _generate_evaluation_report(self, results: dict) -> None:
        """Generate comprehensive evaluation report."""
        # Confusion matrix heatmap
        plt.figure(figsize=self.config.plot.figure_size_medium)
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.data_loader.label_encoder.classes_,
            yticklabels=self.data_loader.label_encoder.classes_
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - Internal Test Set')
        plt.tight_layout()
        
        plot_path = self.output_dir / 'confusion_matrix_internal.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed report
        report_path = self.output_dir / 'evaluation_report_internal.txt'
        with open(report_path, 'w') as f:
            f.write("Human Activity Recognition - Internal Test Results\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"Test Accuracy: {results['accuracy']:.4f}\\n\\n")
            f.write("Individual Model Scores:\\n")
            for model_name, score in self.ensemble_model.get_model_scores().items():
                f.write(f"{model_name.upper()}: {score:.4f}\\n")
            f.write("\\n" + "=" * 50 + "\\n")
            f.write("Classification Report:\\n")
            f.write(results['classification_report'])
        
        print(f"Internal evaluation report saved to: {report_path}")
    
    def _generate_external_evaluation_report(self, results: dict) -> None:
        """Generate external evaluation report."""
        # Confusion matrix heatmap
        plt.figure(figsize=self.config.plot.figure_size_medium)
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Greens',
            xticklabels=self.data_loader.label_encoder.classes_,
            yticklabels=self.data_loader.label_encoder.classes_
        )
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix - External Test Set')
        plt.tight_layout()
        
        plot_path = self.output_dir / 'confusion_matrix_external.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed report
        report_path = self.output_dir / 'evaluation_report_external.txt'
        with open(report_path, 'w') as f:
            f.write("Human Activity Recognition - External Test Results\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(f"External Test Accuracy: {results['accuracy']:.4f}\\n\\n")
            f.write("=" * 50 + "\\n")
            f.write("Classification Report:\\n")
            f.write(results['classification_report'])
        
        print(f"External evaluation report saved to: {report_path}")
    
    def _save_model(self) -> None:
        """Save the trained model and feature pipeline."""
        model_data = {
            'model': self.ensemble_model,
            'feature_pipeline': self.feature_pipeline,
            'config': self.config
        }
        
        model_path = self.output_dir / 'har_ensemble_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {model_path}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Human Activity Recognition Pipeline')
    
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory containing train.csv (default: data)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results (default: results)'
    )
    
    parser.add_argument(
        '--evaluate_only',
        action='store_true',
        help='Only evaluate using pretrained model'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        help='Path to pretrained model (required if --evaluate_only)'
    )
    
    parser.add_argument(
        '--correlation_threshold',
        type=float,
        default=0.8,
        help='Correlation threshold for feature selection (default: 0.8)'
    )
    
    parser.add_argument(
        '--variance_threshold',
        type=float,
        default=0.04,
        help='Variance threshold for feature selection (default: 0.04)'
    )
    
    parser.add_argument(
        '--rfe_features',
        type=int,
        default=50,
        help='Number of features to select with RFE (default: 50)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.evaluate_only and not args.model_path:
        parser.error("--model_path is required when using --evaluate_only")
    
    # Load configuration
    config = get_config()
    
    # Override config with command line arguments
    config.data.train_data_path = os.path.join(args.data_dir, 'train.csv')
    config.data.test_data_path = os.path.join(args.data_dir, 'test.csv')  
    config.feature_selection.correlation_threshold = args.correlation_threshold
    config.feature_selection.variance_threshold = args.variance_threshold
    config.feature_selection.rfe_n_features = args.rfe_features
    
    # Initialize pipeline
    pipeline = HARPipeline(config, args.output_dir)
    
    try:
        if args.evaluate_only:
            # Load pretrained model and evaluate
            pipeline.load_pretrained_model(args.model_path)
            pipeline.load_data()
            pipeline.select_features()
            validation_results = pipeline.evaluate_model()
            holdout_results = pipeline.evaluate_on_holdout_test()
            
            print(f"\\nEvaluation completed!")
            print(f"Validation Accuracy: {validation_results['accuracy']:.4f}")
            print(f"Holdout Test Accuracy: {holdout_results['accuracy']:.4f}")
            
        else:
            # Run full training pipeline
            results = pipeline.run_full_pipeline()
            
            print(f"\\nTraining completed!")
            print(f"Internal Test Accuracy: {results['internal_test']['accuracy']:.4f}")
            print(f"External Test Accuracy: {results['external_test']['accuracy']:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
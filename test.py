# test_ensemble.py  
from config.config import get_config
from src.data.data_loader import DataLoader
from src.features.feature_selector import FeatureSelectionPipeline
from src.models.ensemble_model import EnsembleModel

# Complete pipeline test
config = get_config()

# Load and prepare data
data_loader = DataLoader(config.data)
X_train, X_test, y_train, y_test = data_loader.load_training_data()

# Feature selection
feature_pipeline = FeatureSelectionPipeline(config.feature_selection)
X_train_selected = feature_pipeline.fit_transform(X_train, y_train)
X_test_selected = feature_pipeline.transform(X_test)

# Train ensemble
ensemble = EnsembleModel(config.model)
ensemble.train(X_train_selected, y_train)

# Evaluate
results = ensemble.evaluate(X_test_selected, y_test)
print(f"Test Accuracy: {results['accuracy']:.4f}")
print("Model scores:", ensemble.get_model_scores())
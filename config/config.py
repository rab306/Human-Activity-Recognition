"""
Configuration file for Human Activity Recognition Pipeline
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


@dataclass
class DataConfig:
    """Data-related configuration"""
    train_data_path: str = 'raw_data/train.csv'
    test_data_path: str = 'raw_data/test.csv'  # True holdout test set
    target_column: str = 'Activity'
    subject_column: str = 'subject'
    test_size: float = 0.2  # For train/validation split
    random_state: int = 42


@dataclass
class FeatureSelectionConfig:
    """Feature selection parameters"""
    correlation_threshold: float = 0.8
    variance_threshold: float = 0.04
    rfe_n_features: int = 50
    rfe_random_state: int = 42


@dataclass
class ModelConfig:
    """Model hyperparameter grids and settings"""
    # XGBoost parameters
    xgb_param_grid: Dict[str, List] = None
    
    # SVC parameters  
    svc_param_grid: Dict[str, List] = None
    
    # Logistic Regression parameters
    lr_param_grid: Dict[str, List] = None
    
    # Random Forest parameters
    rf_param_grid: Dict[str, List] = None
    
    # RandomizedSearchCV settings
    n_iter: int = 30
    cv_folds: int = 5
    scoring: str = 'accuracy'
    n_jobs: int = -1
    verbose: int = 2
    search_random_state: int = 42
    
    # Voting classifier weights
    voting_weights: List[float] = None
    voting_type: str = 'soft'

    def __post_init__(self):
        """Initialize parameter grids if not provided"""
        if self.xgb_param_grid is None:
            self.xgb_param_grid = {
                'classifier__learning_rate': [0.3, 0.4, 0.5],
                'classifier__n_estimators': [50, 100, 150]
            }
        
        if self.svc_param_grid is None:
            self.svc_param_grid = {
                'classifier__C': [0.1, 1, 10],
                'classifier__gamma': [1, 0.1, 0.01],
            }
        
        if self.lr_param_grid is None:
            self.lr_param_grid = {
                'classifier__C': [0.1, 1, 10],
            }
        
        if self.rf_param_grid is None:
            self.rf_param_grid = {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__min_samples_split': [2, 4, 6],
            }
        
        if self.voting_weights is None:
            self.voting_weights = [2.2, 2, 1.6, 1.8]  # [xgb, svc, lr, rf]


@dataclass
class PlotConfig:
    """Plotting configuration"""
    figure_size_large: Tuple[int, int] = (12, 6)
    figure_size_medium: Tuple[int, int] = (10, 8)
    histogram_bins: int = 500
    random_state: int = 42


@dataclass
class Config:
    """Main configuration class combining all configs"""
    data: DataConfig = None
    feature_selection: FeatureSelectionConfig = None
    model: ModelConfig = None
    plot: PlotConfig = None
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.data is None:
            self.data = DataConfig()
        if self.feature_selection is None:
            self.feature_selection = FeatureSelectionConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.plot is None:
            self.plot = PlotConfig()


def get_config() -> Config:
    """Factory function to get configuration instance"""
    return Config()


# For easy importing
config = get_config()
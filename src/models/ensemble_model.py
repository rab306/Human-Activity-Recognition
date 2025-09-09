import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config.config import ModelConfig


class ModelFactory:
    """
    Factory class for creating and configuring machine learning models.
    
    Implements Factory Pattern: centralizes model creation logic.
    Follows Open/Closed Principle: easy to add new models without modifying existing code.
    """
    
    @staticmethod
    def create_xgboost(random_state: int = 42) -> xgb.XGBClassifier:
        """Create XGBoost classifier."""
        return xgb.XGBClassifier(
            objective='multi:softmax', 
            num_class=6, 
            random_state=random_state
        )
    
    @staticmethod
    def create_svc(random_state: int = 42) -> SVC:
        """Create SVM classifier."""
        return SVC(probability=True, random_state=random_state)
    
    @staticmethod
    def create_logistic_regression(random_state: int = 42) -> LogisticRegression:
        """Create Logistic Regression classifier."""
        return LogisticRegression(multi_class='auto', random_state=random_state)
    
    @staticmethod
    def create_random_forest(random_state: int = 42) -> RandomForestClassifier:
        """Create Random Forest classifier."""
        return RandomForestClassifier(random_state=random_state)
    
    @staticmethod
    def create_pipeline(model, needs_scaling: bool = False) -> Pipeline:
        """
        Create sklearn pipeline with optional scaling.
        
        Args:
            model: The classifier to wrap
            needs_scaling (bool): Whether to add StandardScaler
            
        Returns:
            Pipeline: Configured pipeline
        """
        if needs_scaling:
            return Pipeline([('scaler', StandardScaler()), ('classifier', model)])
        else:
            return Pipeline([('classifier', model)])


class HyperparameterOptimizer:
    """
    Handles hyperparameter optimization for multiple models.
    
    Follows Single Responsibility Principle: only responsible for hyperparameter tuning.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize optimizer with configuration.
        
        Args:
            config (ModelConfig): Model configuration with parameter grids
        """
        self.config = config
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}
    
    def optimize_model(self, name: str, pipeline: Pipeline, param_grid: Dict) -> Pipeline:
        """
        Optimize hyperparameters for a single model.
        
        Args:
            name (str): Model name for logging
            pipeline (Pipeline): Model pipeline to optimize
            param_grid (Dict): Parameter grid for search
            
        Returns:
            Pipeline: Best model pipeline
        """
        print(f"Optimizing {name}...")
        
        # Create RandomizedSearchCV
        random_search = RandomizedSearchCV(
            pipeline, 
            param_grid,
            n_iter=self.config.n_iter,
            cv=self.config.cv_folds,
            scoring=self.config.scoring,
            n_jobs=self.config.n_jobs,
            verbose=self.config.verbose,
            random_state=self.config.search_random_state
        )
        
        return random_search
    
    def optimize_all_models(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Optimize hyperparameters for all models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            Dict: Dictionary containing optimized models and results
        """
        # Create models and pipelines
        models = {
            'xgb': (ModelFactory.create_xgboost(), False, self.config.xgb_param_grid),
            'svc': (ModelFactory.create_svc(), True, self.config.svc_param_grid),
            'lr': (ModelFactory.create_logistic_regression(), True, self.config.lr_param_grid),
            'rf': (ModelFactory.create_random_forest(), False, self.config.rf_param_grid)
        }
        
        optimized_searches = {}
        
        # Optimize each model
        for name, (model, needs_scaling, param_grid) in models.items():
            pipeline = ModelFactory.create_pipeline(model, needs_scaling)
            search = self.optimize_model(name, pipeline, param_grid)
            search.fit(X_train, y_train)
            optimized_searches[name] = search
            
            # Store results
            self.best_scores[name] = search.best_score_
            self.best_params[name] = self._remove_classifier_prefix(search.best_params_)
            
            print(f"{name.upper()} Best Score: {search.best_score_:.4f}")
        
        return optimized_searches
    
    def _remove_classifier_prefix(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Remove 'classifier__' prefix from parameter names."""
        return {key.split('__')[1]: value for key, value in params.items()}
    
    def get_best_models(self) -> Dict[str, Any]:
        """
        Create models with best parameters found during optimization.
        
        Returns:
            Dict: Dictionary of best models
        """
        best_models = {}
        
        if 'xgb' in self.best_params:
            best_models['xgb'] = ModelFactory.create_xgboost()
            best_models['xgb'].set_params(**self.best_params['xgb'])
        
        if 'svc' in self.best_params:
            best_models['svc'] = ModelFactory.create_svc()
            best_models['svc'].set_params(**self.best_params['svc'])
        
        if 'lr' in self.best_params:
            best_models['lr'] = ModelFactory.create_logistic_regression()
            best_models['lr'].set_params(**self.best_params['lr'])
        
        if 'rf' in self.best_params:
            best_models['rf'] = ModelFactory.create_random_forest()
            best_models['rf'].set_params(**self.best_params['rf'])
        
        return best_models


class EnsembleModel:
    """
    Ensemble model using VotingClassifier.
    
    Follows Composition Pattern: combines multiple models.
    Implements Dependency Inversion: depends on abstractions (sklearn interface).
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize ensemble model.
        
        Args:
            config (ModelConfig): Model configuration
        """
        self.config = config
        self.optimizer = HyperparameterOptimizer(config)
        self.voting_classifier = None
        self.is_fitted = False
    
    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'EnsembleModel':
        """
        Train the ensemble model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            EnsembleModel: Fitted ensemble model
        """
        print("Starting ensemble model training...")
        
        # Optimize hyperparameters for all models
        self.optimizer.optimize_all_models(X_train, y_train)
        
        # Get best models
        best_models = self.optimizer.get_best_models()
        
        # Create voting classifier
        estimators = [
            ('xgb', best_models['xgb']),
            ('svc', best_models['svc']),
            ('lr', best_models['lr']),
            ('rf', best_models['rf'])
        ]
        
        self.voting_classifier = VotingClassifier(
            estimators=estimators,
            voting=self.config.voting_type,
            weights=self.config.voting_weights
        )
        
        # Fit the ensemble
        print("Training voting classifier...")
        self.voting_classifier.fit(X_train, y_train)
        
        self.is_fitted = True
        print("Ensemble model training completed!")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble model.
        
        Args:
            X (pd.DataFrame): Features to predict
            
        Returns:
            np.ndarray: Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.voting_classifier.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the ensemble model.
        
        Args:
            X_test (pd.DataFrame): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            Dict: Evaluation results
        """
        y_pred = self.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        return results
    
    def get_model_scores(self) -> Dict[str, float]:
        """Get the best cross-validation scores for each model."""
        return self.optimizer.best_scores.copy()
    
    def get_model_params(self) -> Dict[str, Dict[str, Any]]:
        """Get the best parameters for each model."""
        return self.optimizer.best_params.copy()
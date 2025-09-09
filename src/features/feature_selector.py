import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Set
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier

from config.config import FeatureSelectionConfig


class FeatureSelector(ABC):
    """
    Abstract base class for feature selection strategies.
    
    Implements Strategy Pattern: each concrete selector implements a different strategy.
    This follows Open/Closed Principle: open for extension, closed for modification.
    """
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'FeatureSelector':
        """Fit the feature selector on training data."""
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted selector."""
        pass
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
        """Fit and transform training data in one step."""
        return self.fit(X_train, y_train).transform(X_train)


class CorrelationSelector(FeatureSelector):
    """
    Removes highly correlated features using correlation matrix analysis.
    
    This implements the correlation-based feature selection from your original code.
    """
    
    def __init__(self, threshold: float = 0.8):
        """
        Initialize correlation selector.
        
        Args:
            threshold (float): Correlation threshold above which features are considered correlated
        """
        self.threshold = threshold
        self.correlated_features: Set[str] = set()
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray = None) -> 'CorrelationSelector':
        """
        Identify highly correlated features in training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Not used for correlation analysis
            
        Returns:
            CorrelationSelector: Fitted selector
        """
        # Compute correlation matrix
        correlation_matrix = X_train.corr()
        
        # Find highly correlated features
        self.correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > self.threshold:
                    colname = correlation_matrix.columns[i]
                    self.correlated_features.add(colname)
        
        print(f"CorrelationSelector: Identified {len(self.correlated_features)} highly correlated features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove highly correlated features from data.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with correlated features removed
        """
        X_transformed = X.drop(columns=self.correlated_features)
        print(f"CorrelationSelector: Shape after removing correlated features: {X_transformed.shape}")
        return X_transformed


class VarianceSelector(FeatureSelector):
    """
    Removes low-variance features using VarianceThreshold.
    
    This implements the variance-based feature selection from your original code.
    """
    
    def __init__(self, threshold: float = 0.04):
        """
        Initialize variance selector.
        
        Args:
            threshold (float): Variance threshold below which features are removed
        """
        self.threshold = threshold
        self.selector = VarianceThreshold(threshold=threshold)
        self.feature_names = None
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray = None) -> 'VarianceSelector':
        """
        Fit variance selector on training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Not used for variance analysis
            
        Returns:
            VarianceSelector: Fitted selector
        """
        self.selector.fit(X_train)
        
        # Store feature names for DataFrame output
        self.feature_names = X_train.columns[self.selector.get_support()].tolist()
        
        n_selected = len(self.feature_names)
        n_total = X_train.shape[1]
        print(f"VarianceSelector: Selected {n_selected} features out of {n_total}")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by removing low-variance features.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Features with low-variance features removed
        """
        X_transformed = self.selector.transform(X)
        X_df = pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
        print(f"VarianceSelector: Shape after removing low-variance features: {X_df.shape}")
        return X_df


class RFESelector(FeatureSelector):
    """
    Recursive Feature Elimination using RandomForest estimator.
    
    This implements the RFE-based feature selection from your original code.
    """
    
    def __init__(self, n_features: int = 50, random_state: int = 42):
        """
        Initialize RFE selector.
        
        Args:
            n_features (int): Number of features to select
            random_state (int): Random state for reproducibility
        """
        self.n_features = n_features
        self.random_state = random_state
        
        # Initialize estimator and RFE selector
        self.estimator = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        self.selector = RFE(self.estimator, n_features_to_select=n_features)
        self.feature_names = None
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'RFESelector':
        """
        Fit RFE selector on training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels (required for RFE)
            
        Returns:
            RFESelector: Fitted selector
        """
        self.selector.fit(X_train, y_train)
        
        # Store selected feature names
        self.feature_names = X_train.columns[self.selector.get_support()].tolist()
        
        print(f"RFESelector: Selected {len(self.feature_names)} features using RFE")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using RFE selection.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: RFE-selected features
        """
        X_transformed = self.selector.transform(X)
        X_df = pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
        print(f"RFESelector: Shape after RFE selection: {X_df.shape}")
        return X_df


class FeatureSelectionPipeline:
    """
    Pipeline that chains multiple feature selectors together.
    
    This demonstrates Composition Pattern: combining multiple strategies.
    Follows Single Responsibility: orchestrates feature selection process.
    """
    
    def __init__(self, config: FeatureSelectionConfig):
        """
        Initialize feature selection pipeline with configuration.
        
        Args:
            config (FeatureSelectionConfig): Feature selection configuration
        """
        self.config = config
        
        # Initialize selectors with config parameters
        self.correlation_selector = CorrelationSelector(config.correlation_threshold)
        self.variance_selector = VarianceSelector(config.variance_threshold)
        self.rfe_selector = RFESelector(config.rfe_n_features, config.rfe_random_state)
        
        self.is_fitted = False
    
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray) -> 'FeatureSelectionPipeline':
        """
        Fit all feature selectors in sequence.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            FeatureSelectionPipeline: Fitted pipeline
        """
        print("Starting feature selection pipeline...")
        print(f"Initial feature shape: {X_train.shape}")
        
        # Step 1: Remove correlated features
        X_temp = self.correlation_selector.fit_transform(X_train, y_train)
        
        # Step 2: Remove low-variance features
        X_temp = self.variance_selector.fit_transform(X_temp, y_train)
        
        # Step 3: Apply RFE
        self.rfe_selector.fit(X_temp, y_train)
        
        self.is_fitted = True
        print("Feature selection pipeline fitted successfully!")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data through all fitted selectors.
        
        Args:
            X (pd.DataFrame): Input features
            
        Returns:
            pd.DataFrame: Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        # Apply transformations in sequence
        X_transformed = self.correlation_selector.transform(X)
        X_transformed = self.variance_selector.transform(X_transformed)
        X_transformed = self.rfe_selector.transform(X_transformed)
        
        return X_transformed
    
    def fit_transform(self, X_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
        """
        Fit pipeline and transform training data.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            pd.DataFrame: Transformed training features
        """
        self.fit(X_train, y_train)
        return self.transform(X_train)
    
    def get_feature_counts(self) -> dict:
        """
        Get summary of features at each selection step.
        
        Returns:
            dict: Feature counts at each step
        """
        return {
            'correlated_features_removed': len(self.correlation_selector.correlated_features),
            'variance_features_selected': len(self.variance_selector.feature_names) if self.variance_selector.feature_names else 0,
            'rfe_features_selected': len(self.rfe_selector.feature_names) if self.rfe_selector.feature_names else 0
        }
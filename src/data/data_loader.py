"""
Data loading and initial validation module for Human Activity Recognition
"""
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from config.config import DataConfig


class DataLoader:
    """
    Handles data loading and basic preprocessing operations.
    
    Follows Single Responsibility Principle: only responsible for loading and basic data operations.
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config (DataConfig): Data configuration parameters
        """
        self.config = config
        self.label_encoder = LabelEncoder()
        self._duplicated_columns = None
        
    def load_and_clean_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data and perform initial cleaning operations.
        
        Args:
            data_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Load data
        data = pd.read_csv(data_path)
        
        # Log basic information
        print(f"Loaded data shape: {data.shape}")
        print(f"Data info:")
        print(data.info())
        
        # Check for missing values
        self._check_missing_values(data)
        
        # Check for duplicated rows
        self._check_duplicated_rows(data)
        
        # Handle duplicated columns
        data = self._remove_duplicated_columns(data)
        
        return data
    
    def prepare_features_and_labels(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Separate features and labels from the dataset.
        
        Args:
            data (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Features and encoded labels
        """
        # Prepare features (remove target and subject columns)
        X = data.drop([self.config.target_column, self.config.subject_column], axis=1)
        
        # Prepare labels
        y = data[self.config.target_column]
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Features shape: {X.shape}")
        print(f"Unique activities: {self.label_encoder.classes_}")
        
        return X, y_encoded
    
    def split_data(self, X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (np.ndarray): Encoded labels
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
    
    def load_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Complete pipeline for loading and preparing training data.
        
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        # Load and clean data
        data = self.load_and_clean_data(self.config.train_data_path)
        
        # Prepare features and labels
        X, y = self.prepare_features_and_labels(data)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def load_test_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load and prepare the TRUE holdout test data from test.csv.
        This is the unseen data for final evaluation.
        
        Returns:
            Tuple: X_test_holdout, y_test_holdout
        """
        # Load true test data
        test_data = pd.read_csv(self.config.test_data_path)
        print(f"Loading TRUE test data from: {self.config.test_data_path}")
        
        # Apply same column removal as training data
        if self._duplicated_columns is not None:
            test_data = test_data.drop(self._duplicated_columns, axis=1)
        
        # Prepare features and labels
        X_test_holdout = test_data.drop([self.config.target_column, self.config.subject_column], axis=1)
        y_test_holdout = test_data[self.config.target_column]
        y_test_holdout_encoded = self.label_encoder.transform(y_test_holdout)
        
        print(f"True test data shape: {X_test_holdout.shape}")
        
        return X_test_holdout, y_test_holdout_encoded
    
    def decode_predictions(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Decode numerical predictions back to original activity labels.
        
        Args:
            y_pred (np.ndarray): Numerical predictions
            
        Returns:
            np.ndarray: Decoded activity labels
        """
        return self.label_encoder.inverse_transform(y_pred)
    
    def _check_missing_values(self, data: pd.DataFrame) -> None:
        """Check and report missing values."""
        missing_values = data.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        
        if len(columns_with_missing) > 0:
            print("Columns with missing values:")
            print(columns_with_missing)
        else:
            print("No missing values found.")
    
    def _check_duplicated_rows(self, data: pd.DataFrame) -> None:
        """Check and report duplicated rows."""
        duplicated_rows = data[data.duplicated()]
        print(f"Duplicated rows: {duplicated_rows.shape[0]}")
    
    def _remove_duplicated_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicated columns and store them for later use."""
        self._duplicated_columns = data.columns[data.T.duplicated()]
        
        if len(self._duplicated_columns) > 0:
            print(f"Removing {len(self._duplicated_columns)} duplicated columns")
            data = data.drop(self._duplicated_columns, axis=1)
            print(f"Data shape after removing duplicated columns: {data.shape}")
        
        return data
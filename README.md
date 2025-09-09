# Human Activity Recognition Pipeline

A production-ready machine learning pipeline for classifying human activities using smartphone sensor data. This project demonstrates clean architecture principles, SOLID design patterns, and modern software engineering practices applied to a machine learning workflow.

## 🎯 Project Overview

This pipeline recognizes 6 different human activities:
- **LAYING**
- **SITTING** 
- **STANDING**
- **WALKING**
- **WALKING_DOWNSTAIRS**
- **WALKING_UPSTAIRS**

**Key Achievement:** 95.15% accuracy on holdout test data using an ensemble of XGBoost, SVM, Random Forest, and Logistic Regression models.

## 🏗️ Architecture

The project follows clean architecture principles with clear separation of concerns:

```
├── config/                 # Configuration Management
│   ├── __init__.py
│   └── config.py          # Centralized configuration using dataclasses
├── src/
│   ├── data/              # Data Loading & Preprocessing
│   │   ├── __init__.py
│   │   └── data_loader.py # Data loading, validation, train/test splits
│   ├── features/          # Feature Engineering
│   │   ├── __init__.py
│   │   └── feature_selector.py # Feature selection strategies
│   └── models/            # Machine Learning Models
│       ├── __init__.py
│       └── ensemble_model.py # Ensemble model with hyperparameter tuning
├── main.py               # CLI interface and pipeline orchestration
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## 📊 Machine Learning Pipeline

### 1. Data Processing
- **Data Loading:** Handles train.csv (7,352 samples) and test.csv (2,947 samples)
- **Data Validation:** Checks for missing values, duplicates, data quality
- **Train/Validation Split:** 80/20 split for model development

### 2. Feature Engineering
Three-stage feature selection pipeline:
- **Correlation Filter:** Removes highly correlated features (threshold: 0.8)
- **Variance Filter:** Removes low-variance features (threshold: 0.04)
- **Recursive Feature Elimination:** Selects top 50 features using Random Forest

**Feature Reduction:** 540 → 142 → 102 → 50 features

### 3. Model Ensemble
- **XGBoost Classifier:** Best individual performer (97.91% CV score)
- **Support Vector Machine:** 96.97% CV score
- **Random Forest:** 96.26% CV score  
- **Logistic Regression:** 94.76% CV score
- **Voting Classifier:** Combines all models with optimized weights [2.2, 2.0, 1.6, 1.8]

### 4. Hyperparameter Optimization
- **RandomizedSearchCV:** 30 iterations, 5-fold cross-validation
- **Automated:** Parameter grids defined in configuration
- **Efficient:** Parallel processing with n_jobs=-1

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd human-activity-recognition

# Install dependencies
pip install -r requirements.txt
```

### Data Setup
Place your data files in the `raw_data/` directory:
```
raw_data/
├── train.csv    # Training dataset (7,352 samples)
└── test.csv     # Holdout test set (2,947 samples)
```

### Basic Usage

#### Full Training Pipeline
```bash
python main.py \
    --data_dir "raw_data" \
    --output_dir "results/experiment_1"
```

#### Evaluation Only (with pretrained model)
```bash
python main.py \
    --data_dir "raw_data" \
    --evaluate_only \
    --model_path "results/experiment_1/har_ensemble_model.pkl" \
    --output_dir "results/evaluation_test"
```

#### Custom Hyperparameters
```bash
python main.py \
    --data_dir "raw_data" \
    --output_dir "results/experiment_2" \
    --correlation_threshold 0.75 \
    --variance_threshold 0.06 \
    --rfe_features 60
```

## 📈 Results & Performance

### Model Performance
- **Validation Accuracy:** 97.28% (20% of train.csv)
- **Holdout Test Accuracy:** 95.15% (test.csv)

### Individual Model Cross-Validation Scores
| Model | CV Score |
|-------|----------|
| XGBoost | 97.91% |
| SVM | 96.97% |
| Random Forest | 96.26% |
| Logistic Regression | 94.76% |

### Output Files
The pipeline generates comprehensive results:
```
results/
├── har_ensemble_model.pkl           # Trained model + feature pipeline
├── activity_distribution.png        # Activity distribution by subject
├── feature_selection_summary.png    # Feature selection progress
├── confusion_matrix_validation.png  # Validation confusion matrix
├── confusion_matrix_holdout.png     # Holdout test confusion matrix
├── evaluation_report_validation.txt # Detailed validation metrics
└── evaluation_report_holdout.txt    # Detailed holdout test metrics
```


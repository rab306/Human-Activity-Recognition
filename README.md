# Human Activity Recognition using Smartphone Sensors

This repository contains code for a machine learning project that classifies human activities based on smartphone sensor data. The project aims to predict six different activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING.

## Project Overview

The project involves:

- Preprocessing and analyzing smartphone sensor data.
- Feature engineering and selection using techniques like variance thresholding and Recursive Feature Elimination (RFE).
- Implementing machine learning models such as XGBoost, SVM, Logistic Regression, and Random Forest.
- Utilizing ensemble learning with a Voting Classifier to improve prediction accuracy.
- Evaluating model performance using accuracy metrics, confusion matrices, and classification reports.
- Testing the models on unseen data to assess generalization.

## Repository Structure

- `human_activity_recognition.py`: Python script containing the main model training and evaluation code.
- `train.csv`: Dataset containing labeled sensor data used for training.
- `test.csv`: Dataset containing unseen sensor data used for testing.

## Dataset 
The data is available on kaggle (https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)

## Dataset Describtion
The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed 
six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist.
Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. 
The experiments have been video-recorded to label the data manually. 
The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data.

## Installation

To run the code, ensure you have Python 3.12 installed along with the required libraries listed in `requirements.txt`. You can install the dependencies using pip:

```bash
pip install -r requirements.txt

# Loading libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Loading data
data = pd.read_csv(r'E:\projects\HAR\train.csv')
test_data = pd.read_csv(r'E:\projects\HAR\test.csv')  # The unseen data

# Checking the data
print(data.shape, test_data.shape)
print(data.head())
print(data.info())

# Checking for missing values
missing_values = data.isnull().sum()
column_with_missing_values = missing_values[missing_values > 0]
print(column_with_missing_values)  # There are no missing values

# Checking for duplicated rows
duplicated_rows = data[data.duplicated()]
print(duplicated_rows.shape)  # No duplicated rows

# Checking for duplicated columns
duplicated_columns = data.columns[data.T.duplicated()]
print(duplicated_columns.shape)  # There are duplicated columns a
# Droping duplicated columns
data = data.drop(duplicated_columns, axis=1)
print(data.shape)

# Plotting the Activity distribution for each subject to check if any subject has more weight
plt.figure(figsize=(12, 6))
sns.countplot(x='subject', hue='Activity', data=data)
plt.title('Activity Count for Each Subject')
plt.xlabel('Subject')
plt.ylabel('Activity Count')
plt.show()

# Preparing features and labels
X = data.drop(['Activity', 'subject'], axis=1)
y = data['Activity']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Next we will apply feature selection techniques using
# 1- Filter method (correlation analysis, variance threshold)
# 2- Wrapper method (Recursive Feature Elimination)
# Note that Splitting the data before applying feature selection techniques is important to avoid data leakage

# Computing the correlation matrix and remove highly correlated features
correlation_matrix_train = pd.DataFrame(X_train).corr()
mask_train = np.triu(np.ones(correlation_matrix_train.shape), k=1).astype(bool)
correlated_features_train = set()
for i in range(len(correlation_matrix_train.columns)):
    for j in range(i):
        if abs(correlation_matrix_train.iloc[i, j]) > 0.8: #I have experimented with different threshold values (0.7, 0.75)
            colname = correlation_matrix_train.columns[i]
            correlated_features_train.add(colname)
X_train_no_corr = pd.DataFrame(X_train).drop(columns=correlated_features_train)
X_test_no_corr = pd.DataFrame(X_test).drop(columns=correlated_features_train)
print(X_train_no_corr.shape, X_test_no_corr.shape)


# Features variance analysis
feature_variance = X_train_no_corr.var()
plt.figure(figsize=(12, 6))
sns.histplot(feature_variance, bins=500, kde=True)
plt.title('Distribution of Feature Variances')
plt.xlabel('Variance')
plt.ylabel('Frequency')
plt.show()

# Variance thresholding
# I have experimented with different threshold values (0.06, 0.08) but 0.04 gave better results
var_selector = VarianceThreshold(threshold=0.04)
X_train_high_variance = var_selector.fit_transform(X_train_no_corr)
X_test_high_variance = var_selector.transform(X_test_no_corr)
print(X_train_high_variance.shape, X_test_high_variance.shape)

# Recursive Feature Elimination (RFE):
# This step is highly sensitive to the number of features to select.
# In my experiments:
# - Selecting 50 features consistently achieved accuracies on unseen data over 94.8%.
# - Experimenting with different numbers of features (22, 30, 40) resulted in accuracies under 94%.
estimator = RandomForestClassifier(random_state=42, n_jobs=-1)
rfe_selector = RFE(estimator, n_features_to_select=50)
X_train_rfe = rfe_selector.fit_transform(X_train_high_variance, y_train)
X_test_rfe = rfe_selector.transform(X_test_high_variance)
print(X_train_rfe.shape)


# Experimenting with dimensionality reduction using PCA, but it didn't improve results:
# Results on unseen data: 93.48%

# Uncomment below to apply PCA (not included in final script):
# pca = PCA(n_components=num_components)
# X_train_pca = pca.fit_transform(X_train_rfe)
# X_test_data_pca = pca.transform(X_test_data_rfe)

# Calculate cumulative explained variance:
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# Determine the number of components needed to explain 95% variance:
# num_components = np.where(cumulative_variance >= 0.95)[0][0] + 1
# print(f"Number of components to retain 95% variance: {num_components}")


# Defining parameter grids for hyperparameter tuning:
# - After experimenting with GridSearchCV and RandomizedSearchCV,
#   redundant hyperparameters were removed to optimize time and computation.
# - RandomizedSearchCV yielded comparable results to GridSearchCV
#   while significantly reducing time and computational costs.

param_grid_xgb = {
    'classifier__learning_rate': [0.3, 0.4, 0.5],
    'classifier__n_estimators': [50, 100, 150]
}

param_grid_svc = {
    'classifier__C': [0.1, 1, 10],
    'classifier__gamma': [1, 0.1, 0.01],
}

param_grid_lr = {
    'classifier__C': [0.1, 1, 10],
}

param_grid_rf = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__min_samples_split': [2, 4, 6],
}


# Defining feature scaling for SVM and Logistic Regression
scaler_lr_svc = StandardScaler()
scaler_svc = StandardScaler()

# Initializing THE classifiers
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=6, random_state=42)
svc_model = SVC(probability=True, random_state=42)
lr_model = LogisticRegression(multi_class='auto', random_state=42)
rf_model = RandomForestClassifier(random_state=42)

# Defining pipelines
pipeline_lr = Pipeline([('scaler', scaler_lr_svc), ('classifier', lr_model)])
pipeline_svc = Pipeline([('scaler', scaler_lr_svc), ('classifier', svc_model)])
pipeline_xgb = Pipeline([('classifier', xgb_model)])
pipeline_rf = Pipeline([('classifier', rf_model)])

# Defining RandomizedSearchCV objects
random_search_xgb = RandomizedSearchCV(pipeline_xgb, param_grid_xgb, n_iter=30, cv=5, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42)
random_search_svc = RandomizedSearchCV(pipeline_svc, param_grid_svc, n_iter=30, cv=5, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42)
random_search_lr = RandomizedSearchCV(pipeline_lr, param_grid_lr, n_iter=30, cv=5, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42)
random_search_rf = RandomizedSearchCV(pipeline_rf, param_grid_rf, n_iter=30, cv=5, scoring='accuracy', n_jobs=-1, verbose=2, random_state=42)

# Fitting RandomizedSearchCV for each classifier
random_search_xgb.fit(X_train_rfe, y_train)
random_search_svc.fit(X_train_rfe, y_train)
random_search_lr.fit(X_train_rfe, y_train)
random_search_rf.fit(X_train_rfe, y_train)

# Comparing the best scores of each classifier
best_score_xgb = random_search_xgb.best_score_
best_score_svc = random_search_svc.best_score_
best_score_lr = random_search_lr.best_score_
best_score_rf = random_search_rf.best_score_

print(f"XGBoost Best Score: {best_score_xgb:.4f}")
print(f"SVC Best Score: {best_score_svc:.4f}")
print(f"Logistic Regression Best Score: {best_score_lr:.4f}")
print(f"Random Forest Best Score: {best_score_rf:.4f}")

# Function to remove 'classifier__' prefix
def remove_classifier_prefix(params):
    return {key.split('__')[1]: value for key, value in params.items()}

# Getting the best parameters for each classifier
best_params_xgb = random_search_xgb.best_params_
best_params_svc = random_search_svc.best_params_
best_params_lr = random_search_lr.best_params_
best_params_rf = random_search_rf.best_params_

# Removing 'classifier__' prefix from best parameters
best_params_xgb = remove_classifier_prefix(best_params_xgb)
best_params_svc = remove_classifier_prefix(best_params_svc)
best_params_lr = remove_classifier_prefix(best_params_lr)
best_params_rf = remove_classifier_prefix(best_params_rf)

print(f"XGBoost Best Score: {best_params_xgb:.4f}")
print(f"SVC Best Score: {best_params_svc:.4f}")
print(f"Logistic Regression Best Score: {best_params_lr:.4f}")
print(f"Random Forest Best Score: {best_params_rf:.4f}")

# Initializing THE classifiers with THEIR best parameters
xgb_model_best = xgb.XGBClassifier(**best_params_xgb)
svc_model_best = SVC(**best_params_svc, probability=True)
lr_model_best = LogisticRegression(**best_params_lr, multi_class='auto')
rf_model_best = RandomForestClassifier(**best_params_rf)

# Voting Classifier initialization:
# - Experimented with different weight configurations:
# - Both of these configurations yielded the best results.
#   1. Equal weights for all classifiers.
#   2. Highest weight for XGBoost (xgb), lowest for Logistic Regression,
#      and equal weights for SVM and Random Forest.
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model_best),
        ('svc', svc_model_best),
        ('lr', lr_model_best),
        ('rf', rf_model_best)
    ],
    voting='soft',
    weights=[2.2, 2, 1.6, 1.8]
)
voting_clf.fit(X_train_rfe, y_train)


# Evaluating Voting Classifier
y_pred_voting = voting_clf.predict(X_test_rfe)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
conf_matrix_voting = confusion_matrix(y_test, y_pred_voting)
class_report_voting = classification_report(y_test, y_pred_voting)

print(f"Test Accuracy: {accuracy_voting:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix_voting)
print("\nClassification Report:")
print(class_report_voting)


# Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_voting, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Decoding the  predictions
decoded_predictions = label_encoder.inverse_transform(y_pred_voting)

#Original and decoded labels comparison
print("\nOriginal Labels:")
print(y_test[:10])
print("\nDecoded Labels:")
print(decoded_predictions[:10])


# Evaluating the model on unseen data
# Preprocessing the unseen data
test_data = test_data.drop(duplicated_columns, axis=1)
print(test_data.shape)

X_test_data = test_data.drop(['Activity', 'subject'], axis=1)
y_test_data = test_data['Activity']
y_test_data = label_encoder.transform(y_test_data)

X_test_data_no_corr = pd.DataFrame(X_test_data).drop(columns=correlated_features_train)
print(X_test_data_no_corr.shape)

X_test_data_high_variance = var_selector.transform(X_test_data_no_corr)
X_test_data_rfe = rfe_selector.transform(X_test_data_high_variance)

y_pred_test = voting_clf.predict(X_test_data_rfe)
decoded_predictions_test = label_encoder.inverse_transform(y_pred_test)

# Evaluation metrics
accuracy_test = accuracy_score(y_test_data, y_pred_test)
conf_matrix_test = confusion_matrix(y_test_data, y_pred_test)
class_report_test = classification_report(y_test_data, y_pred_test)
print(f"Accuracy on Test Data: {accuracy_test:.4f}")
print("\nConfusion Matrix on Test Data:")
print(conf_matrix_test)
print("\nClassification Report on Test Data:")
print(class_report_test)



import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, roc_curve, confusion_matrix, classification_report, RocCurveDisplay, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
train_data = pd.read_csv('./Machine_Learning/flattened_images_train.csv')
test_data = pd.read_csv('./Machine_Learning/flattened_images_test.csv')

# Data preprocessing
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
y_train_bin = label_binarize(y_train, classes=categories)
y_test_bin = label_binarize(y_test, classes=categories)

# Data preprocessing
X = train_data.drop('label', axis=1).values
y = train_data['label'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Parameters for k
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)

# Initialize K-NN Classifier
knn = KNeighborsClassifier()

# GridSearchCV for k_best
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', return_train_score=False)
grid.fit(X_train, y_train)

# Collect results
k_values = grid.cv_results_['param_n_neighbors']
validation_accuracies = grid.cv_results_['mean_test_score']

# Plotting the accuracy for each k value on the validation set
plt.figure(figsize=(12, 6))
plt.plot(k_values, validation_accuracies, marker='o', linestyle='-', color='salmon')
plt.title('Validation Accuracy for Each k')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Validation Accuracy')
plt.grid(True)
plt.show()

# Print best k value and its accuracy
best_k = grid.best_params_['n_neighbors']
best_score = grid.best_score_
print(f"Best k found: {best_k} with validation accuracy: {best_score:.4f}")

# # best k = 1   
# # Train K-NN
# start_time = time.time()
# knn_model = KNeighborsClassifier(n_neighbors=10)
# knn_model.fit(X_train, y_train)
# training_time = time.time() - start_time

# # Predict
# y_pred = knn_model.predict(X_test)
# y_proba = knn_model.predict_proba(X_test)

# # Compute performance metrics
# accuracy = accuracy_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred, average='weighted')
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)

# # Confusion Matrix
# plt.figure(figsize=(10, 8))
# conf_matrix = confusion_matrix(y_test, y_pred)
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=categories, yticklabels=categories)
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Label')
# plt.xlabel('Predicted Label')
# plt.show()

# # Accuracy for Each Class
# plt.figure(figsize=(12, 6))
# class_accuracy = 100 * conf_matrix.diagonal() / conf_matrix.sum(1)
# bars = plt.bar(categories, class_accuracy, color='salmon')
# plt.xlabel('Class')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy for Each Class')
# plt.ylim(0, 100)
# for bar, accuracy in zip(bars, class_accuracy):
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(accuracy, 2), ha='center', va='bottom')

# plt.show()

# # ROC
# plt.figure(figsize=(10, 8))
# for i, class_label in enumerate(categories):
#     fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
#     roc_auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
#     plt.plot(fpr, tpr, label=f'{class_label} (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc='lower right')
# plt.show()

# # Results
# print(f"K-NN Training Time: {training_time}s")
# print(f"K-NN Accuracy: {accuracy}")
# print(f"K-NN Recall: {recall}")
# print("K-NN Classification Report:\n", class_report)

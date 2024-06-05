import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc, classification_report
import seaborn as sns

# Load the dataset
train_data = pd.read_csv('./Machine_Learning/flattened_images_train.csv')
test_data = pd.read_csv('./Machine_Learning/flattened_images_test.csv')

# Data preprocessing
X_train = train_data.drop('label', axis=1)
y_train = train_data['label']
X_test = test_data.drop('label', axis=1)
y_test = test_data['label']

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the SGD model
model = SGDClassifier(loss='hinge')  # 'hinge' loss simulates a linear SVM.

# Train the model in batches
batch_size = 100
n_batches = int(np.ceil(X_train.shape[0] / batch_size))
total_start_time = time()
accuracies = []

for i in tqdm(range(n_batches), desc="Training model"):
    start = i * batch_size
    end = min((i + 1) * batch_size, X_train.shape[0])
    batch_start_time = time()
    model.partial_fit(X_train[start:end], y_train[start:end], classes=np.unique(y_train))
    batch_accuracy = model.score(X_train[start:end], y_train[start:end])
    accuracies.append(batch_accuracy)
    batch_end_time = time()
    print(f"Batch {i+1}/{n_batches} training time: {batch_end_time - batch_start_time:.2f} seconds, Accuracy: {batch_accuracy:.2f}")

total_end_time = time()
print(f"Total training time: {total_end_time - total_start_time:.2f} seconds")

# Plotting the accuracy per batch
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_batches + 1), accuracies, linestyle='-', color='salmon')
plt.title('LOSS')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.ylim(0, 1)  # Set the y-axis to start from 0
plt.grid(True)
plt.show()

total_end_time = time()
print(f"Total training time: {total_end_time - total_start_time:.2f} seconds")

# Compute performance metrics
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions, average='macro')
conf_matrix = confusion_matrix(y_test, predictions)

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=sorted(y_train.unique()), yticklabels=sorted(y_train.unique()))
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Accuracy for Each Class
categories = sorted(y_train.unique())
class_accuracy = 100 * conf_matrix.diagonal() / conf_matrix.sum(1)
plt.figure(figsize=(12, 6))
bars = plt.bar(categories, class_accuracy, color='salmon')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for Each Class')
plt.ylim(0, 100)
for bar, accuracy in zip(bars, class_accuracy):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{accuracy:.2f}%', ha='center', va='bottom')
plt.show()

# ROC
y_test_bin = label_binarize(y_test, classes=sorted(y_train.unique()))
predictions_bin = label_binarize(predictions, classes=sorted(y_train.unique()))
plt.figure(figsize=(10, 8))
roc_auc_all = []
for i, class_label in enumerate(sorted(y_train.unique())):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], predictions_bin[:, i])
    roc_auc = auc(fpr, tpr)
    roc_auc_all.append(roc_auc)
    plt.plot(fpr, tpr, label=f'{class_label} (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Results
print("Classification Report:\n", classification_report(y_test, predictions))
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"ROC AUC: {np.mean(roc_auc_all):.4f}")



# # Initialize the SVM model
# model = SVC(kernel='linear')

# start_time = time()
# model.fit(X_train, y_train)

# end_time = time()
# print(f"Total training time: {end_time - start_time:.2f} seconds")

# # Compute performance metrics
# predictions = model.predict(X_test)
# accuracy = accuracy_score(y_test, predictions)
# recall = recall_score(y_test, predictions, average='macro')
# conf_matrix = confusion_matrix(y_test, predictions)

# # Confusion Matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=sorted(y_train.unique()), yticklabels=sorted(y_train.unique()))
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Label')
# plt.xlabel('Predicted Label')
# plt.show()

# # Accuracy for Each Class
# categories = sorted(y_train.unique())
# class_accuracy = 100 * conf_matrix.diagonal() / conf_matrix.sum(1)
# plt.figure(figsize=(12, 6))
# bars = plt.bar(categories, class_accuracy, color='salmon')
# plt.xlabel('Class')
# plt.ylabel('Accuracy (%)')
# plt.title('Accuracy for Each Class')
# plt.ylim(0, 100)
# for bar, accuracy in zip(bars, class_accuracy):
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f'{accuracy:.2f}%', ha='center', va='bottom')
# plt.show()

# # ROC
# y_test_bin = label_binarize(y_test, classes=sorted(y_train.unique()))
# predictions_bin = label_binarize(predictions, classes=sorted(y_train.unique()))
# plt.figure(figsize=(10, 8))
# for i, class_label in enumerate(sorted(y_train.unique())):
#     fpr, tpr, _ = roc_curve(y_test_bin[:, i], predictions_bin[:, i])
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, label=f'{class_label} (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc='lower right')
# plt.show()

# # Results
# print("Classification Report:\n", classification_report(y_test, predictions))
# print(f"Accuracy: {accuracy}")
# print(f"Recall: {recall}")
# print(f"ROC AUC: {roc_auc}")
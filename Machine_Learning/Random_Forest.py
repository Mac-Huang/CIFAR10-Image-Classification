import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_data = pd.read_csv('./Machine_Learning/flattened_images_train.csv')
test_data = pd.read_csv('./Machine_Learning/flattened_images_test.csv')

# Prepare data
X_train = train_data.drop('label', axis=1).values
y_train = train_data['label'].values
X_test = test_data.drop('label', axis=1).values
y_test = test_data['label'].values

categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
y_train_bin = label_binarize(y_train, classes=categories)
y_test_bin = label_binarize(y_test, classes=categories)

# Train Random Forest
start_time = time.time()
rf_model = RandomForestClassifier(n_estimators=100) # 100 trees
rf_model.fit(X_train, y_train)
training_time = time.time() - start_time

# Predict
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)

# Draw confusion matrix
plt.figure(figsize=(10, 8))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# Draw accuracy bar chart for each class
plt.figure(figsize=(12, 6))
class_accuracy = 100 * conf_matrix.diagonal() / conf_matrix.sum(1)
bars = plt.bar(categories, class_accuracy, color='salmon')
plt.xlabel('Class')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for Each Class')
plt.ylim(0, 100)  # Set the y-axis range from 0 to 100

# Add accuracy values
for bar, accuracy in zip(bars, class_accuracy):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{accuracy:.2f}%', ha='center', va='bottom')

plt.show()

# Draw ROC curve
plt.figure(figsize=(10, 8))
for i, class_label in enumerate(categories):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, label=f'{class_label} (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Output results
print(f"Random Forest Training Time: {training_time}s")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Random Forest Recall: {recall_score(y_test, y_pred, average='weighted')}")
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred))

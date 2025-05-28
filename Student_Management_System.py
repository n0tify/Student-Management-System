import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load dataset and encode categorical columns
df = pd.read_csv("xAPI-Edu-Data.csv")
for col in df.select_dtypes(include='object'):
    df[col] = LabelEncoder().fit_transform(df[col]) 

X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split with stratification to keep class distribution consistent
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42)

# Visualize class distribution before SMOTE
sns.countplot(data=pd.DataFrame({'Class': y_train.map({0:'H', 1:'L', 2:'M'})}), 
              x='Class', hue='Class', palette=['lightpink', 'lightgreen', 'lightskyblue'], legend=False)
plt.title("Class Distribution Before SMOTE")
plt.show()

# Apply SMOTE to balance classes in training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Visualize class distribution after SMOTE
sns.countplot(data=pd.DataFrame({'Class': y_resampled.map({0:'H', 1:'L', 2:'M'})}), 
              x='Class', hue='Class', palette=['lightpink', 'lightgreen', 'lightskyblue'], legend=False)
plt.title("Class Distribution After SMOTE")
plt.show()

# Print class distributions before and after SMOTE for confirmation
print("\nClass distribution before SMOTE:\n", y_train.value_counts())
print("\nClass distribution after SMOTE:\n", pd.Series(y_resampled).value_counts())

# Initialize and train XGBoost classifier
model = XGBClassifier(
    n_estimators=300, learning_rate=0.08, max_depth=5,
    subsample=0.9, colsample_bytree=0.9, eval_metric='mlogloss', random_state=42)
model.fit(X_resampled, y_resampled)

# Predict on test set
y_pred = model.predict(X_test)

# Classification report and accuracy
print("\nClassification Report on Test Set:\n")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='YlGnBu', fmt='d', xticklabels=['H', 'L', 'M'], yticklabels=['H', 'L', 'M'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature importance plot with hue and legend=False to fix FutureWarning
importances = model.feature_importances_
features = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(15)

sns.barplot(
    data=features,
    y='Feature',
    x='Importance',
    hue='Feature',  # Assign hue to avoid FutureWarning
    dodge=False,
    palette=['lightpink', 'lightgreen', 'khaki', 'lightskyblue', 'lightcoral']*3,
    legend=False
)
plt.title("Top 15 Feature Importances")
plt.show()

# ROC and Precision-Recall curves preparation
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_proba = model.predict_proba(X_test)

# ROC Curves for each class
plt.figure(figsize=(6, 5))
for i, color in zip(range(3), ['lightcoral', 'lightgreen', 'lightskyblue']):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    plt.plot(fpr, tpr, label=f'Class {i}', color=color)
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# Precision-Recall Curves for each class
plt.figure(figsize=(6, 5))
for i, color in zip(range(3), ['lightcoral', 'lightgreen', 'lightskyblue']):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
    plt.plot(recall, precision, label=f'Class {i}', color=color)
plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.show()

# Visualize class distribution in test set
sns.countplot(x=y_test.map({0:'H', 1:'L', 2:'M'}), palette='pastel')
plt.title("Class Distribution in Test Set")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# sample actual vs predicted classes 
comparison = pd.DataFrame({'Actual': y_test[:].values, 'Predicted': y_pred[:]})
print("\nSample Predictions vs Actual:")
print(comparison)

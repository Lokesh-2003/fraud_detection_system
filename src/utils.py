# src/utils.py

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

def plot_feature_importance(model, feature_names):
     """Plots feature importance for tree-based models (XGBoost, Decision Tree)."""
     if hasattr(model, 'feature_importances_'):
         importances = model.feature_importances_
         indices = np.argsort(importances)
         plt.figure(figsize=(10, len(feature_names) // 2))  # Adjust figure size as needed
         plt.barh(range(len(indices)), importances[indices], align='center')
         plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
         plt.xlabel('Feature Importance')
         plt.ylabel('Feature')
         plt.title('Feature Importance')
         plt.show()
     else:
         print("Feature importance not available for this model.")
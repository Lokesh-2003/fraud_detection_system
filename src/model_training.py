# src/model_training.py

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve, f1_score
import matplotlib.pyplot as plt
import joblib  # For saving models

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Trains and evaluates a given model.

    Args:
        model: The model to train.
        X_train: Training features.
        y_train: Training labels.
        X_test: Testing features.
        y_test: Testing labels.
        model_name: Name of the model for reporting.

    Returns:
        The trained model.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n--- {model_name} ---")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Visualization (ROC Curve and Precision-Recall Curve)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (F1 = {f1_score(y_test, y_pred):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curve')
    plt.legend()
    plt.show()

    return model

def train_logistic_regression(X_train, y_train, param_grid=None):
    """Trains a Logistic Regression model with optional hyperparameter tuning."""
    model = LogisticRegression(solver='liblinear', random_state=42)
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1')
        grid_search.fit(X_train, y_train)
        print(f"Best Parameters: {grid_search.best_params_}, Best F1: {grid_search.best_score_}")
        return grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model

def train_decision_tree(X_train, y_train, param_grid=None):
    """Trains a Decision Tree model with optional hyperparameter tuning."""

    model = DecisionTreeClassifier(random_state=42)
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1')
        grid_search.fit(X_train, y_train)
        print(f"Best Parameters: {grid_search.best_params_}, Best F1: {grid_search.best_score_}")
        return grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model

def train_xgboost(X_train, y_train, param_grid=None):
    """Trains an XGBoost model with optional hyperparameter tuning."""

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='f1')
        grid_search.fit(X_train, y_train)
        print(f"Best Parameters: {grid_search.best_params_}, Best F1: {grid_search.best_score_}")
        return grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        return model

def save_model(model, filename="model.joblib"):
    """Saves a trained model to a file."""
    joblib.dump(model, filename)

def load_model(filename="model.joblib"):
    """Loads a trained model from a file."""
    return joblib.load(filename)
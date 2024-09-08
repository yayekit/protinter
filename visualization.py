import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, learning_curve
from typing import List, Tuple

def plot_feature_importance(model: XGBClassifier, feature_names: List[str]) -> None:
    """Plot feature importance using seaborn."""
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importance})
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Plot confusion matrix using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

def plot_correlation_matrix(X: np.ndarray, feature_names: List[str]) -> None:
    """Plot correlation matrix of features using seaborn."""
    corr = pd.DataFrame(X, columns=feature_names).corr()
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr, cmap='coolwarm', annot=False, square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.close()

def plot_learning_curve(estimator: XGBClassifier, X: np.ndarray, y: np.ndarray) -> None:
    """Plot learning curve for the model."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10), scoring="roc_auc"
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label="Training score", color="blue")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    plt.plot(train_sizes, test_mean, label="Cross-validation score", color="red")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="red")

    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("ROC AUC Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.close()

def visualize_results(model: XGBClassifier, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray, feature_names: List[str]) -> None:
    """Generate and save all visualization plots."""
    plot_feature_importance(model, feature_names)
    plot_confusion_matrix(y, y_pred)
    plot_correlation_matrix(X, feature_names)
    plot_learning_curve(model, X, y)
    print("All visualization plots have been generated and saved.")
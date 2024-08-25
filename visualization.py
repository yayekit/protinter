import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

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
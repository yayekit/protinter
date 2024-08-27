import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

def train_model_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[XGBClassifier, StandardScaler]:
    """Train XGBoost model with cross-validation and hyperparameter tuning."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'early_stopping_rounds': [10, 20, 30]  # Add early stopping parameter
    }
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, verbose=2, scoring='roc_auc')
    grid_search.fit(X_scaled, y)
    
    best_model = grid_search.best_estimator_
    
    print("Best parameters:", grid_search.best_params_)
    print("Cross-validation results:")
    for i, score in enumerate(grid_search.cv_results_['split_test_score']):
        print(f"Fold {i+1}: {score:.3f}")
    print(f"Mean ROC AUC: {grid_search.best_score_:.3f}")
    
    return best_model, scaler

def evaluate_model(model: XGBClassifier, X: np.ndarray, y: np.ndarray, scaler: StandardScaler) -> Dict[str, float]:
    """Evaluate the model on the entire dataset."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1-score": f1_score(y, y_pred),
        "ROC AUC": roc_auc_score(y, y_pred_proba)
    }
    
    print("Final Model Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    return metrics

def save_model(model: XGBClassifier, scaler: StandardScaler, filename: str) -> None:
    """Save the trained model and scaler to a file."""
    joblib.dump({'model': model, 'scaler': scaler}, filename)
    print(f"Model and scaler saved to {filename}")

def load_model(filename: str) -> Tuple[XGBClassifier, StandardScaler]:
    """Load the trained model and scaler from a file."""
    loaded = joblib.load(filename)
    return loaded['model'], loaded['scaler']

def predict_new_data(model: XGBClassifier, scaler: StandardScaler, X_new: np.ndarray) -> np.ndarray:
    """Make predictions on new data using the trained model and scaler."""
    X_scaled = scaler.transform(X_new)
    return model.predict(X_scaled)
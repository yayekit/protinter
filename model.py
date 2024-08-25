import numpy as np
from typing import Tuple
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

def train_model_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[XGBClassifier, StandardScaler]:
    """Train XGBoost model with cross-validation and hyperparameter tuning."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5]
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

def evaluate_model(model: XGBClassifier, X: np.ndarray, y: np.ndarray, scaler: StandardScaler) -> None:
    """Evaluate the model on the entire dataset."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    
    print("Final Model Performance:")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1-score:", f1_score(y, y_pred))
    print("ROC AUC:", roc_auc_score(y, y_pred_proba))
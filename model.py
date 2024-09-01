import numpy as np
from typing import Tuple, Dict
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib
import optuna

def train_model_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Tuple[XGBClassifier, StandardScaler]:
    """Train XGBoost model with cross-validation and hyperparameter tuning."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        # Remove 'early_stopping_rounds' from param_grid
    }
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', early_stopping_rounds=10)
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

def compare_models(X: np.ndarray, y: np.ndarray, models: Dict[str, Any]) -> Dict[str, float]:
    """Compare multiple models using cross-validation."""
    from sklearn.model_selection import cross_val_score
    
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        results[name] = scores.mean()
    
    return results

def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, n_trials: int = 100) -> Dict[str, Any]:
    """Optimize hyperparameters using Optuna."""
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 9),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        }
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')
        score = cross_val_score(model, X, y, cv=5, scoring='roc_auc').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params

# Usage in main.py:
models = {
    'XGBoost': XGBClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True)
}
model_comparison = compare_models(X, y, models)
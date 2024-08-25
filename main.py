import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from Bio import SeqIO
from Bio.SeqUtils import ProtParam
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def extract_features(sequence):
    """Extract features from a protein sequence."""
    analyser = ProtParam.ProteinAnalysis(str(sequence))
    amino_acid_percent = analyser.get_amino_acids_percent()
    return {
        'length': len(sequence),
        'weight': analyser.molecular_weight(),
        'aromaticity': analyser.aromaticity(),
        'instability': analyser.instability_index(),
        'isoelectric_point': analyser.isoelectric_point(),
        'helix_fraction': analyser.secondary_structure_fraction()[0],
        'turn_fraction': analyser.secondary_structure_fraction()[1],
        'sheet_fraction': analyser.secondary_structure_fraction()[2],
        'gravy': analyser.gravy(),
        **{f'{aa}_percent': percent for aa, percent in amino_acid_percent.items()}
    }

def compute_conjoint_triad(sequence):
    """Compute Conjoint Triad features."""
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    groups = {'A': 0, 'G': 0, 'V': 0,
              'I': 1, 'L': 1, 'F': 1, 'P': 1,
              'Y': 2, 'M': 2, 'T': 2, 'S': 2,
              'H': 3, 'N': 3, 'Q': 3, 'W': 3,
              'R': 4, 'K': 4,
              'D': 5, 'E': 5,
              'C': 6}
    features = [0] * 343  # 7^3 possible triads

    for i in range(len(sequence) - 2):
        triad = (groups[sequence[i]], groups[sequence[i+1]], groups[sequence[i+2]])
        features[triad[0]*49 + triad[1]*7 + triad[2]] += 1

    return features

def prepare_data(positive_file, negative_file):
    """Prepare data from FASTA files of interacting and non-interacting protein pairs."""
    positive_pairs = list(SeqIO.parse(positive_file, "fasta"))
    negative_pairs = list(SeqIO.parse(negative_file, "fasta"))
    
    data = []
    labels = []
    
    for pair in positive_pairs + negative_pairs:
        seq1, seq2 = pair.seq[:len(pair.seq)//2], pair.seq[len(pair.seq)//2:]
        features1 = extract_features(seq1)
        features2 = extract_features(seq2)
        conjoint_triad1 = compute_conjoint_triad(str(seq1))
        conjoint_triad2 = compute_conjoint_triad(str(seq2))
        data.append([*features1.values(), *features2.values(), *conjoint_triad1, *conjoint_triad2])
        labels.append(1 if pair in positive_pairs else 0)
    
    return np.array(data), np.array(labels)

def train_model(X, y):
    """Train XGBoost model with hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5]
    }
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    print("Best parameters:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    
    return best_model, scaler

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(20), importance[indices][:20])
    plt.xticks(range(20), [feature_names[i] for i in indices[:20]], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

def main():
    positive_file = "positive_interactions.fasta"
    negative_file = "negative_interactions.fasta"
    
    X, y = prepare_data(positive_file, negative_file)
    feature_names = list(extract_features("A").keys()) * 2 + [f"CT_{i}" for i in range(686)]
    model, scaler = train_model(X, y)
    
    plot_feature_importance(model, feature_names)
    
    # Save model and scaler
    import joblib
    joblib.dump(model, "xgboost_model.joblib")
    joblib.dump(scaler, "scaler.joblib")

if __name__ == "__main__":
    main()
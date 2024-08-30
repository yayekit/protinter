import numpy as np
from Bio import SeqIO
import joblib
from typing import Tuple, List
from features import extract_features, compute_conjoint_triad
from model import train_model_cv, evaluate_model, save_model
from visualization import plot_feature_importance, plot_confusion_matrix, plot_correlation_matrix

def prepare_data(positive_file: str, negative_file: str) -> Tuple[np.ndarray, np.ndarray]:
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

def main():
    positive_file = "positive_interactions.fasta"
    negative_file = "negative_interactions.fasta"
    
    X, y = prepare_data(positive_file, negative_file)
    
    # Dynamically determine the number of CT features
    sample_seq = SeqIO.read(positive_file, "fasta")[0].seq
    ct_features = compute_conjoint_triad(str(sample_seq))
    feature_names = list(extract_features(sample_seq).keys()) * 2 + [f"CT_{i}" for i in range(len(ct_features))]
    
    model, scaler = train_model_cv(X, y)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    evaluate_model(model, X, y, scaler)
    plot_feature_importance(model, feature_names)
    plot_confusion_matrix(y, y_pred)
    plot_correlation_matrix(X, feature_names)
    
    # Save model and scaler
    save_model(model, scaler, "protein_interaction_model.joblib")

if __name__ == "__main__":
    main()
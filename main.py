import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from Bio import SeqIO
from Bio.SeqUtils import ProtParam

def extract_features(sequence):
    """Extract features from a protein sequence."""
    analyser = ProtParam.ProteinAnalysis(str(sequence))
    return {
        'length': len(sequence),
        'weight': analyser.molecular_weight(),
        'aromaticity': analyser.aromaticity(),
        'instability': analyser.instability_index(),
        'isoelectric_point': analyser.isoelectric_point(),
        'helix_fraction': analyser.secondary_structure_fraction()[0],
        'turn_fraction': analyser.secondary_structure_fraction()[1],
        'sheet_fraction': analyser.secondary_structure_fraction()[2]
    }

def prepare_data(positive_file, negative_file):
    """Prepare data from FASTA files of interacting and non-interacting protein pairs."""
    positive_pairs = list(SeqIO.parse(positive_file, "fasta"))
    negative_pairs = list(SeqIO.parse(negative_file, "fasta"))
    
    data = []
    labels = []
    
    for pair in positive_pairs:
        seq1, seq2 = pair.seq[:len(pair.seq)//2], pair.seq[len(pair.seq)//2:]
        features1 = extract_features(seq1)
        features2 = extract_features(seq2)
        data.append([*features1.values(), *features2.values()])
        labels.append(1)
    
    for pair in negative_pairs:
        seq1, seq2 = pair.seq[:len(pair.seq)//2], pair.seq[len(pair.seq)//2:]
        features1 = extract_features(seq1)
        features2 = extract_features(seq2)
        data.append([*features1.values(), *features2.values()])
        labels.append(0)
    
    return np.array(data), np.array(labels)

def train_model(X, y):
    """Train XGBoost model."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    
    return model, scaler

def main():
    positive_file = "positive_interactions.fasta"
    negative_file = "negative_interactions.fasta"
    
    X, y = prepare_data(positive_file, negative_file)
    model, scaler = train_model(X, y)
    
    # Here you could add code to save the model and scaler for later use

if __name__ == "__main__":
    main()
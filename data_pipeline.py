import numpy as np
from Bio import SeqIO
from typing import Tuple, List
from features import extract_features, compute_conjoint_triad

def load_fasta(file_path: str) -> List[SeqIO.SeqRecord]:
    """Load sequences from a FASTA fi   le."""
    return list(SeqIO.parse(file_path, "fasta"))

def extract_pair_features(seq1: str, seq2: str) -> List[float]:
    """Extract features for a pair of protein sequences."""
    features1 = extract_features(seq1)
    features2 = extract_features(seq2)
    conjoint_triad1 = compute_conjoint_triad(str(seq1))
    conjoint_triad2 = compute_conjoint_triad(str(seq2))
    return [*features1.values(), *features2.values(), *conjoint_triad1, *conjoint_triad2]

def prepare_data(positive_file: str, negative_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare data from FASTA files of interacting and non-interacting protein pairs."""
    positive_pairs = load_fasta(positive_file)
    negative_pairs = load_fasta(negative_file)
    
    data = []
    labels = []
    
    for pair in positive_pairs + negative_pairs:
        seq1, seq2 = pair.seq[:len(pair.seq)//2], pair.seq[len(pair.seq)//2:]
        data.append(extract_pair_features(seq1, seq2))
        labels.append(1 if pair in positive_pairs else 0)
    
    # Dynamically determine feature names
    sample_seq = positive_pairs[0].seq[:len(positive_pairs[0].seq)//2]
    feature_names = list(extract_features(sample_seq).keys()) * 2 + [f"CT_{i}" for i in range(len(compute_conjoint_triad(str(sample_seq))))]
    
    return np.array(data), np.array(labels), feature_names

def load_and_preprocess_data(positive_file: str, negative_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess data from positive and negative interaction files."""
    X, y, feature_names = prepare_data(positive_file, negative_file)
    return X, y, feature_names

def augment_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Augment the dataset using techniques like SMOTE for imbalanced classes."""
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_augmented, y_augmented = smote.fit_resample(X, y)
    return X_augmented, y_augmented

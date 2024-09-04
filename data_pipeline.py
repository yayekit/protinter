import numpy as np
from Bio import SeqIO
from typing import Tuple, List, Dict
from features import extract_features, compute_conjoint_triad
from imblearn.over_sampling import SMOTE

def load_fasta(file_path: str) -> List[SeqIO.SeqRecord]:
    """Load sequences from a FASTA file."""
    return list(SeqIO.parse(file_path, "fasta"))

def extract_pair_features(seq1: str, seq2: str) -> List[float]:
    """Extract features for a pair of protein sequences."""
    features1 = extract_features(seq1)
    features2 = extract_features(seq2)
    conjoint_triad1 = compute_conjoint_triad(str(seq1))
    conjoint_triad2 = compute_conjoint_triad(str(seq2))
    return [*features1.values(), *features2.values(), *conjoint_triad1, *conjoint_triad2]

def split_sequence(seq: SeqIO.SeqRecord) -> Tuple[str, str]:
    """Split a sequence record into two halves."""
    midpoint = len(seq.seq) // 2
    return str(seq.seq[:midpoint]), str(seq.seq[midpoint:])

def process_pairs(pairs: List[SeqIO.SeqRecord], label: int) -> Tuple[List[List[float]], List[int]]:
    """Process a list of sequence pairs and return features and labels."""
    data = []
    labels = []
    for pair in pairs:
        seq1, seq2 = split_sequence(pair)
        data.append(extract_pair_features(seq1, seq2))
        labels.append(label)
    return data, labels

def get_feature_names(sample_seq: str) -> List[str]:
    """Generate feature names based on a sample sequence."""
    feature_dict = extract_features(sample_seq)
    ct_length = len(compute_conjoint_triad(sample_seq))
    return list(feature_dict.keys()) * 2 + [f"CT_{i}" for i in range(ct_length)]

def prepare_data(positive_file: str, negative_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare data from FASTA files of interacting and non-interacting protein pairs."""
    positive_pairs = load_fasta(positive_file)
    negative_pairs = load_fasta(negative_file)
    
    pos_data, pos_labels = process_pairs(positive_pairs, 1)
    neg_data, neg_labels = process_pairs(negative_pairs, 0)
    
    all_data = pos_data + neg_data
    all_labels = pos_labels + neg_labels
    
    sample_seq, _ = split_sequence(positive_pairs[0])
    feature_names = get_feature_names(sample_seq)
    
    return np.array(all_data), np.array(all_labels), feature_names

def load_and_preprocess_data(positive_file: str, negative_file: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess data from positive and negative interaction files."""
    return prepare_data(positive_file, negative_file)

def augment_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Augment the dataset using SMOTE for imbalanced classes."""
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X, y)

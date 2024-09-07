from data_pipeline import load_and_preprocess_data, augment_data
from feature_selection import select_features
from model import train_model_cv, evaluate_model, save_model, compare_models, optimize_hyperparameters
from interpretation import explain_model
from ensemble import create_ensemble
from visualization import plot_feature_importance, plot_confusion_matrix, plot_correlation_matrix

#!/usr/bin/env python3
import argparse
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from features import extract_features
from model import load_model, predict_new_data, train_model_cv, save_model
from data_pipeline import load_and_preprocess_data, augment_data
import numpy as np

def generate_model(positive_file: str, negative_file: str):
    """Generate and save the protein interaction model."""
    print("Loading and preprocessing data...")
    X, y, feature_names = load_and_preprocess_data(positive_file, negative_file)
    
    print("Augmenting data...")
    X_augmented, y_augmented = augment_data(X, y)
    
    print("Training model...")
    model, scaler = train_model_cv(X_augmented, y_augmented)
    
    print("Saving model...")
    save_model(model, scaler, "protein_interaction_model.joblib")
    
    print("Model generation complete. Saved as 'protein_interaction_model.joblib'")

def main():
    parser = argparse.ArgumentParser(description="Predict protein interactions or generate model.")
    parser.add_argument("--generate", action="store_true", help="Generate new model")
    parser.add_argument("--positive", help="Path to positive interaction FASTA file (for model generation)")
    parser.add_argument("--negative", help="Path to negative interaction FASTA file (for model generation)")
    parser.add_argument("sequence_file", nargs="?", help="Path to the protein sequence file (FASTA format) for prediction")
    args = parser.parse_args()

    if args.generate:
        if not args.positive or not args.negative:
            print("Error: Both --positive and --negative files are required for model generation.")
            sys.exit(1)
        generate_model(args.positive, args.negative)
    elif args.sequence_file:
        try:
            # Load the sequence
            with open(args.sequence_file, "r") as handle:
                record = next(SeqIO.parse(handle, "fasta"))
                sequence = record.seq
        except FileNotFoundError:
            print(f"Error: File '{args.sequence_file}' not found.")
            sys.exit(1)
        except StopIteration:
            print(f"Error: No sequences found in '{args.sequence_file}'.")
            sys.exit(1)

        # Extract features
        features = extract_features(sequence)
        X_new = np.array([list(features.values())])

        # Load the pre-trained model
        try:
            model, scaler = load_model("protein_interaction_model.joblib")
        except FileNotFoundError:
            print("Error: Pre-trained model not found. Please ensure 'protein_interaction_model.joblib' is in the current directory.")
            sys.exit(1)

        # Make prediction
        prediction = predict_new_data(model, scaler, X_new)

        # Print result
        result = "likely to interact" if prediction[0] == 1 else "unlikely to interact"
        print(f"The protein sequence in '{args.sequence_file}' is {result}.")
    else:
        print("Error: Please provide either --generate with --positive and --negative files, or a sequence file for prediction.")
        sys.exit(1)

if __name__ == "__main__":
    main()
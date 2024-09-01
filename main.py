from data_pipeline import load_and_preprocess_data, augment_data
from feature_selection import select_features
from model import train_model_cv, evaluate_model, save_model, compare_models, optimize_hyperparameters
from interpretation import explain_model
from ensemble import create_ensemble
from visualization import plot_feature_importance, plot_confusion_matrix, plot_correlation_matrix

def main():
    positive_file = "positive_interactions.fasta"
    negative_file = "negative_interactions.fasta"
    
    X, y, feature_names = load_and_preprocess_data(positive_file, negative_file)
    
    # Data augmentation
    X_augmented, y_augmented = augment_data(X, y)

    # Feature selection
    X_selected, selected_indices = select_features(X_augmented, y_augmented, k=50)

    # Hyperparameter optimization
    best_params = optimize_hyperparameters(X_selected, y_augmented)
    
    # Model training with optimized parameters
    model, scaler = train_model_cv(X_selected, y_augmented, params=best_params)

    # Model interpretation
    explain_model(model, X_selected, [feature_names[i] for i in selected_indices])

    # ... rest of the code ...

    # Save model and scaler
    save_model(model, scaler, "protein_interaction_model.joblib")

if __name__ == "__main__":
    main()
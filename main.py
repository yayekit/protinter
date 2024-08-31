from data_pipeline import load_and_preprocess_data
from model import train_model_cv, evaluate_model, save_model
from visualization import plot_feature_importance, plot_confusion_matrix, plot_correlation_matrix

def main():
    positive_file = "positive_interactions.fasta"
    negative_file = "negative_interactions.fasta"
    
    X, y, feature_names = load_and_preprocess_data(positive_file, negative_file)
    
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
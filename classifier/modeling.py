import numpy as np
from sklearn.model_selection import train_test_split
from classifier.data_processing import preprocess_data, encode_labels
from classifier.model_training import train_model_with_cv, train_model, save_model
from classifier.evaluation import evaluate_model


def run_classification_pipeline(data, models_to_test, test_size=0.25, n_splits=5, random_state=42):
    """
    Runs the complete machine learning pipeline from preprocessing data to model training, evaluation, and saving.

    Args:
        models_to_test (list): List of model names to test.
        data (pd.DataFrame): The input dataset to be processed.
        test_size (float): The size of the test set. Default is 0.25.
        n_splits (int): The number of folds for k-fold cross-validation. Default is 5.
        random_state (int): Random seed for reproducibility. Default is 42.

    Returns:
        None
    """

    # Step 1: Preprocess the data
    processed_data = preprocess_data(data)
    X, y = encode_labels(processed_data)

    # Step 2: Perform k-fold cross-validation and compute performance for each model
    cv_scores = {model: train_model_with_cv(X, y, n_splits=n_splits, model=model) for model in models_to_test}

    # Step 3: Get the best model based on the mean cross-validation accuracy
    best_model_name, _ = max(
        ((model, np.mean(accuracy)) for model, accuracy in cv_scores.items()),
        key=lambda x: x[1]
    )

    # Step 4: Train the best model on the full training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    best_model = train_model(X_train, y_train, best_model_name)

    # Step 5: Evaluate the best model on the test set
    print(f"\n\nBest model (\033[1m {best_model_name} \033[0m) evaluation metrics:")
    evaluate_model(best_model, X_test, y_test)

    # Step 6: Save the best model
    save_model(best_model, model_name=best_model_name)

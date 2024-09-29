import os
import pandas as pd

from classifier.data_processing import load_data
from classifier.modeling import run_classification_pipeline
from config.config import load_env
from generator.synthetic_generator import create_email_data_generator


def load_or_generate_data(file_paths, generate_if_missing=False, num_emails=1000):
    """
    Loads data from the provided file paths. If the data is missing and `generate_if_missing` is True,
    it generates synthetic email data using the provided generator.

    Args:
        file_paths (list): List of file paths to load data from.
        generate_if_missing (bool): Whether to generate data if files are missing.
        num_emails (int): Number of emails to generate if needed.

    Returns:
        pd.DataFrame: The loaded or generated dataset.
    """
    valid_files = [file for file in file_paths if os.path.exists(file)]

    # Attempt to load data from valid file paths
    if valid_files:
        print(f"Loading data from {len(valid_files)} files...")
        data = load_data(valid_files)
        if data is not None:
            return data
    else:
        print("No valid data files found.")

    # If no files are found and generation is not enabled, raise an error
    if not generate_if_missing:
        raise FileNotFoundError(
            f"No valid data files found in {file_paths}, and 'generate_if_missing' is set to False.")

    print(f"Generating {num_emails} synthetic emails...")
    generator = create_email_data_generator()
    # Generate synthetic email data
    results = generator.generate(
        subject="Email Classification",
        extra="Emails must be in English, categories are: Spam, Work, Social, Promotion",
        runs=num_emails  # Number of synthetic emails to generate, with gpt 4o around €7.5 (1000 emails)
    )

    # Convert generated data to DataFrame
    data = pd.DataFrame([{'email': result.email, 'label': result.label} for result in results])

    # Save the generated data to a CSV file for future use
    data.to_csv('./data/synthetic_emails.csv', index=False)
    print("Synthetic data saved to 'data/synthetic_emails.csv'")

    return data


def main():
    # Load environment variables or other configuration settings
    load_env()

    # File paths for the datasets
    # In this case I generated 4 files to better track the costs of gpt 4o API.
    # So I read 4 files with the same features for a total of 1000 emails.
    # Delete content from ./data folder to trigger synthetic emails generation
    file_paths = [
        "./data/synthetic_emails_1.csv",  # 300 emails
        "./data/synthetic_emails_2.csv",  # 300 emails
        "./data/synthetic_emails_3.csv",  # 300 emails
        "./data/synthetic_emails_4.csv",  # 100 emails
        "./data/synthetic_emails.csv"  # Hypothetical file with 1000 emails
    ]

    # List of models to test
    models_to_test = ['logistic', 'svm', 'gradient_boosting', 'random_forest']

    # Step 1: Load or generate the data
    data = load_or_generate_data(file_paths, generate_if_missing=True, num_emails=1000)

    # Step 2: Run classification pipeline with cross-validation
    run_classification_pipeline(data, models_to_test)


if __name__ == "__main__":
    main()

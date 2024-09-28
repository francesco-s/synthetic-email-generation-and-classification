import pandas as pd
from config.config import load_env
from synthetic_data.synthetic_generator import create_email_data_generator


def main():
    load_env()

    # Create the synthetic data generator
    generator = create_email_data_generator()

    # Generate synthetic email data
    results = generator.generate(
        subject="Email Classification",
        extra="Emails must be in English, categories are: Spam, Work, Social, Promotion",
        runs=1000  # 1000 synthetic emails, with gpt 4o around €7.5
    )

    # Convert results to a DataFrame
    data = [{'email': result.email, 'label': result.label} for result in results]
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df.to_csv('data/synthetic_emails_2.csv', index=False)
    print("Data saved to 'data/synthetic_emails.csv'")


if __name__ == "__main__":
    main()

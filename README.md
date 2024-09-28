# Synthetic Email Generation and Classification

This project focuses on building and evaluating a machine learning pipeline for email classification using synthetic data. It supports several classifiers such as Logistic Regression, Support Vector Machines (SVM), Gradient Boosting, and Random Forests. If no pre-existing data is found, synthetic email data is generated using a GPT-based email data generator.

## Table of Contents
- [Overview](#overview)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Environment Configuration](#environment-configuration)
- [Installation](#installation)
- [Docker Setup](#docker-setup)

## Overview

This project classifies emails into different categories such as Spam, Work, Social, and Promotion. It features:
- **Support for multiple models**: Logistic Regression, SVM, Gradient Boosting, and Random Forest.
- **Synthetic data generation**: If real email data is not available, the project uses a synthetic data generator (GPT-4-based).
- **Cross-validation**: The project uses k-fold cross-validation to evaluate model performance.

## Usage

### Synthetic Data Generation
The script will automatically load the provided CSV files (if available) from the `./data` directory. If no valid files are found, it will generate synthetic emails based on predefined prompts and save them for future use.

### Custom Data
To use your own data, replace the file paths in `main.py` with your custom dataset file paths. The data should be in CSV format with at least two columns: `email` (text content) and `label` (email category).

### Model Training and Evaluation
The script automatically runs k-fold cross-validation on the selected models (`Logistic Regression`, `SVM`, `Gradient Boosting`, and `Random Forest`) and displays the evaluation metrics (accuracy, precision, recall, F1-score).

## Project Structure

```
synthetic_email_classification/
├── classifier/
│   ├── model/
│   │   ├── data_processing.py        # Functions for data loading and preprocessing
│   │   ├── evaluation.py             # Model evaluation metrics
│   │   ├── model_training.py         # Functions to train models with cross-validation
│   │   ├── modeling.py               # Model pipeline creation and configuration
├── config/
│   └── config.py                     # Configuration file loader
├── data/                             
│   ├── synthetic_emails_1.csv        # Example synthetic email datasets
│   ├── synthetic_emails_2.csv
│   ├── synthetic_emails_3.csv
│   ├── synthetic_emails_4.csv
│   ├── synthetic_emails.csv          # Fallback dataset (generated if files don't exist)
├── generator/
│   └── synthetic_generator.py        # GPT-4-based synthetic email data generator
├── models/
│   └── email_model.py                # Pydantic model for LLM output
├── prompts/
│   └── prompt_template.py            # Prompt templates for the synthetic generator
├── example.env                       # Environment variables (configurations) - rename to ".env" in prod
├── main.py                           # Main entry point of the project
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker image setup
└── docker-compose.yml                # Docker-compose setup

```

## Environment Configuration
Make sure to have a `.env` file in your project root with Open AI key (used for synthetic email generation).

Example `.env`:
```
OPENAI_API_KEY=<your-gpt-api-key>
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/francesco-s/synthetic-email-generation-and-classification.git
cd synthetic-email-generation-and-classification
```

### 2. Install Python Dependencies
You can install the required packages via `requirements.txt`. Ensure you're using Python 3.8+ and have `pip` installed.
```bash
pip install -r requirements.txt
```

### 3. Run the Project
Run the main script to load the data (or generate it) and evaluate different classification models:
```bash
python main.py
```

## Docker Setup

You can run the entire project using Docker or Docker Compose for ease of use and reproducibility.

### 1. Clone the Repository
```bash
git clone https://github.com/francesco-s/synthetic-email-generation-and-classification.git
cd synthetic-email-generation-and-classification
```

### 2. Build the Docker Image

```bash
docker build -t email_classifier_app .
```

### 3. Run the Docker Container

```bash
docker run -it --rm email_classifier_app
```

### Docker Compose

Instead of managing Docker containers manually, you can use Docker Compose to simplify container orchestration.

1. **Build and Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

2. **Stop the Docker Compose containers**:
   ```bash
   docker-compose down
   ```
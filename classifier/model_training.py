from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

import pickle
import numpy as np
from time import time


# Model pipeline definition
def create_pipeline(classifier_type='logistic'):
    """
    Create a text processing and classification pipeline based on the specified classifier type.

    Parameters:
    - classifier_type: str, type of classifier to use. Options are 'logistic', 'random_forest', 'svm', 'gradient_boosting'.

    Returns:
    - Pipeline: a scikit-learn Pipeline object with the selected classifier.

    Raises:
    - ValueError: if an unknown classifier type is specified.
    """

    # Text vectorization setup
    vectorizer = TfidfVectorizer(
        min_df=1,
        max_df=0.94,
        stop_words="english",
        sublinear_tf=True,
        norm='l1',
        ngram_range=(1, 1)
    )

    # Mapping of classifier types to their respective instances
    classifiers = {
        'logistic': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(),
        'svm': SVC(kernel='linear'),
        'gradient_boosting': GradientBoostingClassifier()
    }

    # Retrieve the appropriate classifier, default to Logistic Regression (our baseline) if unknown
    clf = classifiers.get(classifier_type)
    if clf is None:
        raise ValueError(f"Unknown classifier type: {classifier_type}. Valid options are: {list(classifiers.keys())}.")

    # Create the pipeline with the selected classifier
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('clf', clf)
    ])

    return pipeline


# K-Fold cross-validation implementation
def train_model_with_cv(X, y, n_splits=5, model='logistic'):
    pipeline = create_pipeline(model)

    # Stratified K-Fold to preserve the distribution of the classes
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    start_time = time()

    # Perform cross-validation and collect accuracy scores
    cv_scores = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')

    end_time = time()

    print(f"\n Model: \033[1m {model} \033[0m")
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.3f}")
    print(f"Cross validation time: {end_time - start_time:.2f} seconds")

    return cv_scores


def train_model(X_train, y_train, model='logistic'):
    pipeline = create_pipeline(model)
    model = pipeline.fit(X_train, y_train)

    return model


def save_model(model, model_name='logistic', path='./classifier/model/'):
    with open(path + model_name + '.pickle', 'wb') as f:
        pickle.dump(model, f)

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import re
import pandas as pd
import pickle

# Download necessary datasets for nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

# Load the data
data1 = pd.read_csv("../data/synthetic_emails_1.csv")
data2 = pd.read_csv("../data/synthetic_emails_2.csv")
data3 = pd.read_csv("../data/synthetic_emails_3.csv")
data4 = pd.read_csv("../data/synthetic_emails_4.csv")

# Concatenate the various datasets
result = pd.concat([data1, data2, data3, data4])

# Set stopwords and the lemmatizer
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()


# Tokenization function
def tokenize(text):
    # 1. Normalization: convert to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

    # 2. Tokenization: split the text into words
    tokens = word_tokenize(text)

    # 3. Remove stop words
    words = [w for w in tokens if w not in stop_words]

    # 4. Lemmatize the words
    lemmed_words = [lemmatizer.lemmatize(w) for w in words]

    # Convert the token list back into a string
    text = " ".join(lemmed_words)
    return text


# Apply tokenization to the data
result['processed_emails'] = result['email'].apply(lambda x: tokenize(x))

# Define the variable X (processed emails) and y (labels)
X = result['processed_emails']
result['label_num'] = result['label'].map(
    {'Spam': 0, 'Promotion': 1, 'Work': 2, 'Social': 3})  # Mapping the labels
y = result['label_num']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define the TfidfVectorizer to vectorize the text
vectorizer = TfidfVectorizer(min_df=1, max_df=0.94, stop_words="english", sublinear_tf=True, norm='l1',
                             ngram_range=(1, 1))

# Create the pipeline with TfidfVectorizer, SelectKBest (chi2), and RandomForestClassifier
pipeline = Pipeline([
    ('vect', vectorizer),  # Text vectorization
    ('chi', SelectKBest(chi2, k=1000)),  # Feature selection
    ('clf', RandomForestClassifier())  # Random Forest model
])

# Train the model
model = pipeline.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('RandomForest.pickle', 'wb') as f:
    pickle.dump(model, f)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["spam", "promotion", "work", "personal"]))

# Print the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

nltk.download('stopwords')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


# Load datasets and concatenate them
def load_data(file_paths):
    data_frames = [pd.read_csv(file) for file in file_paths]
    return pd.concat(data_frames)


# Tokenization function
def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words and lemmatize
    lemmed_words = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    # Convert list of tokens back to a string
    return " ".join(lemmed_words)


# Apply tokenization
def preprocess_data(data):
    data['processed_emails'] = data['email'].apply(tokenize)
    return data


# Label encoding
def encode_labels(data):
    label_map = {'Spam': 0, 'Promotion': 1, 'Work': 2, 'Social': 3}
    data['label_num'] = data['label'].map(label_map)
    return data['processed_emails'], data['label_num']

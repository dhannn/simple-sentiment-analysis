import pandas as pd
import re
import string
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from spellchecker import SpellChecker


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Remove HTML tags
    df['text'] = df['text'].apply(lambda x: re.sub(r'<[^>]+>', '', x))

    # Remove URLs
    df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+', '', x))

    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    df['text'] = df['text'].apply(lambda x: emoji_pattern.sub(r'', x))

    # Remove punctuation
    df['text'] = df['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # Lowercase text
    df['text'] = df['text'].apply(lambda x: x.lower())

    # Remove accents
    df['text'] = df['text'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ASCII', 'ignore').decode())

    # Remove unnecessary characters
    df['text'] = df['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

    # Stemming
    stemmer = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: [stemmer.stem(word) for word in x])

    # Spell checking and correction
    spell = SpellChecker()
    df['text'] = df['text'].apply(lambda x: [spell.correction(word) for word in x])

    return df

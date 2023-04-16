import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Remove unnecessary characters
    df['text'] = df['text'].apply(lambda x: re.sub('[^a-zA-Z0-9\s]', '', x))
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    df['text'] = df['text'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
    
    # Stemming
    stemmer = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: [stemmer.stem(word) for word in x])
    
    return df

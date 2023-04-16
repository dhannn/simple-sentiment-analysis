import pandas as pd
from sklearn.naive_bayes import MultinomialNB

'''
    Takes a features of training set (X), labels of training set (y)
    and returns the multinomial Naive Bayes classifier 
'''
def naive_bayes(X_train: pd.DataFrame, y_train: pd.DataFrame) -> MultinomialNB:
    
    # Train the model on the training data set
    X_train, y_train = train_test_split(X, y, random_state = 15)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    return classifier
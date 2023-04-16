import pandas as pd
from sklearn.svm import SVC
'''
    Takes a features of training set (X) and labels of training set (y)
    and returns the multinomial SVC
'''
def support_vector_machine(X_train: pd.DataFrame, y_train: pd.DataFrame) -> SVC:
    svc = SVC(kernel='linear', C=1.0, random_state=42) # initialize SVC with linear kernel and default parameters
    svc.fit(X_train, y_train) # train the SVC on the training set
    return svc # return the trained SVC model


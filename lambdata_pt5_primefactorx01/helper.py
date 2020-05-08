import pandas as pd
import numpy as np
from sklearn import metrics

def print_nulls(df:pd.DataFrame):
    number_of_nulls = df.isna().sum()
    print(f'Nulls:\n{number_of_nulls}')
    print(df.head())

def predict(X: pd.DataFrame):
    length = X.shape[0]
    return np.random.choice([True, False], length)

def print_confusion_matrix(y, y_pred):
    df=pd.DataFrame(metrics.confusion_matrix(y, y_pred), columns = ['positive', 'negative'], index=['True', 'False'])
    print(df)


if __name__ == "__main__":
    X, y = np.arange(10).reshape((5, 2)), [True, False,True, False, True]

    pred = predict(X)

    print_confusion_matrix(y, pred)

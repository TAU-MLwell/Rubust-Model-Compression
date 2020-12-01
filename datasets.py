import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def prepare_val(X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.15)
    return X_train, X_val, y_train, y_val


def prepare_data(dataset, return_test=True):
    X_test = None
    y_test = None

    if dataset == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True)
    elif dataset == 'dermatology':
        X, y = load_dermatology()
    elif dataset == 'heart':
        X, y = load_heart_cleveland()
    elif dataset == 'arrhythmia':
        X, y = load_arrhythmia()
    elif dataset == 'iris':
        X, y = load_iris()
    else:
        raise ValueError('No such dataset')

    if return_test:
        X, X_test, y, y_test = train_test_split(X, y, random_state=0, test_size=0.15)

    return X, y, X_test, y_test


def get_number_of_classes(dataset):
    if dataset == 'breast_cancer':
        return 2
    elif dataset == 'dermatology':
        return 6
    elif dataset == 'heart':
        return 5
    elif dataset == 'arrhythmia':
        return 16
    elif dataset == 'iris':
        return 3
    else:
        raise ValueError('No such dataset')


def load_dermatology():
    df = pd.read_csv('data/dermatology.csv')

    # replace missing values with mean
    df.replace('?', np.nan, inplace=True)
    df = df.astype(float)
    cols = list(df)
    for col in cols:
        df[col].fillna(df[col].mean(), inplace=True)

    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].astype(int).to_numpy()
    # classes are 1-6, changing to 0-5
    y = y - 1

    return X, y


def load_heart_cleveland():
    df = pd.read_csv('data/heart_cleveland.csv')

    # replace missing values with mean
    df.replace('?', np.nan, inplace=True)
    df = df.astype(float)
    cols = list(df)
    for col in cols:
        df[col].fillna(df[col].mean(), inplace=True)

    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].astype(int).to_numpy()

    return X, y


def load_arrhythmia():
    df = pd.read_csv('data/arrhythmia.csv')

    # replace missing values with mean
    df.replace('?', np.nan, inplace=True)
    df = df.astype(float)
    cols = list(df)
    for col in cols:
        df[col].fillna(df[col].mean(), inplace=True)

    X = df.iloc[:, :-1].to_numpy() # might need to remove -2 instead
    y = df.iloc[:, -1].astype(int).to_numpy()
    # classes starting from 1
    y = y - 1

    return X, y


def load_iris():
    df = pd.read_csv('data/iris.csv')

    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].astype(int).to_numpy()

    return X, y
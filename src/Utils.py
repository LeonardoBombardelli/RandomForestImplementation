import pandas as pd
import numpy as np


def bootstrap(dataset: pd.DataFrame, n: int):
    trainData = dataset.sample(n, replace=True)
    testData  = dataset.loc[~dataset.index.isin(trainData.index)]
    return trainData, testData

def generate_kfolds(dataset: pd.DataFrame, k: int):
    foldSize = int(dataset.shape[0] / k)
    folds = []

    for i in range(k-1):
        fold = dataset.sample(foldSize)
        folds.append(fold)
        dataset = dataset.loc[~dataset.index.isin(fold.index)]

    folds.append(dataset)
    return folds

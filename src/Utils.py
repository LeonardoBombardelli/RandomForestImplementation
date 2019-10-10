import pandas as pd
import numpy as np


def bootstrap(dataset: pd.DataFrame, n: int):
    trainData = dataset.sample(n, replace=True)
    testData  = dataset.loc[~dataset.index.isin(trainData.index)]
    return trainData.reset_index(drop=True), testData.reset_index(drop=True)

def generate_kfolds(dataset: pd.DataFrame, target: str, k: int):
    N = int(len(dataset) / k)
    folds = []

    for i in range(k-1):
        datasetLen = len(dataset)
        df = dataset.sample(1).groupby(target, group_keys=False, sort=False).apply(
             lambda x: x.sample(int(len(x)*N/len(dataset))))

        dataset = dataset.loc[~dataset.index.isin(df.index)]

        samples_left = int(N - len(df))
        if samples_left > 0:
            df = pd.concat([df, dataset.sample(samples_left)])

        dataset = dataset.loc[~dataset.index.isin(df.index)]
        folds.append(df.reset_index(drop=True))

    folds.append(dataset.reset_index(drop=True))
    return folds

import pandas as pd
import numpy as np


def bootstrap(dataset: pd.DataFrame, n: int):
    trainData = dataset.sample(n=n, replace=True)
    testData  = dataset.loc[~dataset.index.isin(trainData.index)]
    return trainData, testData

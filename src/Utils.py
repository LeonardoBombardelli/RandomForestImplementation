import pandas as pd
import numpy as np


def calculateF1(results: dict):
    sum  = 0.0
    size = 0.0

    for c in results:
        if not results[c]['eval']:
            continue
        prec = 1.0
        rev  = 1.0

        if results[c]['VP'] != 0 or results[c]['FP'] != 0:
            prec = float(results[c]['VP']) / float(results[c]['VP'] + results[c]['FP'])

        if results[c]['VP'] != 0 and results[c]['FN'] != 0:
            rev  = float(results[c]['VP']) / float(results[c]['VP'] + results[c]['FN'])

        sum += 2.0 * ((prec * rev) / (prec + rev))
        size += 1.0

    return sum / size

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

def getSeparator(path: str):
    line = ''

    with open(path) as file:
        line = next(file)

    if line.find(',') != -1:
        return ','
    else:
        return ';'

def readCSV(path: str):
    """
    WARNING: WE NEED TO LOAD OUR DATASET FOR BOTH TRAIN AND EVAL INSTANCES IN THE SAME CALL OF READCSV
    THIS MAKES SENSE IN OUR ASSIGNMENT SINCE WE'RE NOT SAVING OUR MODELS ANYWHERE AND WE ONLY RUN READCSV ONCE
    BUT IF WE RUN IT TWICE, ONE TIME FOR THE TRAIN DATASET AND THE OTHER TIME FOR THE EVAL DATASET, WE MIGHT
    LOAD THEM WITH DIFFERENT LABELS.
    It could be solved by making the "mapping" variable in this code a permanent atribute of our DecisionTree, though.
    """
    newDF = pd.read_csv(path, sep=getSeparator(path))
    for key in newDF.columns.values.tolist():
        if(newDF[key].dtype.name == "int64" or newDF[key].dtype.name == "float64"):
            newDF = numericalColumnToNumber(newDF, key)
        else:
            newDF = categoricalColumnToNumber(newDF, key)
    return(newDF)

def categoricalColumnToNumber(newDF, key):
    labels = newDF[key].unique().tolist()
    mapping = dict( zip(labels,range(len(labels))) )
    newDF.replace({key: mapping},inplace=True)
    return(newDF)

def numericalColumnToNumber(newDF, key):
    """
    TODO: Generalize cutting point for N different classes
    """
    cuttingPoint = newDF[key].median()
    returnList = []
    for value in newDF[key]:
        if(value > cuttingPoint):
            returnList.append(1)
        else:
            returnList.append(0)
    newDF[key] = pd.Series(returnList)
    return(newDF)

if __name__ == "__main__":
    #print(readCSV("datasets/dadosBenchmark_validacaoAlgoritmoAD.csv"))
    print(readCSV("datasets/german-credit.csv"))

#!/usr/bin/env python3

from RandomForest import RandomForest
import pandas as pd
from Utils import bootstrap, generate_kfolds, readCSV
import argparse
from math import sqrt, floor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', nargs=1, type=str, required=True)
    parser.add_argument('-target', nargs=1, type=str, required=True)
    parser.add_argument('-alias', nargs=1, type=str, required=True)
    arguments = parser.parse_args()

    newDF = readCSV(arguments.dataset[0])

    nTrees = [5, 10, 20, 30, 40, 50]
    k = 10
    m = floor(sqrt(newDF.shape[1]))

    for n in nTrees:
        print("-----------------")
        print("Number of trees: " + str(n))
        newDF = readCSV(arguments.dataset[0])

        folds = generate_kfolds(newDF, arguments.target[0], k)

        for i in range(k):
            trainList = [x for j, x in enumerate(folds) if j != i]
            trainDF = pd.concat(trainList)
            testDF = folds[i]

            randForest = RandomForest()
            randForest.train(trainDF, arguments.target[0], n,
                            m, False)
            newStr = randForest.eval(testDF)
            with open("outputResults/" + str(n) + arguments.alias[0] + ".txt", "a") as outputFile:
                outputFile.write(newStr)

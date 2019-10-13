import argparse

import pandas as pd
import numpy as np

from DecisionTree import DecisionTree
from Utils import *

class RandomForest():
    def __self___(self):
        self.dataset = None
        self.targetClass = None
        self.trees = []

    def train(self, dataset: pd.DataFrame, targetClass: str, n: int, m: int, verbose: bool):
        self.dataset = dataset
        self.targetClass = targetClass
        self.trees = []

        for i in range(n):
            treeData, _ = bootstrap(self.dataset, self.dataset.shape[0])
            tree = DecisionTree()
            tree.train(treeData, targetClass, m, verbose)
            self.trees.append(tree)

    def eval(self, testData: pd.DataFrame):
        testDataSize = testData.shape[0]
        targetClassValues = np.unique(testData[self.targetClass])
        targetClassValues = np.append(targetClassValues,
                            np.unique(self.dataset[self.targetClass]))
        targetClassValues = np.unique(targetClassValues)

        testLabels = testData[self.targetClass].values
        outputLabels = []

        treesEvaluations = []
        for tree in self.trees:
            treesEvaluations.append(tree.eval(testData))

        hits = 0.0
        results = {}

        for c in targetClassValues:
            results[c] = {'VP': 0, 'FP': 0, 'FN': 0, 'eval': False}


        # print(treesEvaluations)
        treesEvaluations = np.asarray(treesEvaluations)
        for i in range(testDataSize):
            treesVotes = treesEvaluations[:,i]
            classes, totalVotes = np.unique(treesVotes, return_counts=True)
            outputClass = classes[np.argmax(totalVotes)]
            instanceLabel = testData.iloc[i][self.targetClass]

            outputLabels.append(outputClass)
            results[outputClass]['eval'] = True
            results[instanceLabel]['eval'] = True
            if outputClass == instanceLabel:
                hits += 1
                results[outputClass]['VP'] += 1
            else:
                results[outputClass]['FP'] += 1
                results[instanceLabel]['FN'] += 1

        f1 = calculateF1(results)
        print('TEST SIZE  := %d' % testDataSize)
        print('ACCURACY   := %f' % (hits / testDataSize))
        print('F1         := %f' % f1)
        newStr = '\nTEST SIZE := ' + str(testDataSize) + "\nACCURACY  := " + str(hits / testDataSize) + '\nF1        := ' + str(f1)
        return(newStr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-k', nargs=1, type=int, required=True)
    parser.add_argument('-m', nargs=1, type=int, required=True)
    parser.add_argument('-n', nargs=1, type=int, required=True)
    parser.add_argument('-dataset', nargs=1, type=str, required=True)
    parser.add_argument('-target', nargs=1, type=str, required=True)
    arguments = parser.parse_args()

    # df = pd.DataFrame(
    #     {
    #         "label1": [1, 2, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2],
    #         "label2": [7, 8, 9, 10, 11, 7, 8, 11, 9, 10, 6, 8, 10, 7, 8, 9, 11, 9],
    #         "label3": [13, 14, 15, 16, 17, 15, 14, 13, 17, 17, 11, 15, 16, 11, 14, 21, 13, 12],
    #         "target": [5, 3, 4, 5, 2, 5, 2, 5, 3, 4, 3, 4, 2, 3, 4, 3, 5, 2]
    #     }
    # )

    df = readCSV(arguments.dataset[0], arguments.target[0])

    print(df.shape)
    # Just for testing
    folds = generate_kfolds(df, arguments.target[0], arguments.k[0])

    for i in range(arguments.k[0]):
        trainList = [x for j, x in enumerate(folds) if j != i]
        trainDF = pd.concat(trainList)
        testDF = folds[i]

        randForest = RandomForest()
        randForest.train(trainDF, arguments.target[0], arguments.n[0],
                         arguments.m[0], arguments.verbose)
        randForest.eval(testDF)

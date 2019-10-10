import argparse

import pandas as pd
import numpy as np

from DecisionTree import DecisionTree
from Utils import bootstrap, generate_kfolds

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
        treesEvaluations = []
        for tree in self.trees:
            treesEvaluations.append(tree.eval(testData))

        hits = 0.0
        treesEvaluations = np.asarray(treesEvaluations)
        for i in range(testDataSize):
            treesVotes = treesEvaluations[:,i]
            classes, totalVotes = np.unique(treesVotes, return_counts=True)
            outputClass = classes[np.argmax(totalVotes)]
            instanceLabel = testData.iloc[i][self.targetClass]

            if outputClass == instanceLabel:
                hits += 1

        # TODO: compute other metrics
        print('ACCURACY := %f' % (hits / testDataSize))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-k', nargs=1, type=int, required=True)
    parser.add_argument('-m', nargs=1, type=int, required=True)
    parser.add_argument('-n', nargs=1, type=int, required=True)
    arguments = parser.parse_args()

    df = pd.DataFrame(
        {
            "label1": [1, 2, 1, 1, 1, 2, 2, 1, 2, 2],
            "label2": [7, 8, 9, 10, 11, 7, 8, 11, 9, 10],
            "label3": [13, 14, 15, 16, 17, 15, 14, 13, 17, 17],
            "target": [5, 3, 5, 5, 5, 3, 3, 4, 2, 3]
        }
    )

    # Just for testing
    folds = generate_kfolds(df, arguments.k[0])

    for i in range(arguments.k[0]):
        trainList = [x for j, x in enumerate(folds) if j != i]
        trainDF = pd.concat(trainList)
        testDF = folds[i]

        randForest = RandomForest()
        randForest.train(trainDF, "target", arguments.n[0],
                         arguments.m[0], arguments.verbose)
        randForest.eval(testDF)

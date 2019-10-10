import random
import pandas as pd
import numpy as np
from math import log2


class DecisionTree():
    def __init__(self):
        self.dataset = None # Expected: pd.DataFrame
        self.targetClass = None # Expected: str
        self.firstNode = None # Expected: DecisionTreeNode
        self.atributes = None # Expected: list()

    def train(self, dataset: pd.DataFrame, targetClass: str, m: int, verbose: bool):
        if(dataset.shape[0] != dataset[targetClass].size):
            raise Exception("Number of rows in dataset != nmumber of labels in target class")

        self.dataset = dataset
        self.targetClass = targetClass
        self.atributes = dataset.columns.values.tolist()

        if verbose:
            print('DecisionTree\'s train data:')
            print(self.dataset)
        self.firstNode = DecisionTreeNode(self.dataset, self.targetClass, m, verbose)

    def eval(self, dataToEval: pd.DataFrame):
        for column in dataToEval.columns.values.tolist():
            if(column not in self.atributes):
                raise Exception("Not all columns is an atribute in the Decision Tree!")

        returnList = []
        for _, row in dataToEval.iterrows():
            returnList.append(self._evalOneInstance(row))
        return(returnList)

    def _evalOneInstance(self, instance):
        instanceClass = None
        actualNode = self.firstNode

        while(instanceClass == None):
            if(actualNode.predictedClass != None):
                instanceClass = actualNode.predictedClass
            else:
                try:
                    if actualNode.divisionColumn is not None and \
                       instance[actualNode.divisionColumn] in actualNode.childs:
                        actualNode = actualNode.childs[instance[actualNode.divisionColumn]]
                    else:
                        return actualNode.majorityClass()
                except:
                    return actualNode.majorityClass()

        return(instanceClass)

class DecisionTreeNode():
    def __init__(self, dataset: pd.DataFrame, targetClass: str, m: int, verbose: bool):
        self.dataset = dataset
        self.targetClass = targetClass
        self.divisionColumn = None # Expected: str, but only if it will split into more nodes
        self.predictedClass = None # Expected: any class or None
        self.childs = {} # The keys to the dics will be every value the atribute can have
        self.m = m       # m attributes to choose from

        if(self.dataset[self.targetClass].count() == 0):
            self.predictedClass = -1
            return

        # If all instances in target column have the same class, returns this class
        if(self.dataset[self.targetClass].nunique() == 1):
            self.predictedClass = self.dataset[self.targetClass].tolist()[0]
            return

        # If there is only one atribute, we can't split the tree anymore
        if(len(dataset.columns) == 1):
            self.predictedClass = self.dataset[self.targetClass].value_counts().index.tolist()[0]
            return

        self.divisionColumn = self._bestDivisionCriteria()
        if(verbose):
            print('-------------')
            print(self.divisionColumn)
            print('-------------')

        if self.divisionColumn is not None:
            for key, datasetGroup in self.dataset.groupby(self.divisionColumn):
                self.childs[key] = DecisionTreeNode(datasetGroup.drop(self.divisionColumn, axis=1),
                                                    self.targetClass, m, verbose)
        else:
            self.predictedClass = self.majorityClass()

        return

    def majorityClass(self):
        targetArray = self.dataset[self.targetClass].to_numpy()
        classes, totalVotes = np.unique(targetArray, return_counts=True)
        return classes[np.argmax(totalVotes)]

    def _bestDivisionCriteria(self):
        atributes = list(self.dataset.columns.values)
        atributes.pop(atributes.index(self.targetClass))
        atributes = random.sample(atributes, min(self.m, len(atributes)))
        generalEntropy = self._entropy(self.dataset[self.targetClass])

        selectedAtribute = None
        maxGain = 0
        for atribute in atributes:
            newGain = self._gainForOneClass(atribute, generalEntropy)
            if(newGain > maxGain):
                maxGain = newGain
                selectedAtribute = atribute

        return(selectedAtribute)

    def _entropy(self, column: pd.Series):
        n = column.count()
        valueCounts = column.value_counts().tolist()
        result = 0
        for x in valueCounts:
            result += (-x/n) * log2(x/n)
        return(result)

    def _gainForOneClass(self, atribute, generalEntropy: float):
        n = self.dataset[self.targetClass].count()
        meanEntropy = 0

        for _, group in self.dataset.groupby(atribute):
            groupSize = group[self.targetClass].count()
            meanEntropy += (groupSize/n) * self._entropy(group[self.targetClass])

        return(generalEntropy - meanEntropy)


if __name__ == "__main__":
    testDF = pd.DataFrame(
        {
            "label1": [1, 2, 1, 1, 1],
            "label2": [7, 8, 9, 10, 11],
            "label3": [13, 14, 15, 16, 17],
            "target": [5, 3, 5, 5, 5]
        }
    )


    decTree = DecisionTree()
    decTree.train(testDF, "target", 2)

    evalDF = pd.DataFrame(
        {
            "label1": [1],
            "label2": [5],
            "label3": [8]
        }
    )
    print(decTree.eval(evalDF))

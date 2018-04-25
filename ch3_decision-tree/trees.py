from math import log
import operator
import pickle

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannoEnt(dataSet):
    # The more labels you have, the bigger the entropy will be.
    numEntries = len(dataSet)
    labelCounts = dict()
    for featVec in dataSet:
        currentLable = featVec[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1
    shannoEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannoEnt -= prob * log(prob, 2)
    return shannoEnt

def splitDataSet(dataSet, axis, value):
    '''
    >>> myDat
    [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    >>> trees.splitDataSet(myDat, 0, 1)
    [[1, 'yes'], [1, 'yes'], [0, 'no']]
    >>> trees.splitDataSet(myDat, 0, 0)
    [[1, 'no'], [1, 'no']]
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # take featVec[axis] out from feature vector, and append it to return DataSet
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannoEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # In this example uniqueVals = {0, 1} as there's only 0s and 1s in feature vector.
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            #print(calcShannoEnt(subDataSet))
            newEntropy += prob * calcShannoEnt(subDataSet)
            # print(newEntropy)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = dict()
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

def classify(inputTree, featLabels, testVec):
    # To judge if provided test vector lenght is equal to feature lables, if not then raise execption as it's supposed 
    # to be the same.
    if len(testVec) != len(featLabels):
        raise Exception("Test vector should have same dimension of feature labels")
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # featIndex = 0, because featLabels[0] = 'no surfacing', and program will start with this label first as it's the
    # most outside judgement.
    featIndex = featLabels.index(firstStr)
    # secondDict.keys() = [0, 1]
    for key in secondDict.keys():
        # testVec[0] = 1 if given testVec = [1, 0]
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    with open(filename, 'wb') as treeFile:
        pickle.dump(inputTree, treeFile)

def grabTree(filename):
    with open(filename, 'rb') as treeFile:
        return pickle.load(treeFile)
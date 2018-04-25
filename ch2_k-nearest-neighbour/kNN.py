import numpy as np
import operator
import matplotlib
from matplotlib import pyplot as plt
from os import listdir


def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    # Input:
    #   1. inX: input vector for classification (verification examples)
    #   2. dataSet: training examples
    #   3. labels: label vector (result of training examples)
    #   4. k: number of nearest neighour 
    #
    # How to calculate distance between two points A (1, 2) (x = 1 and y = 2) and B (3, 4) (x = 3 and y = 4) in 2D 
    # coordinate system: 
    #   sqrt((3 - 1) ** 2 + (4 - 2) ** 2)
    # Or A (1, 0, 0, 1) and B (7, 6, 9, 4)
    #   sqrt((7 - 1) ** 2 + (6 - 0) ** 2 + (9 - 0) ** 2 + (4 - 1) ** 2)
    #
    # Get number of training examples, result is something like (4, 2) whick means 4 is row number and 2 is column number, 
    # 4 training examples and each example has 2 features
    dataSetSize = dataSet.shape[0]
    # This is to get a matrix with same row number of training set which is full of input X vector and to get distance with 
    # all training examples
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # Firstly square the difference between each feature, which equals to (3 - 1) ** 2, (4 - 2) ** 2
    sqDiffMat = diffMat ** 2
    # Sum the difference up, (3 - 1) ** 2 + (4 - 2) ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    # Square root the summed difference
    distances = sqDistances ** 0.5
    # return a sorted accending index matrix.
    # E.g. matrix = [1.48660687 1.41421356 0.         0.1       ], according to accending rule from smallest, it should be 
    # [0.         0.1        1.41421356 1.48660687], and corresponding index of matrix should be [2 3 1 0] as follows:
    # matrix[2] = 0. 
    # matrix[3] = 0.1
    # matrix[1] = 1.41421356
    # matrix[0] =  1.48660687
    sortedDisIndicies = distances.argsort()
    classCount = dict()
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sort the dictionary with field number 1.
    # field number 0 is key, field number 1 is value, in our case is dict_items([('B', 2), ('A', 1)]), sort the array 
    # with value which is 2.
    # classCount = {'B': 2, 'A': 1}
    # classCount.items() = dict_items([('B', 2), ('A', 1)])
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  


def file2matrix(filename):
    with open(filename, 'r') as fr:
        arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        lastFromLine = line.split('\t')
        returnMat[index, :] = lastFromLine[0:3]
        classLabelVector.append(int(lastFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def ploting0():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 10.0 * np.array(datingLabels), 10.0 * np.array(datingLabels))
    plt.show()


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    testRatio = 0.20
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * testRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("The classifier came back with: {0}, the real number is: {1}".format(classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("The total error rate is: {}".format(errorCount / float(numTestVecs)))


def classifyPerson():
    resultList = ['Not at all', 'in small doses', 'in large doses']
    ffMiles = float(input("Frequent flyer miles earned per year: "))
    percentTats = float(input("Percentage of time spent in playing video game: "))
    iceCream = float(input("Liter of icecream consumed per year: "))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inputArray = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inputArray - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: {}".format(resultList[classifierResult - 1]))


#classifyPerson()
#datingClassTest()


#datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
#normMat, ranges, minVals = autoNorm(datingDataMat)
#print(normMat)
#ploting0()
#verification_example = [0, 0]
#group, labels = createDataSet()
#classify0(verification_example, group, labels, 3)  


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename, "r")
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            # keep adding read numbers from file to 1 row vector
            returnVect[0, 32 * i + j] = int(lineStr[j])
    fr.close()
    return returnVect


def handwritingClassTest():
    hwLables = []
    # return files of folder in a list like [1.txt, 2.txt]
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/{}'.format(fileNameStr))
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLables, 3)
        print("The classifier came back with: {0}, the real numer is: {1}".format(classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\nThe total number of error is: {}".format(errorCount))
    print("\nThe total error rate is: {}".format(errorCount / float(mTest)))


handwritingClassTest()
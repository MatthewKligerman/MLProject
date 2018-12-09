import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def getAccuracy(ypred, ytest):

    numTrue = 0
    numFalse = 0

    for val1, val2 in ypred, ytest:

        if val1 == 1:

            if '>50k' is in val2:
                numTrue += 1

            else:
                numFalse += 1

        else:

            if '>50k' is in val2:
                numFalse += 1

            else:
                numTrue +=1 

    return numTrue/(numTrue+numFalse)

#graph the data and support vectors
def graphIt():
    return

#calculates support vector:
def SVM(a, x, y):

    result = 0
    sumA = np.sum(a)

    return 0

#classifies whether person makes more or equal to/less than $50k/year
def classifyIncome(w, u, b):
    #call SVM; true = >50k; false = <=50k;
    if w*u+b>=0:
        return '>50k'
    else:
        return '<=50k'

#updates weights with sigmoid function
def getWeights(a, x, y):
    return 0

#tests accuracy on testing data
def testAccuracy():
    return 0

#select features based on how it affects test results
def ftSelect():
    return

#perform analysis to get optimal performance
def getResults():
    return

def getLikelihood(train, header):
    return 0

def avgProbability(train):

    trueSum = 0
    falseSum = 0

    for item in train.T[-1]:

        if '>50K' in item:
            trueSum += 1

        else:
            falseSum += 1

    return trueSum/(trueSum+falseSum)

def probability50k(train, header, label):

    trueSum = 0
    falseSum = 0

    for i, item in enumerate(train[:].T):

        if item[0] == header:

            for e, val in enumerate(item[1:]):

                if val == label:

                    if '>50K' in train.T[len(train[:][0])-1][e+1]:
                        trueSum += 1

                    else:
                        falseSum += 1

    return (trueSum/(trueSum+falseSum), trueSum, falseSum)

    #{label:(trueSum/(trueSum+falseSum), trueSum, falseSum)}

headers = []
trainData = []

with open("adult.train.csv") as csvfile:

    reader = csv.reader(csvfile) # change contents to floats

    for i, row in enumerate(reader): # each row is a list

        if i == 0:
            headers = row
        if ' ?' not in row:
            trainData.append(row)


print('Headers : ')
print(headers)
print('\n')

print('Training Data : ')
trainData = np.asarray(trainData)
print(trainData)

testData = []

with open("adult.test.csv") as csvfile:

    reader = csv.reader(csvfile) # change contents to floats

    for i, row in enumerate(reader): # each row is a list

        #if i == 0:
            #headers = row
        if ' ?' not in row:
            testData.append(row)


print('\n')
print('Testing Data : ')
testData = np.asarray(testData)
print(testData)

print('\n\n')
# Average for each feature independently: workclass, education, marital-status, occupation, relationship, race, sex,
# native-country
avgList = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'earnings']
avgs =  {}

# creates dictionary, each containing the header as a key with the value as an array of possible characteristics

for i, item in enumerate(trainData.T):

    if headers[i] in avgList:

        avgs[headers[i]] = []

        for info in item:

            if info not in avgs[headers[i]]:

                if '?' not in info and info not in headers[i]:

                    avgs[headers[i]].append(info)

probs = {}

# creates a dictionary, each containing a header as a key with an array of dictionaries as the values
# these inner dictionaries contain possible headers as keys and probabilities of making 50k or more as values
#for i, key in enumerate(avgs.keys()):

for item in trainData.T:

    if item[0] in avgs.keys():

        probs[item[0]] = {}

        for key in avgs[item[0]]:

            probs[item[0]][key] = probability50k(trainData, item[0], key)
    else:

        probs[item[0]] = {}
        probs[item[0]] = getLikelihood(trainData, item[0])

probs['other'] = avgProbability(trainData)

print()
print(probs)
print()
print()

dfTrain = pd.DataFrame(data = trainData[1:, :],  columns = trainData[0, :]) # index = trainData[1:, 0],
dfTest = pd.DataFrame(data = testData[1:, :], columns = testData[0, :])

count = 1

for column in list(dfTrain.columns):

    if column in avgList:

        for i, item in enumerate(dfTrain[column]):
            count += 1
            dfTrain[column][i] = probs[column][item][0]

    print(column)

#print(dfTrain)

xtrain = dfTrain.drop('earnings', axis=1)
print('shortened')
print(list(xtrain.values)[:][:10])
print('xtrain')
#print(list(xtrain.values))
#xtrain = xtrain[1:][:]
ytrain = dfTrain['earnings']
#ytrain = ytrain[1:]

#svclassifier = SVC(kernel='linear')
#svclassifier.fit(list(xtrain.values)[:][1:10], list(ytrain.values)[1:10])

for column in list(dfTest.columns)[:-1]:

    if column in avgList:

        for i, item in enumerate(dfTest[column]):
            count += 1
            if item not in probs[column].keys():
                dfTest[column][i] = probs['other']

            else:
                dfTest[column][i] = probs[column][item][0]

    print(column)

xtest = dfTest.drop('earnings', axis=1)
#xtest = xtest[1:][:]
ytest = dfTest['earnings']
#ytest= ytest[1:]
print('ytest : ')
print(ytest)

svclassifier = SVC(kernel='linear')
svclassifier.fit(list(xtrain.values)[:][:], list(ytrain.values)[:])
ypred = svclassifier.predict(list(xtest.values))

print(ypred)
skAccuracy = getAccuracy(ypred, list(ytest.values))
print('Accuracy', str(skAccuracy))

#print(dfTrain)

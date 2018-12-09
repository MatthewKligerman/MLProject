import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

#how to generate??
xvals = [];
yvals = [];

#graph the data and support vectors
def graphIt():
    plt.figure(figsize=(8,6))
    plt.scatter(xvals,yvals,marker='+',color='green')
    plt.show()

#calculates support vector
def SVM():
    return 0

#classifies whether person makes more or equal to/less than $50k/year
def classifyIncome():
    #call SVM; true = >50k; false = <=50k;
    return 0

#updates weights with sigmoid function
def mySigmoid():
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
    return 1

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

    return {label:(trueSum/(trueSum+falseSum), trueSum, falseSum)}

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
        if '?' not in row:
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
        probs[item[0]] = []

        for key in avgs[item[0]]:
            probs[item[0]].append(probability50k(trainData, item[0], key))

    else:
        probs[item[0]] = []
        probs[item[0]].append(getLikelihood(trainData, item[0]))



print()
print(probs)
print()
print()

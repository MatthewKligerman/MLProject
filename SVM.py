import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random as rnd
#import copy
import decimal

def drange(x, y, jump):

    fList = []

    while x <= y:
        fList += x
        yield float(x)
        x += decimal.Decimal(jump)

    return fList

def getAccuracy(ypred, ytest):

    numTrue = 0
    numFalse = 0

    for val1, val2 in zip(ypred, ytest):

        if int(val1) == 1 and '>50K' in val2:
            numTrue += 1
        elif int(val1) == 0 and '<=50K' in val2:
            numTrue += 1
        else:
            numFalse += 1

    return numTrue/(numTrue+numFalse)

#creates and shows bar graph
def barGraph(x, y, category):

    ylabel = drange(0.0, 1.0, 0.05)
    x_pos = range(len(x))

    plt.figure(figsize=(10,6))
    plt.bar(x = x_pos, height = y, width = 0.25, bottom = 0, align='center', alpha=0.5, orientation='vertical')#, tick_label = ylabel)
    plt.xticks(x_pos, x)
    plt.ylabel('Probability')
    plt.xlabel(category)
    plt.title('Probability of making >$50k for each '+category)

    plt.show()

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


#CLASSIFY STARTS
def fit(X, y):
    # Initialization
    n, d = X.shape[0], X.shape[1]
    alpha = np.zeros((n))
    C = 1.0
    kernels = {
        'linear' : kernel_linear,
        'quadratic' : kernel_quadratic
    }
    kernel = kernels['linear']
    count = 0
    epsilon=0.001
    max_iter=10000
    while True:
        count += 1
        alpha_prev = np.copy(alpha)
        for j in range(0, n):
            i = get_rnd_int(0, n-1, j) # Get random int i~=j
            x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
            k_ij = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
            if k_ij == 0:
                continue
            alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
            (L, H) = compute_L_H(C, alpha_prime_j, alpha_prime_i, y_j, y_i)

            # Compute model parameters
            w = calc_w(alpha, y, X)
            b = calc_b(X, y, w)

            # Compute E_i, E_j
            E_i = E(x_i, y_i, w, b)
            E_j = E(x_j, y_j, w, b)

            # Set new alpha values
            alpha[j] = alpha_prime_j + float(y_j * (E_i - E_j))/k_ij
            alpha[j] = max(alpha[j], L)
            alpha[j] = min(alpha[j], H)

            alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

        # Check convergence
        diff = np.linalg.norm(alpha - alpha_prev)
        if diff < epsilon:
            break

        if count >= max_iter:
            print("Iteration number exceeded the max of %d iterations" % (max_iter))
            return
    # Compute final model parameters
    b = calc_b(X, y, w)
    if self.kernel_type == 'linear':
        w = calc_w(alpha, y, X)
    # Get support vectors
    alpha_idx = np.where(alpha > 0)[0]
    support_vectors = X[alpha_idx, :]

    print(predict(X, w, b))

    return support_vectors, count

def predict(X, w, b):
    return h(X, w, b)

def calc_b(X, y, w):
    b_tmp = y - np.dot(w.T, X.T)
    return np.mean(b_tmp)

def calc_w(alpha, y, X):
    return np.dot(X.T, np.multiply(alpha,y))

# Prediction
def h(X, w, b):
    return np.sign(np.dot(w.T, X.T) + b).astype(int)

# Prediction error
def E(x_k, y_k, w, b):
    return h(x_k, w, b) - y_k

def compute_L_H(C, alpha_prime_j, alpha_prime_i, y_j, y_i):
    if(y_i != y_j):
        return (max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j))
    else:
        return (max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j))

def get_rnd_int(a,b,z):
    i = z
    cnt=0
    while i == z and cnt<1000:
        i = rnd.randint(a,b)
        cnt=cnt+1
    return i

# Define kernels
def kernel_linear(x1, x2):
    return np.dot(x1, x2.T)
def kernel_quadratic(x1, x2):
    return (np.dot(x1, x2.T) ** 2)

#CLASSIFY ENDS

def avgProbability(train):

    trueSum = 0
    falseSum = 0

    for item in train.T[-1][1:]:
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

#AGE BINNING
for i, item in enumerate(trainData[1:]):
    if int(item[0]) >= 0 and int(item[0]) < 10:
        trainData[i+1][0] = '0-9'
    elif int(item[0]) >= 10 and int(item[0]) < 20:
        trainData[i+1][0] = '10-19'
    elif int(item[0]) >= 20 and int(item[0]) < 30:
        trainData[i+1][0] = '20-29'
    elif int(item[0]) >= 30 and int(item[0]) < 40:
        trainData[i+1][0] = '30-39'
    elif int(item[0]) >= 40 and int(item[0]) < 50:
        trainData[i+1][0] = '40-49'
    elif int(item[0]) >= 50 and int(item[0]) < 60:
        trainData[i+1][0] = '50-59'
    elif int(item[0]) >= 60 and int(item[0]) < 70:
        trainData[i+1][0] = '60-69'
    elif int(item[0]) >= 70:
        trainData[i+1][0] = '>=70'

#Gain binnigng
for i, item in enumerate(trainData[1:]):
    if int(item[10]) == 0:
        trainData[i+1][10] = 'None'
    else:
        trainData[i+1][10] = 'Gain'

#Loss binninig
for i, item in enumerate(trainData[1:]):
    if int(item[11]) == 0:
        trainData[i+1][11] = 'None'
    else:
        trainData[i+1][11] = 'Loss'

#Hours per week binnning
for i, item in enumerate(trainData[1:]):
    if int(item[12]) >= 0 and int(item[12]) < 20:
        trainData[i+1][12] = '0-19'
    elif int(item[12]) >= 20 and int(item[12]) < 35:
        trainData[i+1][12] = '20-24'
    elif int(item[12]) >= 35 and int(item[12]) < 50:
        trainData[i+1][12] = '35-49'
    elif int(item[12]) >= 50:
        trainData[i+1][12] = '>=50'

#Earnings
trainData2 = trainData.copy()
trainData2 = trainData2[1:]
for i, item in enumerate(trainData2[:]):
    if '<=50K' in item[14]:
        trainData2[i][14] = 0
    else:
        trainData2[i][14] = 1

print(trainData2[0:5])
print('\n\n')
# Average for each feature independently: workclass, education, marital-status, occupation, relationship, race, sex,
# native-country
avgList = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'earnings']
avgs =  {}

# creates dictionary, each containing the header as a key with the value as an array of possible characteristics

for i, item in enumerate(trainData.T):

    if headers[i] in avgList:
        avgs[headers[i]] = []

        for info in item:

            if info not in avgs[headers[i]]:

                if '?' not in info and info not in headers[i]:
                    avgs[headers[i]].append(info)



print(avgs)

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

#print(trainData2.T[-1])
#print(trainData2[:][:-1])
fit(trainData2, trainData2.T[-1])

for item in trainData.T:

    yList = []

    for av in probs[item[0]].keys():
        print('avg : ', probs[item[0]][av][0])
        yList.append(probs[item[0]][av][0])
        print(item[0])
    barGraph(probs[item[0]].keys(), yList, str(item[0]))
'''
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
#print('ytest : ')
print(ytest)

svclassifier = SVC(kernel='linear')
svclassifier.fit(list(xtrain.values)[:][:], list(ytrain.values)[:])
ypred = svclassifier.predict(list(xtest.values))

print(len(ypred[:]))
#print(list(ypred[:10]))
print(len(list(ytest[:])))
#print(list(ytest[:10]))
skAccuracy = getAccuracy(ypred[:], list(ytest[:]))
print('Accuracy', str(skAccuracy))
'''
#print(dfTrain)

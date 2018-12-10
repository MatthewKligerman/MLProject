import csv
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import random as rnd
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
    plt.tick_params(labelsize=6)
    plt.ylabel('Probability')
    plt.xlabel(category)
    plt.title('Probability of making >$50k for each '+category)

    plt.show()

    return

def getLikelihood(train, header):
    return 0


#CLASSIFY STARTS
def fit_data(data, y, k = 'linear'):
    # Initialization
    num = data.shape[0]

    alpha = np.zeros((num))
    penalty = 1.0
    kernel_type = k
    
    kernels = {
        'linear' : kernel_linear,
        'quadratic' : kernel_quadratic
    }
    
    kernel = kernels[kernel_type]
    count = 0
    step = 0.001
    loops =10000

    while True:

        count += 1
        previous_alpha = np.copy(alpha)

        for j in range(0, num):

            i = random(0, num-1, j)
            x_i = data[i,:]
            x_j = data[j,:]
            y_i = y[i]
            y_j = y[j]

            kernel_trick = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
            
            if kernel_trick == 0:
                continue
            
            hyper_alpha_j = alpha[j]
            hyper_alpha_i = alpha[i]
            (L, H) = get_LH(penalty, hyper_alpha_j, hyper_alpha_i, y_j, y_i)

            #Creating the model
            weights = get_weights(alpha, y, data)
            b = get_b(data, y, weights)

            #obtaining errors
            error_i = error(x_i, y_i, weights, b)
            error_j = error(x_j, y_j, weights, b)

            #obtaining new alpha values
            alpha[j] = hyper_alpha_j + float(y_j * (error_i - error_j))/kernel_trick
            alpha[j] = max(alpha[j], L)
            alpha[j] = min(alpha[j], H)
            alpha[i] = hyper_alpha_i + y_i * y_j * (hyper_alpha_j - alpha[j])

        diff = np.linalg.norm(alpha - previous_alpha)
        if diff < step:
            break
        if count >= loops:
            return

    b = get_b(data, y, weights)
    if kernel_type == 'linear':
        weights = get_weights(alpha, y, data)

    alpha_loc = np.where(alpha > 0)[0]
    support_vectors = data[alpha_loc, :]

    data_pred = predict(data, weights, b)

    for i, element in enumerate(data_pred):
        if element == -1:
            data_pred[i] = 0

    print('Prediction Values:', data_pred)

    new_y = []
    for i, element in enumerate(y):
        if element == 1:
            new_y.append('>50K')
        else:
            new_y.append('<=50K')


    accuracy = getAccuracy(data_pred, new_y)

    if k == 'linear':
        print('Classify Linear Accuracy:', accuracy)
    else:
        print('Classify Quadratic Accuracy:', accuracy)
    return support_vectors, count

#Get the prediction values
def predict(data, weights, b):
    return predict_calc(data, weights, b)

#Calculate b
def get_b(data, y, weights):
    new_b = y - np.dot(weights.T, data.T)
    return np.mean(new_b)

#Calculate the weights
def get_weights(alpha, y, data):
    return np.dot(data.T, np.multiply(alpha,y))

#Calculate the prediction values
def predict_calc(data, weights, b):
    return np.sign(np.dot(weights.T, data.T) + b).astype(int)

#Calculate the error
def error(x_kernel, y_kernel, weights, b):
    return predict_calc(x_kernel, weights, b) - y_kernel

#Calculate formula
def get_LH(penalty, hyper_alpha_j, hyper_alpha_i, y_j, y_i):
    if(y_i != y_j):
        return (max(0, hyper_alpha_j - hyper_alpha_i), min(penalty, penalty - hyper_alpha_i + hyper_alpha_j))
    else:
        return (max(0, hyper_alpha_i + hyper_alpha_j - penalty), min(penalty, hyper_alpha_i + hyper_alpha_j))

#create a random number
def random(a,b,c):
    i = c
    count = 0
    while i == c and count<1000:
        i = rnd.randint(a,b)
        count += 1
    return i

#Kernel trick for linear function
def kernel_linear(x1, x2):
    return np.dot(x1, x2.T)

#Kernel trick for quadratic function
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
#fit(trainData2, trainData2.T[-1])
''''
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

for column in list(dfTrain.columns):
    if column in avgList:
        for i, item in enumerate(dfTrain[column]):
            dfTrain[column][i] = probs[column][item][0]  


trainData2 = dfTrain.values
print(trainData2.T[-1])
fit_data(trainData2.astype(float), trainData2.T[-1].astype(float), 'linear')
fit_data(trainData2.astype(float), trainData2.T[-1].astype(float), 'quadratic')
print()
print()


xtrain = dfTrain.drop('earnings', axis=1)
ytrain = dfTrain['earnings']

for column in list(dfTest.columns)[:-1]:
    if column in avgList:
        for i, item in enumerate(dfTest[column]):
            if item not in probs[column].keys():
                dfTest[column][i] = probs['other']
            else:
                dfTest[column][i] = probs[column][item][0]

xtest = dfTest.drop('earnings', axis=1)
ytest = dfTest['earnings']

svclassifier = SVC(kernel='linear')
svclassifier.fit(list(xtrain.values)[:][:], list(ytrain.values)[:])
ypred = svclassifier.predict(list(xtest.values))

skAccuracy = getAccuracy(ypred[:], list(ytest[:]))
print('Learned Linear Accuracy:', str(skAccuracy))

svclassifier = SVC(kernel='poly', degree = 2)
svclassifier.fit(list(xtrain.values)[:][:], list(ytrain.values)[:])
ypred = svclassifier.predict(list(xtest.values))

skAccuracy = getAccuracy(ypred[:], list(ytest[:]))
print('Learned Quadratic Accuracy:', str(skAccuracy))

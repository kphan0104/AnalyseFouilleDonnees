#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

##### Génération de données #####

np.random.seed(123)

def generateData(mu, sigma, nb, classId, dataList, labelList):
	data = np.random.multivariate_normal(mu, sigma, nb).tolist()
	for i in range (0, nb):
		dataList.append(data[i])
	for i in range (0, nb):
		labelList.append(classId)

def getX(list):
	return [x[0] for x in list]

def getY(list):
	return [y[1] for y in list]

def getData(classId, dataList, labelList):
	labelList = np.array(labelList)
	return [dataList[i] for i in np.where(labelList == classId)[0]]

mu0 = (0,0)
mu1 = (3,2)
sigma = [[1,1/2],[1/2,1]]

learningData = []
learningLabel = []
generateData(mu0, sigma, 10, 0, learningData, learningLabel)
generateData(mu1, sigma, 10, 1, learningData, learningLabel)

testData = []
testLabel = []
generateData(mu0, sigma, 1000, 0, testData, testLabel)
generateData(mu1, sigma, 1000, 1, testData, testLabel)


#plt.scatter(getX(getData(0, learningData, learningLabel)), getY(getData(0, learningData, learningLabel)), c = "b")
#plt.scatter(getX(getData(1, learningData, learningLabel)), getY(getData(1, learningData, learningLabel)), c = "r")
plt.scatter(getX(getData(0, testData, testLabel)), getY(getData(0, testData, testLabel)), c = "b")
plt.scatter(getX(getData(1, testData, testLabel)), getY(getData(1, testData, testLabel)), c = "r")
plt.show()



##### Analyse Discriminante Linéaire #####

def muhat(classId, dataList, labelList):
	dataClass = getData(classId, dataList, labelList)
	return [sum(getX(dataClass)) / len(dataClass), sum(getY(dataClass)) / len(dataClass)]

def sigmahat(classId, dataList, labelList):
	mu = np.array([muhat(classId, dataList, labelList)])
	dataClass = getData(classId, dataList, labelList)
	matrixList = [np.dot(np.transpose(np.array([x]) - mu), np.array([x]) - mu) for x in dataClass]
	return sum(matrixList)/len(dataClass)

def pihat(classId, dataList, labelList):
	dataClass = getData(classId, dataList, labelList)
	return len(dataClass) / len(dataList)

def weighted_Sigma_Hat (sigma0, sigma1, nb_obs0, nb_obs1) :
    return (nb_obs0*sigma0 + nb_obs1*sigma1)/(nb_obs0+nb_obs1)



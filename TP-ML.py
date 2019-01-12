import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

##### Génération de données #####

np.random.seed(123)

mu0 = (0,0)
mu1 = (3,2)
sigma = [[1,1/2],[1/2,1]]

def getData(mu0, mu1, sigma0, sigma1, nb, percentageC0):
    
    labels = np.concatenate((np.repeat(0,int(nb * percentageC0)) , np.repeat(1,int(nb * (1 - percentageC0)))), axis = None)
    c0Data = np.random.multivariate_normal(mu0, sigma0, int(nb * percentageC0))
    c1Data = np.random.multivariate_normal(mu1, sigma1, int(nb * (1 - percentageC0)))
    cData  = np.concatenate((c0Data, c1Data), axis = 0)
    return([cData, labels])

mu0, mu1, sigma0, sigma1= mu0, mu1, sigma, sigma

trainData, trainLabel = getData(mu0, mu1, sigma0, sigma1, 20, 0.5)
testData, testLabel = getData(mu0, mu1, sigma0, sigma1, 2000, 0.5)

plt.scatter(testData[:,0], testData[:,1], c = testLabel)
#plt.scatter(getX(getData(1, testData, testLabel)), getY(getData(1, testData, testLabel)), c = "r")
plt.show()

##### Analyse Discriminante Linéaire #####

def muhat(classId, data, labelList):
    dt  = data[labelList == classId]
    res = np.sum(dt, axis=0)/len(dt)
    return(res)

def sigmahat(classId, data, labelList):
    mu = muhat(classId, data, labelList)
    dt = data[labelList == classId]
    matrix = [np.dot(np.transpose(np.array([x]) - mu), np.array([x]) - mu) for x in dt]
    return (sum(matrix)/len(dt))

def pihat(classId, data, labelList):
	dt = data [labelList == classId]
	return len(dt)/len(data)

def weighted_Sigma_Hat (sigma0, sigma1, nb_obs0, nb_obs1) :
    return (nb_obs0*sigma0 + nb_obs1*sigma1)/(nb_obs0+nb_obs1)

#InvertedWeighted_Sigma_Hat = np.linalg.inv(weighted_Sigma_Hat)


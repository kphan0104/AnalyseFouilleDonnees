import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import sklearn as sk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis




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

def LDA(x, data, labelList) :
    #moyenne
    mu0 = muhat(0, data, labelList)
    mu1 = muhat(1, data, labelList)
    #covariance
    sigma0 = sigmahat(0, data, labelList)
    sigma1 = sigmahat(1, data, labelList)
    
    #covariance pondérée
    sigma = weighted_Sigma_Hat(sigma0, sigma1, 10,10)
    
    #proportion des classes
    pi0 = pihat(0, data, labelList)
    pi1 = pihat(1, data, labelList)

    delta0 = (np.dot(np.transpose(x), np.dot(np.linalg.inv(sigma), mu0))) - 0.5 * (np.dot(np.transpose(mu0),np.dot(np.linalg.inv(sigma), mu0))) + np.log10(pi0)
    delta1 = (np.dot(np.transpose(x),np.dot(np.linalg.inv(sigma), mu1))) - 0.5 * (np.dot(np.transpose(mu1),np.dot(np.linalg.inv(sigma), mu1))) + np.log10(pi1)
    
    #décision
    if (delta1 > delta0) :
        return 1
    else :
        return 0

def classificationRate(data, labelList, nb_obs0):
    rightPrediction = 0
        
    for i in range(0, nb_obs0) :
        if(LDA (np.transpose(np.asmatrix(data[i])), data, labelList) == 0) :
            rightPrediction += 1
    for i in range(nb_obs0,len(data)) :
        if(LDA (np.transpose(np.asmatrix(data[i])), data, labelList) == 1) :
            rightPrediction += 1
    
    rate = rightPrediction/len(data)
    return rate

def classificationRateUsingSklearn(data, labelList) :
    rightPrediction = 0
    clf = DiscrimantLinearAnalysis()
    clf.fit(data, labelList)
    for i in range(0, nb_obs0) :
        if(clf.predict(np.asmatrix(data[i])) == 0) :
            rightPrediction += 1
    for i in range(nb_obs0,len(data)) :
        if(clf.predict(np.asmatrix(data[i])) == 1) :
            rightPrediction+= 1
    rate = rightPrediction/len(data)
    return rate

print("LDA Apprentissage :",classificationRate(trainData, trainLabel, 10))
print("LDA Test : ", classificationRate(testData, testLabel, 1000))

#Tracer la frontière de décision 

def decisionBoudary(data, labelList):
    #means
    mu0 = muhat(0, data, labelList)
    mu1 = muhat(1, data, labelList)
    #proportion of classes
    pi0 = pihat(0, data, labelList)
    pi1 = pihat(1, data, labelList)
    #covariance
    sigma0 = sigmahat(0, data, labelList)
    sigma1 = sigmahat(1, data, labelList)
    #weighted covariance 
    sigma = weighted_Sigma_Hat(sigma0, sigma1, 10,10)
    #x coefficient
    w = np.dot(np.linalg.inv(sigma),(mu0-mu1))
    #w coordinates 
    alpha = w[0]
    beta = w[1]
    #b 
    b = -0.5 * np.dot((mu0-mu1), np.dot(np.linalg.inv(sigma),(mu0+mu1))) + np.log10(pi0/pi1)
    #x coordinates 
    return [[0,-b/beta],[-b/alpha, 0]]

def decisionBoudary(data, labelList):
    #means
    mu0 = muhat(0, data, labelList)
    mu1 = muhat(1, data, labelList)
    #proportion of classes
    pi0 = pihat(0, data, labelList)
    pi1 = pihat(1, data, labelList)
    #covariance
    sigma0 = sigmahat(0, data, labelList)
    sigma1 = sigmahat(1, data, labelList)
    #weighted covariance 
    sigma = weighted_Sigma_Hat(sigma0, sigma1, 10,10)
    #x coefficient
    w = np.dot(np.linalg.inv(sigma),(mu0-mu1))
    #w coordinates 
    alpha = w[0]
    beta = w[1]
    #b 
    b = -0.5 * np.dot((mu0-mu1), np.dot(np.linalg.inv(sigma),(mu0+mu1))) + np.log10(pi0/pi1)
    #x coordinates 
    return [[0,-b/beta],[-b/alpha, 0]]

def drawDecisionBoundary(data, labelList) :
    p1,p2    = decisionBoudary(data, labelList)
    x_list   = [p1[0], p2[0]]
    y_list   = [p1[1], p2[1]]
    x_list2  = [p2[0], 2*p2[0] - p1[0]]
    y_list2  = [p2[1], 2*p2[1] - p1[1]]
    fig, ax = plt.subplots()
    ax.scatter(data[:,0], data[:,1], c = labelList)
    line = mlines.Line2D(x_list, y_list, color='black')
    line2 = mlines.Line2D(x_list2, y_list2, color='black')
    ax.add_line(line)
    ax.add_line(line2)
    plt.show()

    
    
    
    

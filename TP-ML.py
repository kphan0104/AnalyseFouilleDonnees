import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
import sklearn as sk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



##### I\ Génération de données #####

np.random.seed(123)

mu0 = (0,0)
mu1 = (3,2)
sigma = [[1,1/2],[1/2,1]]

def getData(mu0, mu1, sigma0, sigma1, nbObs0, nbObs1):
    labels = np.concatenate((np.repeat(0,nbObs0) , np.repeat(1,nbObs1)), axis = None)
    c0Data = np.random.multivariate_normal(mu0, sigma0, nbObs0)
    c1Data = np.random.multivariate_normal(mu1, sigma1, nbObs1)
    cData  = np.concatenate((c0Data, c1Data), axis = 0)
    return([cData, labels])

mu0, mu1, sigma0, sigma1= mu0, mu1, sigma, sigma

trainData, trainLabel = getData(mu0, mu1, sigma0, sigma1, 10, 10)
testData, testLabel = getData(mu0, mu1, sigma0, sigma1, 1000, 1000)

#plt.scatter(testData[:,0], testData[:,1], c = testLabel)
#plt.show()

##### II\ Analyse Discriminante Linéaire #####

#Qu1 : 

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

def weighted_Sigma_Hat (sigma0, sigma1, nbObs0, nbObs1) :
    return np.dot((np.dot(nbObs0,sigma0) + np.dot(nbObs1,sigma1)), 1/(nbObs0+nbObs1))

def LDA(x):
    #moyenne
    mu0 = muhat(0, trainData, trainLabel)
    mu1 = muhat(1, trainData, trainLabel)
    #covariance
    sigma0 = sigmahat(0, trainData, trainLabel)
    sigma1 = sigmahat(1, trainData, trainLabel)
    
    #covariance pondérée
    sigma = weighted_Sigma_Hat(sigma0, sigma1, len(trainLabel == 0),len(trainLabel == 1))
    
    #proportion des classes
    pi0 = pihat(0, trainData, trainLabel)
    pi1 = pihat(1, trainData, trainLabel)

    delta0 = (np.dot(np.transpose(x), np.dot(np.linalg.inv(sigma), mu0))) - 0.5 * (np.dot(np.transpose(mu0),np.dot(np.linalg.inv(sigma), mu0))) + np.log10(pi0)
    delta1 = (np.dot(np.transpose(x),np.dot(np.linalg.inv(sigma), mu1))) - 0.5 * (np.dot(np.transpose(mu1),np.dot(np.linalg.inv(sigma), mu1))) + np.log10(pi1)
    
    #décision
    if (delta1 > delta0) :
        return 1
    else :
        return 0

    
def classificationRate(data, labelList):
    rightPrediction = 0

    for i in range(0, len(data)) :
        if(LDA (np.transpose(np.asmatrix(data[i]))) == labelList[i]) :
            rightPrediction += 1
    
    rate = rightPrediction/len(data)
    return rate

def classificationRateUsingSklearn(data, labelList) :
    rightPrediction = 0
    
    clf = LinearDiscriminantAnalysis()
    clf.fit(trainData, trainLabel).score(data, labelList)

    for i in range(0, len(data)) :
        if(clf.predict(np.asmatrix(data[i])) == labelList[i]) :
            rightPrediction += 1

    rate = rightPrediction/len(data)
    return rate

print("LDA Apprentissage Rate: ",classificationRate(trainData, trainLabel))
print("LDA Test Rate: ", classificationRate(testData, testLabel))
print("LDA SKLEARN Apprentissage Rate: ", classificationRateUsingSklearn(trainData, trainLabel))
print("LDA SKLEARN Test Rate: ", classificationRateUsingSklearn(testData, testLabel))

##############################################################################
#Qu2 : 

#Ajout du point (-10, -10)

trainData[0] = [-10]

#Calcul des paramètres à nouveau

#means
muhat0 = muhat(0, trainData, trainLabel)
#print(muhat0)

#covariance
sigma0 = sigmahat(0, trainData, trainLabel)
#print(sigma0)

#covariance pondérée
sigma = weighted_Sigma_Hat(sigma0, sigma1, len(trainLabel == 0),len(trainLabel == 1))
#print(sigma)
print("LDA Apprentissage Rate après l'ajout du point (-10, -10) : ",classificationRate(trainData, trainLabel))
print("LDA Test Rate du point (-10, -10): ", classificationRate(testData, testLabel))

#Commentaire : 

#Le point aberrant (-10,-10) influence les paramètres de la LDA, cela diminue les performances et l'efficacité de notre méthode LDA (sensible aux données aberrantes)
##############################################################################

#Qu3 Tracer la frontière de décision 


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
    p1,p2    = decisionBoudary(trainData, trainLabel)
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
    
#Cas sans la donnée aberrante 
trainData[0] = [0.44151096, 1.4388564] 
decisionBoudary(trainData, trainLabel)
drawDecisionBoundary(testData, testLabel)
#Cas avec la donnée aberrante 
trainData[0] = [-10]
decisionBoudary(trainData, trainLabel)
drawDecisionBoundary(testData, testLabel)

#Commentaire : 
#Le point aberrant modifie complétement la direction de la frontière de décision.
#On remarque que la frontière de décision avec la valeur aberrante n'est 
#pas bonne en rajoutant les données de test (voir la figure)


##############################################################################@
#Qu4

### Les performances de l'analyse discriminante linéiare vont décroitre lorsque :
## - le nombre d'observations dans les données d'apprentissage est faible
## - les matrices de covariance entre les classes sont très différentes
### Cette méthode généralise LDA pour lambda = 1 (et sigma1=sigma2)


###################################





#print(classificationRateBis(testData, testLabel))

##############################################################################@
#Qu5
def variantWeightedSigmaHat (sigma0, sigma1, nbObs0, nbObs1, param) :
    return np.dot(param, np.dot((np.dot(nbObs0,sigma0) + np.dot(nbObs1,sigma1)), 1/(nbObs0+nbObs1))) + np.dot((1-param), np.identity(2))

#
def variantLDA(x, param):
    #moyenne
    mu0 = muhat(0, trainData, trainLabel)
    mu1 = muhat(1, trainData, trainLabel)
    #covariance
    sigma0 = sigmahat(0, trainData, trainLabel)
    sigma1 = sigmahat(1, trainData, trainLabel)
    
    #covariance pondérée
    sigma = variantWeightedSigmaHat(sigma0, sigma1, len(trainLabel == 0),len(trainLabel == 1), param)
    
    #proportion des classes
    pi0 = pihat(0, trainData, trainLabel)
    pi1 = pihat(1, trainData, trainLabel)

    delta0 = (np.dot(np.transpose(x), np.dot(np.linalg.inv(sigma), mu0))) - 0.5 * (np.dot(np.transpose(mu0),np.dot(np.linalg.inv(sigma), mu0))) + np.log10(pi0)
    delta1 = (np.dot(np.transpose(x),np.dot(np.linalg.inv(sigma), mu1))) - 0.5 * (np.dot(np.transpose(mu1),np.dot(np.linalg.inv(sigma), mu1))) + np.log10(pi1)
    
    #décision
    if (delta1 > delta0) :
        return 1
    else :
        return 0

def classificationRateBis2(data, labelList, param):
    rightPrediction = 0

    for i in range(0, len(data)) :
        if(variantLDA(np.transpose(np.asmatrix(data[i])), param) == labelList[i]) :
            rightPrediction += 1
    
    rate = rightPrediction/len(data)
    return rate

#print(classificationRateBis2(testData, testLabel, 0.5))
#print(classificationRateBis2(testData, testLabel, 1)) ## classic LDA


def variantLDAGraph():
    graph = []
    for i in range (0, 101) : 
        graph.append([i*0.01, classificationRateBis2(testData, testLabel, i*0.01)])
    numpyGraph = np.array(graph)
    plt.scatter(numpyGraph[:,0], numpyGraph[:,1])
    plt.show()

#variantLDAGraph()
##############################################################################@
#Qu 6
def crossValidation(dataList, labelList) :
    #teta i : lambda entre 0 et 1
    teta                     = np.linspace(0,1,10)
    tetaClassificationRate = np.empty(len(teta))
    
    for i in range (0, dataList.shape[0]) :
        test_data  = dataList[i]
        test_label = [labelList[i]]
        train_data = np.delete(dataList, i, axis = 0)
        label_data = np.delete(labelList, i)
        for teta_I in range (0, len(teta)) :
               tetaClassificationRate[teta_I] = tetaClassificationRate[teta_I] + classificationRate(np.asmatrix(test_data), test_label)
            
    tetaClassificationRate = tetaClassificationRate/dataList.shape[0]
    print("Taux de bonne classification teta = ", teta, " taux : ", tetaClassificationRate)
            
        
######## III/ A VOUS DE JOUER ########################

#Qu1 : 
#1er classifieur : SVM
#Il s'agit à peu prêt du même concept que LDA : pour chaque entrée donnée, on doit être capable de prédire
#si ce nouveau point fait partie de la classe 0 ou de a classe 1.
#On choisit une frontière qui va séparer les catégories de nos points 
#Le SVM est une méthode de classification très performante quand on dispose de peu de données d'entraînement. 
#2ème classifieur : Random Forest

from sklearn.datasets.samples_generator import make_moons
from sklearn.ensemble import RandomForestClassifier
##############################################################################@

#Qu2 :
#generate dataset

trainData, trainLabel = make_moons(n_samples=100)
testData, testLabel = make_moons(n_samples=100)

#Performing LDA
clfLDA = LinearDiscriminantAnalysis()
print("LDA performance train data : ",clfLDA.fit(trainData, trainLabel).score(trainData, trainLabel))
print("LDA performance test data: ",clfLDA.fit(trainData, trainLabel).score(testData, testLabel))

#Performing SVM
clfSVM = svm.SVC(gamma='scale')
print("SVM performance train data : "clfSVM.fit(trainData, trainLabel).score(trainData, trainLabel))
print("SVM performance test data : "clfSVM.fit(trainData, trainLabel).score(testData, testLabel))

#Performing The Random Forest Algorithm
clfRFC = RandomForestClassifier(n_estimators=10, max_depth=2,random_state=0)
print("Random Forest Class. performance train data : "clfRFC.fit(trainData, trainLabel).score(trainData, trainLabel))
print("Random Forest Class. performance test data : "clfRFC.fit(trainData, trainLabel).score(testData, testLabel))

##############################################################################@

display(trainData, trainLabel)

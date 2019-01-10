import numpy as np
#import matplotlib.pyplot as plt
import sklearn as sk

# Génération de données
mu0 = (0,0)
mu1 = (3,2)
sigma = [[1,1/2],[1/2,1]]

np.random.seed(123)

#def generateData(mu, sigma, nb, classId, listData, labelList):
#	for i in range (0, nb):
#		labelList.append(classId)
#	listData = np.concatenate((listData, np.random.multivariate_normal(mu, sigma, nb)), axis=0)
#
#app = []
#appLabel = []
#generateData(mu0, sigma, 10, 0, app, appLabel)
#print(app)
#print(appLabel)
#generateData(mu1, sigma, 10, 1, app, appLabel)
#print(app)
#print(appLabel)




c0App = np.random.multivariate_normal(mu0, sigma, 10)
c1App = np.random.multivariate_normal(mu1, sigma, 10)
app = np.concatenate([c0App,c1App])

c0Test = np.random.multivariate_normal(mu0, sigma, 1000)
c1Test = np.random.multivariate_normal(mu1, sigma, 1000)
test = np.concatenate([c0Test,c1Test])

c0AppX = np.array([x[0] for x in c0App])
#print("c0AppX")
#print(c0AppX)
c0AppY = [x[1] for x in c0App]
#print("c0AppY")
#print(c0AppY)

c0TestX = [x[0] for x in c0Test]
c0TestY = [x[1] for x in c0Test]

c1AppX = [x[0] for x in c1App]
c1AppY = [x[1] for x in c1App]

c1TestX = [x[0] for x in c1Test]
c1TestY = [x[1] for x in c1Test]


#plt.scatter(c0AppX, c0AppY)
#plt.scatter(c1AppX, c1AppY)
#plt.scatter(c0TestX, c0TestY)
#plt.scatter(c1TestX, c1TestY)

#plt.show()

# Analyse Discriminante Linéaire

muhat0App = np.array([sum(c0AppX) / len(c0App), sum(c0AppY) / len(c0App)])
print("muhat0App")
print(muhat0App)
#print("sum(c0AppX)")
#print(sum(c0AppX))
#print("sum(c0AppY)")
#print(sum(c0AppY))

muhat1App = [sum(c1AppX) / len(c1App), sum(c1AppY) / len(c1App)]
#print(muhat1App)
pihat0App = len(c0App) / len(app)
pihat1App = len(c1App) / len(app)
print("c0App")
print(c0App)
temp = (c0App - muhat0App)
npTemp = np.array(temp)[np.newaxis]

print("temp")
print(npTemp)
matrix = np.mat('[0 0 ; 0 0]')
for i in range (0, len(npTemp)):
	print("temp")
	print(npTemp[i].reshape((2,1)))
	print("transposeTemp")
	print(npTemp[i].T)
	print(np.dot(temp[i], np.transpose(temp[i])))
	np.add(matrix, np.dot(temp[i], np.transpose(temp[i])))
print("matrix")
print(matrix)

#Génération de données
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd

def generating_data(mu, sigma, nb_obs) :
	np.random.seed(123)
	x = np.random.multivariate_normal(mu, sigma, nb_obs)
	return x
#seed
mu0 = np.array([0,0])
mu1 = np.array([3,2])
sigma = np.array([[1,1/2],[1/2,1]])
c0 = generating_data(mu0,sigma,10)
c1 = generating_data(mu1,sigma,10)
c = np.concatenate([c0,c1])
print(c)
c0Test = generating_data(mu0,sigma,1000)
c1Test = generating_data(mu1,sigma,1000)
cTest = c0Test + c1Test

c0X = [x[0] for x in c0]
c0Y = [y[1] for y in c0]
c1X = [x[0] for x in c1]
c1Y = [y[1] for y in c1]

c0TestX = [x[0] for x in c0Test]
c0TestY = [y[1] for y in c0Test]
c1TestX = [x[0] for x in c1Test]
c1TestY = [y[1] for y in c1Test]

plt.scatter(c0TestX,c0TestY)
plt.scatter(c1TestX,c0TestY)
plt.title("Scatter plot")
plt.savefig("Scatter plot.png")
plt.show()
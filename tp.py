import numpy
import matplotlib.pyplot as plt
import sklearn
import itertools as it

mu0 = (0,0)
mu1 = (3,2)
sigma = [[1,1/2],[1/2,1]]

numpy.random.seed(123)

c0App = numpy.random.multivariate_normal(mu0, sigma, 10)
c1App = numpy.random.multivariate_normal(mu1, sigma, 10)
app
c0Test = numpy.random.multivariate_normal(mu0, sigma, 1000)
c1Test = numpy.random.multivariate_normal(mu1, sigma, 1000)
test = c0Test + c1Test 



c0AppX = [x[0] for x in c0App]
c0AppY = [x[1] for x in c0App]


c0TestX = [x[0] for x in c0Test]
c0TestY = [x[1] for x in c0Test]

c1AppX = [x[0] for x in c1App]
c1AppY = [x[1] for x in c1App]

c1TestX = [x[0] for x in c1Test]
c1TestY = [x[1] for x in c1Test]

plt.scatter(c0AppX, c0AppY)
plt.scatter(c1AppX, c1AppY)
plt.scatter(c0TestX, c0TestY)
plt.scatter(c1TestX, c1TestY)

plt.show()

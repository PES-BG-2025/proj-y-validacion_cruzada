import numpy as np 
import matplotlib.pyplot as plt
from sklearn import svm, datasets

iris = datasets.load_iris()

"""
    solo tomamos las primeras 2 caracteristicas 
"""
X = iris.data[:,:2]

y= iris.target

h = 0.2 
c = 1.0
#svm LINEAL 
svc = svm.SVC(kernel= "linear", C=c).fit(X,y)

svc